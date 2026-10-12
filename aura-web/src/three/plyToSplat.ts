/**
 * Gaussian-splat PLY → 32-byte `.splat` records (position f32×3, scale f32×3, RGBA u8×4,
 * rotation u8×4), the format lightweightSplat.ts reads. Same math as the backend's
 * `_convert_gaussian_ply_to_splat`, minus its Nerfstudio Z-up axis swap: downloaded
 * PLYs use other conventions, so axes stay raw like a standard `.splat` export.
 * Only the base colour (f_dc_*) is kept; higher-order spherical harmonics are dropped.
 */

const SH_C0 = 0.28209479177387814
const REQUIRED = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2']

const TYPE_SIZE: Record<string, number> = {
  char: 1, int8: 1, uchar: 1, uint8: 1, short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8,
}

type Prop = { name: string; type: string; offset: number }

function reader(view: DataView, type: string): (at: number) => number {
  switch (type) {
    case 'float': case 'float32': return at => view.getFloat32(at, true)
    case 'double': case 'float64': return at => view.getFloat64(at, true)
    case 'char': case 'int8': return at => view.getInt8(at)
    case 'uchar': case 'uint8': return at => view.getUint8(at)
    case 'short': case 'int16': return at => view.getInt16(at, true)
    case 'ushort': case 'uint16': return at => view.getUint16(at, true)
    case 'int': case 'int32': return at => view.getInt32(at, true)
    default: return at => view.getUint32(at, true)
  }
}

export type PlyConversion = { splat: ArrayBuffer; count: number; skipped: number }

export function convertGaussianPlyToSplat(ply: ArrayBuffer): PlyConversion {
  const head = new TextDecoder('ascii').decode(new Uint8Array(ply, 0, Math.min(ply.byteLength, 64 * 1024)))
  if (!head.startsWith('ply')) throw new Error('Not a PLY file')
  const marker = head.match(/end_header\r?\n/)
  if (!marker || marker.index == null) throw new Error('PLY header has no end_header')
  const headerEnd = marker.index + marker[0].length
  const lines = head.slice(0, marker.index).split(/\r?\n/).map(l => l.trim())
  if (!lines.includes('format binary_little_endian 1.0')) {
    throw new Error('Only binary little-endian PLY is supported. Re-export it as binary, or convert it in SuperSplat.')
  }

  // The vertex element must come first: anything before it would shift the data offset.
  let vertexCount = 0, stride = 0, seenElement = false, inVertex = false
  const props: Prop[] = []
  for (const line of lines) {
    const parts = line.split(/\s+/)
    if (parts[0] === 'element') {
      if (!seenElement && parts[1] !== 'vertex') throw new Error('PLY vertex data must be the first element')
      if (seenElement && inVertex) break
      seenElement = true
      inVertex = parts[1] === 'vertex'
      vertexCount = Number(parts[2])
    } else if (inVertex && parts[0] === 'property') {
      if (parts[1] === 'list') throw new Error('PLY vertex lists are not supported: this looks like a mesh, not a Gaussian splat')
      const size = TYPE_SIZE[parts[1]]
      if (!size) throw new Error(`Unsupported PLY property type "${parts[1]}"`)
      props.push({ name: parts[2], type: parts[1], offset: stride })
      stride += size
    }
  }
  if (!vertexCount || !props.length) throw new Error('PLY has no vertex data')
  const byName = new Map(props.map(p => [p.name, p]))
  const missing = REQUIRED.filter(n => !byName.has(n))
  if (missing.length) throw new Error(`Not a Gaussian-splat PLY (missing ${missing.join(', ')}). Meshes and plain point clouds can't be converted.`)
  if (headerEnd + vertexCount * stride > ply.byteLength) throw new Error('PLY vertex data is truncated')

  const view = new DataView(ply, headerEnd)
  const field = (name: string) => {
    const p = byName.get(name)!
    const read = reader(view, p.type)
    return (row: number) => read(row * stride + p.offset)
  }
  const [x, y, z] = ['x', 'y', 'z'].map(field)
  const [s0, s1, s2] = ['scale_0', 'scale_1', 'scale_2'].map(field)
  const [c0, c1, c2] = ['f_dc_0', 'f_dc_1', 'f_dc_2'].map(field)
  const opacity = field('opacity')
  const rot = byName.has('rot_0') && byName.has('rot_1') && byName.has('rot_2') && byName.has('rot_3')
    ? ['rot_0', 'rot_1', 'rot_2', 'rot_3'].map(field) : null

  const out = new ArrayBuffer(vertexCount * 32)
  const f32 = new Float32Array(out)
  const u8 = new Uint8Array(out)
  const clampByte = (v: number) => Math.max(0, Math.min(255, Math.round(v)))
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
  let n = 0
  for (let i = 0; i < vertexCount; i++) {
    const px = x(i), py = y(i), pz = z(i)
    const sx = Math.exp(clamp(s0(i), -12, 8)), sy = Math.exp(clamp(s1(i), -12, 8)), sz = Math.exp(clamp(s2(i), -12, 8))
    const alpha = 1 / (1 + Math.exp(-clamp(opacity(i), -20, 20)))
    // Same filter as the backend: drop non-finite and fully transparent splats.
    if (!Number.isFinite(px + py + pz + sx + sy + sz) || alpha <= 1 / 255) continue
    const f = n * 8, b = n * 32
    f32[f] = px; f32[f + 1] = py; f32[f + 2] = pz
    f32[f + 3] = sx; f32[f + 4] = sy; f32[f + 5] = sz
    u8[b + 24] = clampByte((0.5 + SH_C0 * c0(i)) * 255)
    u8[b + 25] = clampByte((0.5 + SH_C0 * c1(i)) * 255)
    u8[b + 26] = clampByte((0.5 + SH_C0 * c2(i)) * 255)
    u8[b + 27] = clampByte(alpha * 255)
    if (rot) {
      const q = rot.map(r => r(i))
      const norm = Math.max(Math.hypot(q[0], q[1], q[2], q[3]), 1e-8)
      for (let k = 0; k < 4; k++) u8[b + 28 + k] = clamp(Math.floor((q[k] / norm) * 128 + 128), 0, 255)
    } else {
      u8[b + 28] = 255; u8[b + 29] = 128; u8[b + 30] = 128; u8[b + 31] = 128 // identity (w, x, y, z) = (1, 0, 0, 0)
    }
    n++
  }
  if (!n) throw new Error('PLY contains no visible splats')
  return { splat: sortByImportance(out, n), count: n, skipped: vertexCount - n }
}

/** Most visible first (volume × opacity), as the reference .splat converter writes them,
 *  so viewers that keep only the first N records still show the scene's main surfaces. */
function sortByImportance(records: ArrayBuffer, n: number): ArrayBuffer {
  const f32 = new Float32Array(records)
  const u8 = new Uint8Array(records)
  const importance = new Float32Array(n)
  for (let i = 0; i < n; i++) importance[i] = f32[i * 8 + 3] * f32[i * 8 + 4] * f32[i * 8 + 5] * u8[i * 32 + 27]
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => importance[b] - importance[a])
  const sorted = new Uint8Array(n * 32)
  for (let i = 0; i < n; i++) sorted.set(u8.subarray(order[i] * 32, order[i] * 32 + 32), i * 32)
  return sorted.buffer
}
