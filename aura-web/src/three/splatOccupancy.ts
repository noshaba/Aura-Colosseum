import { Euler, Matrix4, Vector3 } from 'three'

export type Occupancy = {
  cell_size: number
  bounds: [number, number, number, number] // xmin, xmax, zmin, zmax (Aura convention)
  floor_y: number
  occupied: [number, number][]
}

/** Conservative two-sigma spheres enclose each rotated Gaussian ellipsoid.
 * All records contribute, independently of the viewer's rendering budget. */
export function deriveSplatOccupancy(buffer: ArrayBuffer, rotation: [number, number, number], fit: boolean,
  scale: number, floorOverride: number | null, cellSize = 0.1, robotHeight = 1.4) {
  if (!buffer.byteLength || buffer.byteLength % 32) throw new Error('Invalid .splat: expected 32-byte records')
  if (!Number.isFinite(scale) || scale <= 0) throw new Error('Scene scale must be positive')
  const view = new DataView(buffer)
  const matrix = new Matrix4().makeRotationFromEuler(new Euler(...rotation))
  const points: { x: number; y: number; z: number; r: number }[] = []
  const p = new Vector3()
  for (let offset = 0; offset < buffer.byteLength; offset += 32) {
    if (view.getUint8(offset + 27) < 64) continue
    const values = Array.from({ length: 6 }, (_, j) => view.getFloat32(offset + j * 4, true))
    if (!values.every(Number.isFinite) || values.slice(3).some(v => v < 0)) continue
    p.set(values[0], values[1], values[2]).multiplyScalar(scale).applyMatrix4(matrix)
    points.push({ x: p.x, y: p.y, z: p.z, r: 2 * Math.max(...values.slice(3)) * scale })
  }
  if (points.length < 10) throw new Error('Too few visible Gaussians to estimate a floor')
  const quantile = (axis: 'x' | 'y' | 'z', q: number) => {
    const sorted = points.map(p => p[axis]).sort((a, b) => a - b)
    return sorted[Math.floor((sorted.length - 1) * q)]
  }
  // Estimate a horizontal floor from the strongest height bin in the lower 20%.
  const low = quantile('y', 0.02), high = quantile('y', 0.2)
  const bins = new Map<number, number>()
  for (const p of points) if (p.y >= low && p.y <= high) {
    const bin = Math.floor((p.y - low) / 0.05)
    bins.set(bin, (bins.get(bin) || 0) + 1)
  }
  const peak = [...bins].sort((a, b) => b[1] - a[1])[0]?.[0] || 0
  const estimatedFloor = low + (peak + 0.5) * 0.05
  const offset: [number, number, number] = fit
    ? [-(quantile('x', 0.02) + quantile('x', 0.98)) / 2, -estimatedFloor, -(quantile('z', 0.02) + quantile('z', 0.98)) / 2]
    : [0, 0, 0]
  const floor = floorOverride ?? estimatedFloor + offset[1]
  const xmin = Math.floor((quantile('x', 0.02) + offset[0]) / cellSize) * cellSize
  const xmax = Math.ceil((quantile('x', 0.98) + offset[0]) / cellSize) * cellSize
  const zmin = Math.floor((quantile('z', 0.02) + offset[2]) / cellSize) * cellSize
  const zmax = Math.ceil((quantile('z', 0.98) + offset[2]) / cellSize) * cellSize
  const nx = Math.round((xmax - xmin) / cellSize), nz = Math.round((zmax - zmin) / cellSize)
  if (nx < 10 || nz < 10 || nx * nz > 250_000) throw new Error('Scene bounds must cover at least 1 m per axis and at most 250,000 cells. Adjust scale.')
  const cells = new Set<number>()
  let work = 0
  for (const p of points) {
    const y = p.y + offset[1]
    // Remove floor centres; retain objects whose extent intersects the robot body.
    if (y <= floor + 0.08 || y - p.r > floor + robotHeight || y + p.r < floor + 0.08) continue
    const x = p.x + offset[0], z = p.z + offset[2]
    const x0 = Math.max(0, Math.floor((x - p.r - xmin) / cellSize))
    const x1 = Math.min(nx - 1, Math.floor((x + p.r - xmin) / cellSize))
    const z0 = Math.max(0, Math.floor((z - p.r - zmin) / cellSize))
    const z1 = Math.min(nz - 1, Math.floor((z + p.r - zmin) / cellSize))
    work += Math.max(0, x1 - x0 + 1) * Math.max(0, z1 - z0 + 1)
    if (work > 30_000_000) throw new Error('Gaussian extents are too large or dense. Check scene scale or use a smaller scan.')
    for (let ix = x0; ix <= x1; ix++) for (let iz = z0; iz <= z1; iz++) cells.add(ix * nz + iz)
  }
  const occupancy: Occupancy = { cell_size: cellSize, floor_y: floor, bounds: [xmin, xmax, zmin, zmax],
    occupied: [...cells].map(key => [Math.floor(key / nz), key % nz]) }
  return { occupancy, offset }
}
