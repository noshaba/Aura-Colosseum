import * as THREE from 'three'

/**
 * Preview renderer for the common 32-byte `.splat` interchange format
 * (position f32×3, scale f32×3, RGBA u8×4, rotation quaternion u8×4 as w,x,y,z).
 *
 * Each splat's 3D covariance (rotation + scale) is projected to a screen-space ellipse
 * (EWA splatting, as in the reference 3DGS web viewers), and splats are depth-sorted
 * back-to-front so transparency composites correctly. It still caps the splat count,
 * renders only the base colour (no view-dependent SH), and sorts on the main thread.
 * Collision/robot training keeps using Aura's explicit geometry proxy, not the splats.
 */
export function createLightweightSplat(buffer: ArrayBuffer, maxSplats = 140_000) {
  const stride = 32
  const total = Math.floor(buffer.byteLength / stride)
  if (!total) throw new Error('This .splat file contains no 32-byte splat records.')
  const view = new DataView(buffer)

  // Keep the most visible splats (volume × opacity) rather than every Nth record, so
  // large surfaces survive the cap and tiny, faint splats are what get dropped.
  let selected: Uint32Array
  if (total <= maxSplats) {
    selected = new Uint32Array(total).map((_, i) => i)
  } else {
    const importance = new Float32Array(total)
    for (let i = 0; i < total; i++) {
      const o = i * stride
      importance[i] = Math.abs(view.getFloat32(o + 12, true) * view.getFloat32(o + 16, true) * view.getFloat32(o + 20, true)) * view.getUint8(o + 27)
    }
    const threshold = Float32Array.from(importance).sort()[total - maxSplats]
    selected = new Uint32Array(maxSplats)
    let n = 0
    for (let i = 0; i < total && n < maxSplats; i++) if (importance[i] >= threshold) selected[n++] = i
    selected = selected.subarray(0, n)
  }
  const count = selected.length

  // Source data in file order; the GPU attributes below hold the depth-sorted copy.
  const srcCenter = new Float32Array(count * 3)
  const srcCovA = new Float32Array(count * 3)
  const srcCovB = new Float32Array(count * 3)
  const srcColor = new Float32Array(count * 4)
  for (let k = 0; k < count; k++) {
    const o = selected[k] * stride
    srcCenter[k * 3] = view.getFloat32(o, true)
    srcCenter[k * 3 + 1] = view.getFloat32(o + 4, true)
    srcCenter[k * 3 + 2] = view.getFloat32(o + 8, true)
    const sx = view.getFloat32(o + 12, true), sy = view.getFloat32(o + 16, true), sz = view.getFloat32(o + 20, true)
    let w = (view.getUint8(o + 28) - 128) / 128, x = (view.getUint8(o + 29) - 128) / 128
    let y = (view.getUint8(o + 30) - 128) / 128, z = (view.getUint8(o + 31) - 128) / 128
    const qn = Math.hypot(w, x, y, z) || 1
    w /= qn; x /= qn; y /= qn; z /= qn
    // Σ = R · diag(s²) · Rᵀ with R from the (w, x, y, z) quaternion.
    const r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - w * z), r02 = 2 * (x * z + w * y)
    const r10 = 2 * (x * y + w * z), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - w * x)
    const r20 = 2 * (x * z - w * y), r21 = 2 * (y * z + w * x), r22 = 1 - 2 * (x * x + y * y)
    const a = sx * sx, b = sy * sy, c = sz * sz
    srcCovA[k * 3] = r00 * r00 * a + r01 * r01 * b + r02 * r02 * c
    srcCovA[k * 3 + 1] = r00 * r10 * a + r01 * r11 * b + r02 * r12 * c
    srcCovA[k * 3 + 2] = r00 * r20 * a + r01 * r21 * b + r02 * r22 * c
    srcCovB[k * 3] = r10 * r10 * a + r11 * r11 * b + r12 * r12 * c
    srcCovB[k * 3 + 1] = r10 * r20 * a + r11 * r21 * b + r12 * r22 * c
    srcCovB[k * 3 + 2] = r20 * r20 * a + r21 * r21 * b + r22 * r22 * c
    srcColor[k * 4] = view.getUint8(o + 24) / 255
    srcColor[k * 4 + 1] = view.getUint8(o + 25) / 255
    srcColor[k * 4 + 2] = view.getUint8(o + 26) / 255
    srcColor[k * 4 + 3] = view.getUint8(o + 27) / 255
  }

  const dynamic = (array: Float32Array, size: number) =>
    new THREE.InstancedBufferAttribute(array, size).setUsage(THREE.DynamicDrawUsage)
  const aCenter = dynamic(srcCenter.slice(), 3)
  const aCovA = dynamic(srcCovA.slice(), 3)
  const aCovB = dynamic(srcCovB.slice(), 3)
  const aColor = dynamic(srcColor.slice(), 4)

  // Quad spans ±√8 standard deviations; the fragment cuts off where the Gaussian < e⁻⁴.
  const Q = Math.sqrt(8)
  const geometry = new THREE.InstancedBufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    -Q, -Q, 0, Q, -Q, 0, Q, Q, 0,
    -Q, -Q, 0, Q, Q, 0, -Q, Q, 0,
  ], 3))
  geometry.setAttribute('iCenter', aCenter)
  geometry.setAttribute('iCovA', aCovA)
  geometry.setAttribute('iCovB', aCovB)
  geometry.setAttribute('iColor', aColor)
  geometry.instanceCount = count

  const viewport = new THREE.Vector2(1, 1)
  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.NormalBlending,
    uniforms: { viewport: { value: viewport } },
    vertexShader: `
      attribute vec3 iCenter;
      attribute vec3 iCovA;
      attribute vec3 iCovB;
      attribute vec4 iColor;
      uniform vec2 viewport;
      varying vec2 vUv;
      varying vec4 vColor;
      void main() {
        vec4 cam = modelViewMatrix * vec4(iCenter, 1.0);
        vec4 clip = projectionMatrix * cam;
        float bound = 1.2 * clip.w;
        if (cam.z > -0.01 || clip.x < -bound || clip.x > bound || clip.y < -bound || clip.y > bound) {
          gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // behind the camera or off screen: cull
          return;
        }
        // World covariance → view space (includes the mesh's orientation correction).
        mat3 W = mat3(modelViewMatrix);
        mat3 S = mat3(iCovA.x, iCovA.y, iCovA.z, iCovA.y, iCovB.x, iCovB.y, iCovA.z, iCovB.y, iCovB.z);
        mat3 V = W * S * transpose(W);
        // Jacobian of the perspective projection to pixels, one row per screen axis.
        vec2 focal = vec2(projectionMatrix[0][0], projectionMatrix[1][1]) * viewport * 0.5;
        float iz = 1.0 / cam.z;
        vec3 j0 = vec3(-focal.x * iz, 0.0, focal.x * cam.x * iz * iz);
        vec3 j1 = vec3(0.0, -focal.y * iz, focal.y * cam.y * iz * iz);
        // 2D covariance in pixels², with the usual 0.3 px² low-pass so tiny splats stay visible.
        float a = dot(j0, V * j0) + 0.3;
        float b = dot(j0, V * j1);
        float d = dot(j1, V * j1) + 0.3;
        float mid = 0.5 * (a + d);
        float radius = length(vec2(0.5 * (a - d), b));
        float l1 = mid + radius;
        float l2 = max(mid - radius, 0.1);
        vec2 e1 = abs(b) > 1e-6 ? normalize(vec2(b, l1 - a)) : (a >= d ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
        vec2 e2 = vec2(-e1.y, e1.x);
        // Axes in pixels (1 σ), capped like the reference viewers to avoid screen-filling floaters.
        vec2 major = min(sqrt(l1), 1024.0) * e1;
        vec2 minor = min(sqrt(l2), 1024.0) * e2;
        vec2 offsetPx = position.x * major + position.y * minor;
        gl_Position = vec4(clip.xy / clip.w + 2.0 * offsetPx / viewport, clip.z / clip.w, 1.0);
        vUv = position.xy;
        vColor = iColor;
      }
    `,
    fragmentShader: `
      varying vec2 vUv;
      varying vec4 vColor;
      void main() {
        float r2 = dot(vUv, vUv);
        if (r2 > 8.0) discard;
        float alpha = exp(-0.5 * r2) * vColor.a;
        if (alpha < 1.0 / 255.0) discard;
        gl_FragColor = vec4(vColor.rgb, alpha);
      }
    `,
  })

  const mesh = new THREE.Mesh(geometry, material)
  mesh.frustumCulled = false
  mesh.renderOrder = -5

  // Back-to-front depth sort: 16-bit counting sort, re-run when the view changes
  // (throttled, since each sort re-uploads the instance buffers).
  const depth = new Float32Array(count)
  const keys = new Uint16Array(count)
  const order = new Uint32Array(count)
  const counts = new Uint32Array(65536)
  const mv = new THREE.Matrix4()
  // Infinity (not NaN: NaN comparisons are always false) so the first frame registers as a change.
  const lastView = new Float32Array(4).fill(Infinity)
  let dirty = true
  let lastSort = -Infinity
  const sortBackToFront = () => {
    const e = mv.elements
    let min = Infinity, max = -Infinity
    for (let k = 0; k < count; k++) {
      const z = e[2] * srcCenter[k * 3] + e[6] * srcCenter[k * 3 + 1] + e[10] * srcCenter[k * 3 + 2] + e[14]
      depth[k] = z
      if (z < min) min = z
      if (z > max) max = z
    }
    // View-space z is negative in front of the camera: smallest z = farthest = drawn first.
    const scale = max > min ? 65535 / (max - min) : 0
    counts.fill(0)
    for (let k = 0; k < count; k++) { keys[k] = (depth[k] - min) * scale; counts[keys[k]]++ }
    for (let i = 1; i < 65536; i++) counts[i] += counts[i - 1]
    for (let k = count - 1; k >= 0; k--) order[--counts[keys[k]]] = k
    const c3 = aCenter.array as Float32Array, ca = aCovA.array as Float32Array
    const cb = aCovB.array as Float32Array, c4 = aColor.array as Float32Array
    for (let i = 0; i < count; i++) {
      const k = order[i]
      c3[i * 3] = srcCenter[k * 3]; c3[i * 3 + 1] = srcCenter[k * 3 + 1]; c3[i * 3 + 2] = srcCenter[k * 3 + 2]
      ca[i * 3] = srcCovA[k * 3]; ca[i * 3 + 1] = srcCovA[k * 3 + 1]; ca[i * 3 + 2] = srcCovA[k * 3 + 2]
      cb[i * 3] = srcCovB[k * 3]; cb[i * 3 + 1] = srcCovB[k * 3 + 1]; cb[i * 3 + 2] = srcCovB[k * 3 + 2]
      c4[i * 4] = srcColor[k * 4]; c4[i * 4 + 1] = srcColor[k * 4 + 1]; c4[i * 4 + 2] = srcColor[k * 4 + 2]; c4[i * 4 + 3] = srcColor[k * 4 + 3]
    }
    aCenter.needsUpdate = aCovA.needsUpdate = aCovB.needsUpdate = aColor.needsUpdate = true
  }

  const drawingSize = new THREE.Vector2()
  mesh.onBeforeRender = (renderer, _scene, camera) => {
    renderer.getDrawingBufferSize(drawingSize)
    viewport.copy(drawingSize)
    mv.multiplyMatrices(camera.matrixWorldInverse, mesh.matrixWorld)
    const e = mv.elements
    if (Math.abs(e[2] - lastView[0]) + Math.abs(e[6] - lastView[1]) + Math.abs(e[10] - lastView[2]) + Math.abs(e[14] - lastView[3]) > 1e-4) {
      dirty = true
      lastView[0] = e[2]; lastView[1] = e[6]; lastView[2] = e[10]; lastView[3] = e[14]
    }
    const now = performance.now()
    if (dirty && now - lastSort > 90) { sortBackToFront(); dirty = false; lastSort = now }
  }

  /** Moves the splat (with its current rotation) so it is centred over the origin and
   *  rests on the floor. Uses 2nd/98th percentiles of a sample so stray floaters don't
   *  skew the fit. For files with an arbitrary origin; captures keep their own placement. */
  const placeOnFloor = (floorY = 0) => {
    mesh.updateMatrix()
    const rot = new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeRotationFromEuler(mesh.rotation))
    const step = Math.max(1, Math.floor(count / 20_000))
    const xs: number[] = [], ys: number[] = [], zs: number[] = []
    const p = new THREE.Vector3()
    for (let k = 0; k < count; k += step) {
      p.set(srcCenter[k * 3], srcCenter[k * 3 + 1], srcCenter[k * 3 + 2]).applyMatrix3(rot)
      xs.push(p.x); ys.push(p.y); zs.push(p.z)
    }
    const pct = (v: number[], q: number) => { v.sort((a, b) => a - b); return v[Math.min(v.length - 1, Math.floor(q * v.length))] }
    const midX = (pct(xs, 0.02) + pct(xs, 0.98)) / 2
    const midZ = (pct(zs, 0.02) + pct(zs, 0.98)) / 2
    const bottom = pct(ys, 0.02), top = pct(ys, 0.98)
    mesh.position.set(-midX, floorY - bottom, -midZ)
    return { height: top - bottom }
  }

  return {
    object: mesh,
    totalSplats: total,
    renderedSplats: count,
    placeOnFloor,
    dispose() { geometry.dispose(); material.dispose() },
  }
}
