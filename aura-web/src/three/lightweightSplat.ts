import * as THREE from 'three'

/**
 * Lightweight preview for the common 32-byte `.splat` interchange format.
 * This deliberately does NOT claim full production 3DGS fidelity: it uses
 * camera-facing Gaussian billboards, caps the visible splat count, and leaves
 * collision/robot training to Aura's explicit geometry proxy.
 */
export function createLightweightSplat(buffer: ArrayBuffer, maxSplats = 140_000) {
  const stride = 32
  const total = Math.floor(buffer.byteLength / stride)
  if (!total) throw new Error('This .splat file contains no 32-byte splat records.')
  const step = Math.max(1, Math.ceil(total / maxSplats))
  const count = Math.ceil(total / step)
  const view = new DataView(buffer)
  const centers = new Float32Array(count * 3)
  const scales = new Float32Array(count * 2)
  const colors = new Float32Array(count * 4)

  let out = 0
  for (let src = 0; src < total; src += step) {
    const o = src * stride
    centers[out * 3] = view.getFloat32(o + 0, true)
    centers[out * 3 + 1] = view.getFloat32(o + 4, true)
    centers[out * 3 + 2] = view.getFloat32(o + 8, true)
    const sx = Math.abs(view.getFloat32(o + 12, true))
    const sy = Math.abs(view.getFloat32(o + 16, true))
    const sz = Math.abs(view.getFloat32(o + 20, true))
    // Conservative billboard footprint. Full covariance/rotation is intentionally
    // not used in this lightweight hackathon preview.
    const major = Math.max(0.002, Math.min(0.7, Math.max(sx, sy, sz)))
    const minor = Math.max(0.002, Math.min(0.7, Math.max(Math.min(sx, sy), Math.min(sy, sz), Math.min(sx, sz))))
    scales[out * 2] = major * 2.25
    scales[out * 2 + 1] = minor * 2.25
    colors[out * 4] = view.getUint8(o + 24) / 255
    colors[out * 4 + 1] = view.getUint8(o + 25) / 255
    colors[out * 4 + 2] = view.getUint8(o + 26) / 255
    colors[out * 4 + 3] = view.getUint8(o + 27) / 255
    out += 1
  }

  const geometry = new THREE.InstancedBufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    -1, -1, 0, 1, -1, 0, 1, 1, 0,
    -1, -1, 0, 1, 1, 0, -1, 1, 0,
  ], 3))
  geometry.setAttribute('iCenter', new THREE.InstancedBufferAttribute(centers, 3))
  geometry.setAttribute('iScale', new THREE.InstancedBufferAttribute(scales, 2))
  geometry.setAttribute('iColor', new THREE.InstancedBufferAttribute(colors, 4))
  geometry.instanceCount = count

  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.NormalBlending,
    vertexShader: `
      attribute vec3 iCenter;
      attribute vec2 iScale;
      attribute vec4 iColor;
      varying vec2 vUv;
      varying vec4 vColor;
      void main() {
        vec4 mv = modelViewMatrix * vec4(iCenter, 1.0);
        mv.xy += position.xy * iScale;
        gl_Position = projectionMatrix * mv;
        vUv = position.xy;
        vColor = iColor;
      }
    `,
    fragmentShader: `
      varying vec2 vUv;
      varying vec4 vColor;
      void main() {
        float r2 = dot(vUv, vUv);
        if (r2 > 1.0) discard;
        float alpha = exp(-4.2 * r2) * vColor.a * 0.86;
        if (alpha < 0.015) discard;
        gl_FragColor = vec4(vColor.rgb, alpha);
      }
    `,
  })

  const mesh = new THREE.Mesh(geometry, material)
  mesh.frustumCulled = false
  mesh.renderOrder = -5
  return {
    object: mesh,
    totalSplats: total,
    renderedSplats: count,
    dispose() { geometry.dispose(); material.dispose() },
  }
}
