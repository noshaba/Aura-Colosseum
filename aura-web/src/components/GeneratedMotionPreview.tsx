import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { BVHLoader } from 'three/examples/jsm/loaders/BVHLoader.js'

/** Robot-native skeletal inspection for saved generated motion. */
export function GeneratedMotionPreview({ file }: { file: string }) {
  const mount = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = mount.current
    if (!el) return
    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#f7f7f2')
    const camera = new THREE.PerspectiveCamera(44, 1, 0.01, 500)
    camera.position.set(2.4, 1.8, 4)
    camera.lookAt(0, 1, 0)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    el.appendChild(renderer.domElement)
    const grid = new THREE.GridHelper(6, 18, '#ceb575', '#ddd8c8')
    scene.add(grid)
    const mixerHolder: { mixer?: THREE.AnimationMixer } = {}
    const clock = new THREE.Clock()
    let frame = 0
    let disposed = false
    const fit = () => {
      const w = el.clientWidth || 500, h = el.clientHeight || 320
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    fit()
    const resize = new ResizeObserver(fit)
    resize.observe(el)
    let g1Data: {positions: number[][][]; fps: number; parents: number[]} | null = null
    let g1Line: THREE.LineSegments | null = null
    let g1Elapsed = 0
    if (file.endsWith('.g1.json')) {
      fetch(file).then(r => { if (!r.ok) throw new Error(`Preview HTTP ${r.status}`); return r.json() })
        .then((data: {format: string; positions: number[][][]; parents: number[]; fps: number}) => {
          if (disposed) return
          if (data.format !== 'g1-joints-v1' || !data.positions?.length || data.parents?.length !== 34) throw new Error('Invalid G1 preview data')
          g1Data = data
          const geom = new THREE.BufferGeometry()
          const edges = data.parents.filter(p => p >= 0).length
          geom.setAttribute('position',new THREE.BufferAttribute(new Float32Array(edges*6),3))
          g1Line = new THREE.LineSegments(geom,new THREE.LineBasicMaterial({color:'#bd963a',linewidth:2}))
          scene.add(g1Line)
          // Text2Motion Aura already exports G1 joints in Three.js-compatible axes:
          // X right, Y up, Z forward. Preview the complete trajectory instead
          // of aiming at just the first frame (walking would leave the view).
          const bounds = new THREE.Box3()
          const sampleStride = Math.max(1, Math.floor(data.positions.length / 120))
          for (let f = 0; f < data.positions.length; f += sampleStride) {
            for (const point of data.positions[f]) {
              bounds.expandByPoint(new THREE.Vector3(point[0], point[1], point[2]))
            }
          }
          // Include the final frame even when stride skips it.
          for (const point of data.positions[data.positions.length - 1]) {
            bounds.expandByPoint(new THREE.Vector3(point[0], point[1], point[2]))
          }
          const center = bounds.getCenter(new THREE.Vector3())
          const extent = bounds.getSize(new THREE.Vector3())
          const radius = Math.max(2, extent.length() * 0.7)
          camera.position.copy(center).add(new THREE.Vector3(radius * 0.55, radius * 0.35, radius * 1.25))
          camera.lookAt(center)
          camera.near = 0.01
          camera.far = Math.max(100, radius * 15)
          camera.updateProjectionMatrix()
        }).catch(err => { if (!disposed) console.error('G1 preview failed',err) })
    } else new BVHLoader().load(file, bvh => {
      if (disposed) return
      const root = bvh.skeleton.bones[0]
      scene.add(root)
      const helper = new THREE.SkeletonHelper(root)
      const helperMaterial = helper.material
      if (!Array.isArray(helperMaterial) && helperMaterial instanceof THREE.LineBasicMaterial) helperMaterial.color.set('#b99a46')
      scene.add(helper)
      const bounds = new THREE.Box3().setFromObject(helper)
      const size = bounds.getSize(new THREE.Vector3()).length() || 2
      const center = bounds.getCenter(new THREE.Vector3())
      camera.position.copy(center.clone().add(new THREE.Vector3(size * .7, size * .5, size * 1.2)))
      camera.lookAt(center)
      camera.near = Math.max(.01, size / 1000)
      camera.far = Math.max(100, size * 100)
      camera.updateProjectionMatrix()
      const mixer = new THREE.AnimationMixer(root)
      mixer.clipAction(bvh.clip).play()
      mixerHolder.mixer = mixer
    }, undefined, err => {
      if (!disposed) console.error('BVH preview failed', err)
    })
    const tick = () => {
      if (disposed) return
      frame = requestAnimationFrame(tick)
      const dt = Math.min(clock.getDelta(), .05)
      mixerHolder.mixer?.update(dt)
      if (g1Data && g1Line) {
        g1Elapsed += dt
        const data = g1Data as {positions:number[][][];fps:number;parents:number[]}
        const coords = data.positions[Math.floor(g1Elapsed * data.fps) % data.positions.length]
        const attr = g1Line.geometry.getAttribute('position') as THREE.BufferAttribute
        let edge = 0
        for (let joint=0;joint<data.parents.length;joint++) {
          const parent=data.parents[joint]
          if (parent<0) continue
          // Text2Motion Aura's generated G1 positions are Y-up and Z-forward,
          // matching Three.js. Do NOT apply MuJoCo's Z-up transform here.
          const a=coords[joint], b=coords[parent]
          attr.setXYZ(edge*2,a[0],a[1],a[2]); attr.setXYZ(edge*2+1,b[0],b[1],b[2]); edge++
        }
        attr.needsUpdate=true
      }
      renderer.render(scene, camera)
    }
    tick()
    return () => {
      disposed = true
      cancelAnimationFrame(frame)
      resize.disconnect()
      mixerHolder.mixer?.stopAllAction()
      g1Line?.geometry.dispose()
      if (g1Line) (g1Line.material as THREE.Material).dispose()
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [file])
  return <div className="generated-preview" ref={mount} aria-label="Animated generated motion skeletal preview" />
}
