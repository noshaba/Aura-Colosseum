import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { BVHLoader } from 'three/examples/jsm/loaders/BVHLoader.js'
import { AURA_OVERLAY_LAYER, createAuraWorld } from '../three/auraWorld'
import { createStrokedSegments, type StrokedSegments } from '../three/thickLines'
import { FairyRig, createPoseSample, dressFairy, loadMixamoAsset, sampleFromPositions, type MixamoAsset } from '../three/fairyRig'
import { useCharacter } from '../hooks/useCharacter'

/** Robot-native skeletal inspection for saved generated motion. */
export function GeneratedMotionPreview({ file }: { file: string }) {
  const mount = useRef<HTMLDivElement>(null)
  const { character } = useCharacter()
  useEffect(() => {
    const el = mount.current
    if (!el) return
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(44, 1, 0.01, 500)
    camera.position.set(2.4, 1.8, 4)
    camera.lookAt(0, 1, 0)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    el.appendChild(renderer.domElement)
    // Stylised Aura world (sky, ground, mesas, ink pass) replaces the old grid void.
    const world = createAuraWorld(scene, renderer, { contactShadow: true, ring: true, props: { density: 0.45, keepOut: 2.6 } })
    world.observe(el)
    // Thick screen-space skeleton: navy under-stroke + eggshell core, drawn after the ink pass.
    let strokes: StrokedSegments | null = null
    let cssW = 1, cssH = 1
    const makeStrokes = (segments: number) => {
      const st = createStrokedSegments(segments, { coreWidth: 3.5, underWidth: 7.5, layer: AURA_OVERLAY_LAYER })
      st.setResolution(cssW, cssH)
      scene.add(st.group)
      return st
    }
    const mixerHolder: { mixer?: THREE.AnimationMixer } = {}
    const clock = new THREE.Clock()
    let frame = 0
    let disposed = false
    const fit = () => {
      const w = el.clientWidth || 500, h = el.clientHeight || 320
      renderer.setSize(w, h)
      cssW = w; cssH = h
      strokes?.setResolution(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    fit()
    const resize = new ResizeObserver(fit)
    resize.observe(el)
    let g1Data: {positions: number[][][]; fps: number; parents: number[]} | null = null
    let bvhHelper: THREE.SkeletonHelper | null = null
    let g1Elapsed = 0
    // A mixamo-retarget character (three/characters.ts) performs the G1 joints (positions-only
    // data, rotations estimated in fairyRig) instead of the stroked skeleton; the G1 body keeps it.
    const mixamo = character.kind === 'mixamo-retarget' ? character : null
    const useFairy = !!mixamo && file.endsWith('.g1.json')
    let fairy: FairyRig | null = null
    const fairyMats: THREE.Material[] = []
    const fairySample = createPoseSample()
    if (useFairy) {
      // No tone mapping on this renderer, so softer than the G1 preview's ACES-mapped rig.
      scene.add(new THREE.HemisphereLight(0xfff4e4, 0x2a231c, 0.9))
      const key = new THREE.DirectionalLight(0xfff0dc, 1.6); key.position.set(-3, 5, 4); scene.add(key)
      const fill = new THREE.DirectionalLight(0xd8c7ad, 0.7); fill.position.set(4, 2.5, 3); scene.add(fill)
    }
    if (file.endsWith('.g1.json')) {
      Promise.all([
        fetch(file).then(r => { if (!r.ok) throw new Error(`Preview HTTP ${r.status}`); return r.json() }),
        useFairy && mixamo ? loadMixamoAsset(mixamo) : Promise.resolve(null),
      ]).then(([data, asset]: [{format: string; positions: number[][][]; parents: number[]; fps: number}, MixamoAsset | null]) => {
          if (disposed) return
          if (data.format !== 'g1-joints-v1' || !data.positions?.length || data.parents?.length !== 34) throw new Error('Invalid G1 preview data')
          g1Data = data
          if (asset && mixamo) {
            fairy = new FairyRig(asset, mixamo)
            fairy.root.scale.setScalar(1 / fairy.scale) // fairy legs = G1 legs: follows the G1 trajectory exactly
            const { body, ink } = dressFairy(fairy, asset)
            fairyMats.push(body, ink)
            scene.add(fairy.root)
          } else strokes = makeStrokes(data.parents.filter(p => p >= 0).length)
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
          world.setFocus(center.x, center.z, Math.max(2.6, Math.hypot(extent.x, extent.z) * 0.6 + 1.1))
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
      // The helper only computes bone segments (root-local); the thick strokes draw them.
      helper.visible = false
      scene.add(helper)
      bvhHelper = helper
      strokes = makeStrokes((helper.geometry.getAttribute('position') as THREE.BufferAttribute).count / 2)
      strokes.group.matrixAutoUpdate = false
      strokes.group.matrix = helper.matrix // SkeletonHelper's matrix is the root's world matrix
      const bounds = new THREE.Box3().setFromObject(helper)
      const size = bounds.getSize(new THREE.Vector3()).length() || 2
      const center = bounds.getCenter(new THREE.Vector3())
      world.setFocus(center.x, center.z, size * 0.7)
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
      if (!world.visible) return
      mixerHolder.mixer?.update(dt)
      if (bvhHelper && strokes) {
        bvhHelper.updateMatrixWorld(true)
        const src = bvhHelper.geometry.getAttribute('position') as THREE.BufferAttribute
        strokes.positions.set((src.array as Float32Array).subarray(0, strokes.positions.length))
        strokes.commit()
      }
      if (g1Data && (fairy || strokes)) {
        g1Elapsed += dt
        const data = g1Data as {positions:number[][][];fps:number;parents:number[]}
        const coords = data.positions[Math.floor(g1Elapsed * data.fps) % data.positions.length]
        if (fairy) fairy.applyPose(sampleFromPositions(coords, fairySample))
        else if (strokes) {
          const out = strokes.positions
          let edge = 0
          for (let joint=0;joint<data.parents.length;joint++) {
            const parent=data.parents[joint]
            if (parent<0) continue
            // Text2Motion Aura's generated G1 positions are Y-up and Z-forward,
            // matching Three.js. Do NOT apply MuJoCo's Z-up transform here.
            const a=coords[joint], b=coords[parent]
            out[edge*6]=a[0]; out[edge*6+1]=a[1]; out[edge*6+2]=a[2]
            out[edge*6+3]=b[0]; out[edge*6+4]=b[1]; out[edge*6+5]=b[2]; edge++
          }
          strokes.commit()
        }
        world.setShadow(coords[0][0], coords[0][2])
      }
      world.render(camera, dt)
    }
    tick()
    return () => {
      disposed = true
      cancelAnimationFrame(frame)
      resize.disconnect()
      mixerHolder.mixer?.stopAllAction()
      strokes?.dispose()
      fairy?.dispose(); fairyMats.forEach(m => m.dispose()) // body + ink from dressFairy; geometry is shared (cached GLB)
      if (bvhHelper) { bvhHelper.geometry.dispose(); (bvhHelper.material as THREE.Material).dispose() }
      world.dispose()
      renderer.dispose()
      renderer.forceContextLoss() // a character switch rebuilds the viewer: free the context now, not at GC
      renderer.domElement.remove()
    }
    // character (not just its id) is read inside: the registry entry is stable per id.
  }, [file, character.id])
  return <div className="generated-preview" ref={mount} aria-label="Animated generated motion skeletal preview" />
}
