import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { AURA_OVERLAY_LAYER, AURA_PALETTE, createAuraRenderer, createAuraWorld, prefersReducedMotion } from '../three/auraWorld'
import { RobotFlare, effectorsByName } from '../three/robotFlare'
import { toonStylize, type ToonStylizeHandle } from '../three/toonStylize'
import { createThickPath } from '../three/thickLines'
import { FILE_TO_JOINT, loadGeometry, loadMeshTransforms, matrixToQuat, type G1Preview } from '../three/g1Rig'
import { FAIRY_EFFECTORS, FairyRig, ROBOT_MODEL, createPoseSample, dressFairy, loadFairyAsset, sampleFromPreview } from '../three/fairyRig'

/** Bind-pose height the fairy is fitted to here, in this viewer's (G1-sized) metres. */
const FAIRY_DISPLAY_HEIGHT = 1.4

type RigItem = {
  mesh: THREE.Mesh
  joint: number
  geomPos: THREE.Vector3
  geomQuat: THREE.Quaternion
}

function materialFor(file: string, marbleTexture: THREE.Texture) {
  // Aura comic palette: glazed ivory + antique gold, rendered with discrete toon
  // light bands rather than physically-based reflections.
  const gold = /pelvis\.STL|pelvis_contour|hip_pitch|ankle_roll|logo_link|head_link/i.test(file)
  if (!gold) {
    marbleTexture.colorSpace = THREE.SRGBColorSpace
    marbleTexture.wrapS = THREE.RepeatWrapping
    marbleTexture.wrapT = THREE.RepeatWrapping
    marbleTexture.repeat.set(0.62, 0.62)
    marbleTexture.center.set(0.5, 0.5)
    marbleTexture.rotation = -0.08
  }
  return new THREE.MeshToonMaterial({
    color: gold ? 0xa8957a : 0xf2e9da,
    map: gold ? null : marbleTexture,
    emissive: new THREE.Color(gold ? 0x1f1912 : 0x14100c),
    emissiveIntensity: gold ? 0.06 : 0.025,
    side: THREE.DoubleSide,
  })
}

function addComicOutline(mesh: THREE.Mesh) {
  const outline = new THREE.Mesh(
    mesh.geometry,
    new THREE.MeshBasicMaterial({ color: 0x2a2c40, side: THREE.BackSide, transparent: true, opacity: 0.72 }),
  )
  outline.name = `${mesh.name}_comic_outline`
  outline.scale.setScalar(1.013)
  outline.renderOrder = -1
  mesh.add(outline)
}

export function GeneratedG1RobotPreview({ file, compact = false }: { file: string; compact?: boolean }) {
  const mount = useRef<HTMLDivElement>(null)
  const [paused, setPaused] = useState(false)
  const [status, setStatus] = useState('Loading generated G1 motion…')
  const [duration, setDuration] = useState(0)
  const [progress, setProgress] = useState(0)
  const pausedRef = useRef(false)
  pausedRef.current = paused

  useEffect(() => {
    setPaused(false)
    setProgress(0)
    const el = mount.current
    if (!el) return
    let disposed = false
    let raf = 0
    let data: G1Preview | null = null
    let rig: RigItem[] = []
    // Blossom Fairy path (ROBOT_MODEL === 'fairy'): one skinned clone retargeted from the G1 joints.
    let fairy: FairyRig | null = null
    let fairyMats: THREE.Material[] = []
    const fairySample = createPoseSample()
    let travelScale = 1
    let elapsed = 0
    let last = performance.now()
    let lastUiUpdate = 0

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(36, 1, 0.02, 100)
    camera.position.set(2.15, 1.45, 3.1)
    const renderer = createAuraRenderer({ powerPreference: 'high-performance' }) // canvas MSAA only without the world composite
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.06
    el.appendChild(renderer.domElement)
    // Stylised Aura world: sky, ground (receives the robot's shadow), mesas, ink pass.
    const world = createAuraWorld(scene, renderer, { ring: true, props: { density: 0.7, keepOut: 2.6 } })
    world.observe(el)

    // Marble skin of the G1 rollback body: loaded only on that path (see the rig build below).
    let marbleTexture: THREE.Texture | null = null

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.enablePan = false
    controls.target.set(0, .75, 0)
    controls.minDistance = 1.5
    controls.maxDistance = 7

    scene.add(new THREE.HemisphereLight(0xfff4e4, 0x2a231c, 1.75))
    const key = new THREE.DirectionalLight(0xfff0dc, 4.6); key.position.set(-3, 5, 4); key.castShadow = true; scene.add(key)
    const fill = new THREE.DirectionalLight(0xd8c7ad, 2.15); fill.position.set(4, 2.5, 3); scene.add(fill)
    const rim = new THREE.DirectionalLight(0xcdb898, 2.0); rim.position.set(-2, 3, -4); scene.add(rim)


    const motionRoot = new THREE.Group(); scene.add(motionRoot)
    // Thick terra root path on the ground; it stays in the main scene so the robot
    // occludes it and the ink pass gives it a navy edge.
    let path: ReturnType<typeof createThickPath> | null = null
    let flare: RobotFlare | null = null
    let stylized: ToonStylizeHandle | null = null
    let cssW = 1, cssH = 1

    const resize = () => {
      const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight)
      renderer.setSize(w, h, false); cssW = w; cssH = h; path?.setResolution(w, h); camera.aspect = w / h; camera.updateProjectionMatrix()
    }
    const observer = new ResizeObserver(resize); observer.observe(el); resize()

    const base = `${import.meta.env.BASE_URL || '/'}models/g1-native/`
    Promise.all([
      fetch(file, { cache: 'no-store' }).then(async response => {
        if (!response.ok) throw new Error(`Generated motion request failed (${response.status})`)
        return response.json() as Promise<G1Preview>
      }),
      ROBOT_MODEL === 'fairy' ? loadFairyAsset() : loadMeshTransforms(base),
    ]).then(async ([preview, model]) => {
      if (disposed) return
      if (preview.format !== 'g1-joints-v2' || !preview.positions?.length || !preview.global_rot_mats?.length) {
        throw new Error('Generated preview needs G1 pose rotations. Restart the Aura library server once to upgrade old previews.')
      }
      data = preview
      setDuration(preview.positions.length / preview.fps)
      if (!(model instanceof Map)) {
        fairy = new FairyRig(model)
        const fit = FAIRY_DISPLAY_HEIGHT / fairy.height
        fairy.root.scale.setScalar(fit)
        travelScale = fit * fairy.scale
        const { body, ink } = dressFairy(fairy, model, { receiveShadow: true })
        fairyMats = [body, ink]
        motionRoot.add(fairy.root)
      }
      const transforms = model instanceof Map ? model : new Map<string, { pos: THREE.Vector3; quat: THREE.Quaternion }>()
      const entries = fairy ? [] : [...FILE_TO_JOINT.entries()]
      if (!fairy) marbleTexture = new THREE.TextureLoader().load(`${import.meta.env.BASE_URL || '/'}textures/marble-gold.png`)
      const marble = marbleTexture
      rig = (await Promise.all(entries.map(async ([meshFile, joint]) => {
        const transform = transforms.get(meshFile) || { pos: new THREE.Vector3(), quat: new THREE.Quaternion() }
        const geometry = (await loadGeometry(`${base}meshes/${meshFile}`)).clone()
        const mesh = new THREE.Mesh(geometry, materialFor(meshFile, marble!))
        mesh.name = `generated_${meshFile}`
        addComicOutline(mesh)
        mesh.castShadow = true; mesh.receiveShadow = true; mesh.frustumCulled = false
        motionRoot.add(mesh)
        return { mesh, joint, geomPos: transform.pos.clone(), geomQuat: transform.quat.clone() }
      }))).filter(Boolean)
      // Paint the G1 body into the world's palette language (toonStylize.ts); the fairy already
      // gets it in createFairyMaterial (fairyRig.ts).
      if (!fairy) stylized = toonStylize(motionRoot)
      // Fairy flare riding the fastest limb (hands, feet, head); none under reduced motion.
      if (!prefersReducedMotion()) {
        flare = new RobotFlare(effectorsByName(motionRoot, fairy ? FAIRY_EFFECTORS : /rubber_hand|ankle_roll_link|head_link/i), { scale: 1, strength: 0.8, layer: AURA_OVERLAY_LAYER })
        scene.add(flare.points)
      }

      const firstRoot = preview.positions[0][0]
      const pathPoints = preview.positions.map(frame => new THREE.Vector3((frame[0][0] - firstRoot[0]) * travelScale, .012, (frame[0][2] - firstRoot[2]) * travelScale))
      path = createThickPath(pathPoints, AURA_PALETTE.terra, 4)
      path.setResolution(cssW, cssH)
      scene.add(path.line)

      const xs = pathPoints.map(p => p.x), zs = pathPoints.map(p => p.z)
      const span = Math.max(1.3, Math.max(...xs) - Math.min(...xs), Math.max(...zs) - Math.min(...zs))
      camera.position.set(span * .85 + 1.4, 1.55, span * 1.15 + 2.0)
      controls.target.set((Math.min(...xs) + Math.max(...xs)) / 2, .72, (Math.min(...zs) + Math.max(...zs)) / 2)
      world.setFocus(controls.target.x, controls.target.z, Math.max(2.6, span * 0.6 + 1.1))
      controls.update()
      setStatus('Playing Aura candidate')
      last = performance.now()
    }).catch(error => {
      if (!disposed) setStatus(error instanceof Error ? error.message : String(error))
    })

    const tmpOffset = new THREE.Vector3()
    const tmpQuat = new THREE.Quaternion()
    const updatePose = (frameIndex: number) => {
      if (!data) return
      const frame = data.positions[frameIndex]
      const rotations = data.global_rot_mats[frameIndex]
      const root0 = data.positions[0][0]
      if (fairy) {
        fairy.applyPose(sampleFromPreview(frame, rotations, fairySample, root0[0], root0[2]))
        motionRoot.updateMatrixWorld(true) // the flare reads these matrices as is (see tick)
        return
      }
      for (const item of rig) {
        const p = frame[item.joint]
        const q = matrixToQuat(rotations[item.joint])
        tmpOffset.copy(item.geomPos).applyQuaternion(q)
        item.mesh.position.set(p[0] - root0[0], p[1], p[2] - root0[2]).add(tmpOffset)
        tmpQuat.copy(q).multiply(item.geomQuat)
        item.mesh.quaternion.copy(tmpQuat)
      }
      motionRoot.updateMatrixWorld(true)
    }

    const tick = (now: number) => {
      if (disposed) return
      raf = requestAnimationFrame(tick)
      const dt = Math.max(0, Math.min((now - last) / 1000, .08)); last = now // rAF time can precede the load-time performance.now()
      if (!world.visible) return
      if (data && !pausedRef.current) elapsed += dt
      if (data) {
        const clipDuration = data.positions.length / data.fps
        if (clipDuration > 0) elapsed %= clipDuration
        const frameIndex = Math.min(data.positions.length - 1, Math.floor(elapsed * data.fps))
        updatePose(frameIndex)
        if (now - lastUiUpdate > 100) {
          setProgress(clipDuration > 0 ? elapsed / clipDuration : 0)
          lastUiUpdate = now
        }
      }
      controls.update()
      // updatePose already ran updateMatrixWorld over the posed body: the flare skips its own matrix refresh
      if (flare) { camera.updateMatrixWorld(); flare.update(dt, renderer.getPixelRatio(), camera, cssH, 1, true) }
      world.render(camera, dt)
    }
    raf = requestAnimationFrame(tick)

    return () => {
      disposed = true; cancelAnimationFrame(raf); observer.disconnect(); controls.dispose()
      rig.forEach(item => {
        item.mesh.geometry.dispose(); (item.mesh.material as THREE.Material).dispose()
        item.mesh.children.forEach(child => { if (child instanceof THREE.Mesh) (child.material as THREE.Material).dispose() })
      })
      stylized?.dispose()
      fairy?.dispose(); fairyMats.forEach(m => m.dispose()) // fairy geometry is shared (cached GLB)
      path?.dispose(); flare?.dispose(); marbleTexture?.dispose(); world.dispose(); renderer.dispose(); renderer.domElement.remove()
    }
  }, [file])

  return <div className={compact ? 'generated-g1-player compact' : 'generated-g1-player'}>
    <div className="generated-g1-canvas" ref={mount} />
    <div className="generated-g1-overlay"><span><i /> AURA → {ROBOT_MODEL === 'fairy' ? 'FAIRY' : 'G1'}</span><small>{status}</small></div>
    <div className="generated-g1-controls">
      <button type="button" onClick={() => setPaused(value => !value)}>{paused ? '▶ Play' : 'Ⅱ Pause'}</button>
      <div><span style={{ width: `${progress * 100}%` }} /></div>
      <small>{duration ? `${duration.toFixed(1)} s · loops automatically` : 'loading…'}</small>
    </div>
  </div>
}
