import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'

type G1Preview = {
  format: 'g1-joints-v2'
  fps: number
  positions: number[][][]
  global_rot_mats: number[][][][]
}

type RigItem = {
  mesh: THREE.Mesh
  joint: number
  geomPos: THREE.Vector3
  geomQuat: THREE.Quaternion
}

const G1_MESH_JOINTS: Record<number, string[]> = {
  0: ['pelvis.STL', 'pelvis_contour_link.STL'],
  1: ['left_hip_pitch_link.STL'],
  2: ['left_hip_roll_link.STL'],
  3: ['left_hip_yaw_link.STL'],
  4: ['left_knee_link.STL'],
  5: ['left_ankle_pitch_link.STL'],
  6: ['left_ankle_roll_link.STL'],
  8: ['right_hip_pitch_link.STL'],
  9: ['right_hip_roll_link.STL'],
  10: ['right_hip_yaw_link.STL'],
  11: ['right_knee_link.STL'],
  12: ['right_ankle_pitch_link.STL'],
  13: ['right_ankle_roll_link.STL'],
  15: ['waist_yaw_link_rev_1_0.STL'],
  16: ['waist_roll_link_rev_1_0.STL'],
  17: ['torso_link_rev_1_0.STL', 'logo_link.STL', 'head_link.STL'],
  18: ['left_shoulder_pitch_link.STL'],
  19: ['left_shoulder_roll_link.STL'],
  20: ['left_shoulder_yaw_link.STL'],
  21: ['left_elbow_link.STL'],
  22: ['left_wrist_roll_link.STL'],
  23: ['left_wrist_pitch_link.STL'],
  24: ['left_wrist_yaw_link.STL', 'left_rubber_hand.STL'],
  26: ['right_shoulder_pitch_link.STL'],
  27: ['right_shoulder_roll_link.STL'],
  28: ['right_shoulder_yaw_link.STL'],
  29: ['right_elbow_link.STL'],
  30: ['right_wrist_roll_link.STL'],
  31: ['right_wrist_pitch_link.STL'],
  32: ['right_wrist_yaw_link.STL', 'right_rubber_hand.STL'],
}

const FILE_TO_JOINT = new Map<string, number>()
Object.entries(G1_MESH_JOINTS).forEach(([joint, files]) => files.forEach(file => FILE_TO_JOINT.set(file, Number(joint))))

// MuJoCo (z-up, x-forward) -> Text2Motion Aura / Three.js (y-up, z-forward).
const MUJOCO_TO_TEXT2MOTION_AURA = new THREE.Matrix4().set(
  0, 1, 0, 0,
  0, 0, 1, 0,
  1, 0, 0, 0,
  0, 0, 0, 1,
)
const TEXT2MOTION_AURA_TO_MUJOCO = MUJOCO_TO_TEXT2MOTION_AURA.clone().transpose()
const geometryCache = new Map<string, Promise<THREE.BufferGeometry>>()
let transformCache: Promise<Map<string, { pos: THREE.Vector3; quat: THREE.Quaternion }>> | null = null

function parseVec(raw: string | null, n: number) {
  const values = (raw || '').trim().split(/\s+/).filter(Boolean).map(Number)
  return Array.from({ length: n }, (_, i) => values[i] ?? (i === 0 && n === 4 ? 1 : 0))
}

function matrixToQuat(matrix: number[][]) {
  const m = new THREE.Matrix4().set(
    matrix[0][0], matrix[0][1], matrix[0][2], 0,
    matrix[1][0], matrix[1][1], matrix[1][2], 0,
    matrix[2][0], matrix[2][1], matrix[2][2], 0,
    0, 0, 0, 1,
  )
  return new THREE.Quaternion().setFromRotationMatrix(m).normalize()
}

function mujocoQuatToText2Motion Aura(wxyz: number[]) {
  const q = new THREE.Quaternion(wxyz[1], wxyz[2], wxyz[3], wxyz[0]).normalize()
  const r = new THREE.Matrix4().makeRotationFromQuaternion(q)
  const converted = MUJOCO_TO_TEXT2MOTION_AURA.clone().multiply(r).multiply(TEXT2MOTION_AURA_TO_MUJOCO)
  return new THREE.Quaternion().setFromRotationMatrix(converted).normalize()
}

async function loadMeshTransforms(base: string) {
  if (transformCache) return transformCache
  transformCache = (async () => {
    const response = await fetch(`${base}g1.xml`)
    if (!response.ok) throw new Error(`G1 rig XML request failed (${response.status})`)
    const xml = new DOMParser().parseFromString(await response.text(), 'application/xml')
    if (xml.querySelector('parsererror')) throw new Error('Could not parse G1 rig XML')
    const meshNameToFile = new Map<string, string>()
    xml.querySelectorAll('asset > mesh').forEach(mesh => {
      const name = mesh.getAttribute('name'); const file = mesh.getAttribute('file')
      if (name && file) meshNameToFile.set(name, file)
    })
    const byFile = new Map<string, { pos: THREE.Vector3; quat: THREE.Quaternion }>()
    xml.querySelectorAll('geom[mesh]').forEach(geom => {
      const meshName = geom.getAttribute('mesh')
      const file = meshName ? meshNameToFile.get(meshName) : undefined
      if (!file || !FILE_TO_JOINT.has(file)) return
      const [x, y, z] = parseVec(geom.getAttribute('pos'), 3)
      const [w, qx, qy, qz] = parseVec(geom.getAttribute('quat'), 4)
      const pos = new THREE.Vector3(x, y, z).applyMatrix4(MUJOCO_TO_TEXT2MOTION_AURA)
      byFile.set(file, { pos, quat: mujocoQuatToText2Motion Aura([w, qx, qy, qz]) })
    })
    return byFile
  })()
  return transformCache
}

function loadGeometry(url: string) {
  let promise = geometryCache.get(url)
  if (!promise) {
    promise = new STLLoader().loadAsync(url).then(geometry => {
      geometry.applyMatrix4(MUJOCO_TO_TEXT2MOTION_AURA)
      geometry.computeVertexNormals()
      return geometry
    })
    geometryCache.set(url, promise)
  }
  return promise
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
    color: gold ? 0xc49a43 : 0xf4eee4,
    map: gold ? null : marbleTexture,
    emissive: new THREE.Color(gold ? 0x2a1708 : 0x16090b),
    emissiveIntensity: gold ? 0.06 : 0.025,
    side: THREE.DoubleSide,
  })
}

function addComicOutline(mesh: THREE.Mesh) {
  const outline = new THREE.Mesh(
    mesh.geometry,
    new THREE.MeshBasicMaterial({ color: 0x251014, side: THREE.BackSide, transparent: true, opacity: 0.72 }),
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
    let elapsed = 0
    let last = performance.now()
    let lastUiUpdate = 0

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#741c26')
    scene.fog = new THREE.FogExp2(0x741c26, 0.020)
    const camera = new THREE.PerspectiveCamera(36, 1, 0.02, 100)
    camera.position.set(2.15, 1.45, 3.1)
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.06
    el.appendChild(renderer.domElement)

    const marbleTexture = new THREE.TextureLoader().load(`${import.meta.env.BASE_URL || '/'}textures/marble-gold.png`)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.enablePan = false
    controls.target.set(0, .75, 0)
    controls.minDistance = 1.5
    controls.maxDistance = 7

    scene.add(new THREE.HemisphereLight(0xfff1dc, 0x3d1016, 1.75))
    const key = new THREE.DirectionalLight(0xffe0ad, 4.6); key.position.set(-3, 5, 4); key.castShadow = true; scene.add(key)
    const fill = new THREE.DirectionalLight(0xe05b64, 2.15); fill.position.set(4, 2.5, 3); scene.add(fill)
    const rim = new THREE.DirectionalLight(0xe0b862, 2.0); rim.position.set(-2, 3, -4); scene.add(rim)

    const ground = new THREE.Mesh(new THREE.CircleGeometry(5.6, 96), new THREE.MeshPhysicalMaterial({ color: 0xf1e7d7, roughness: .84, metalness: .01, clearcoat: .08, clearcoatRoughness: .82 }))
    ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground)
    const floorHalo = new THREE.Mesh(new THREE.RingGeometry(2.55, 3.55, 96), new THREE.MeshBasicMaterial({ color: 0xb82734, transparent: true, opacity: .24, side: THREE.DoubleSide }))
    floorHalo.rotation.x = -Math.PI / 2; floorHalo.position.y = .006; scene.add(floorHalo)
    const ring = new THREE.Mesh(new THREE.RingGeometry(1.92, 1.98, 96), new THREE.MeshBasicMaterial({ color: 0xcaa45a, transparent: true, opacity: .72, side: THREE.DoubleSide }))
    ring.rotation.x = -Math.PI / 2; ring.position.y = .009; scene.add(ring)

    const motionRoot = new THREE.Group(); scene.add(motionRoot)
    const pathMaterial = new THREE.LineBasicMaterial({ color: 0xffc0aa, transparent: true, opacity: .62 })
    let pathLine: THREE.Line | null = null

    const resize = () => {
      const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight)
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix()
    }
    const observer = new ResizeObserver(resize); observer.observe(el); resize()

    const base = `${import.meta.env.BASE_URL || '/'}models/g1-native/`
    Promise.all([
      fetch(file, { cache: 'no-store' }).then(async response => {
        if (!response.ok) throw new Error(`Generated motion request failed (${response.status})`)
        return response.json() as Promise<G1Preview>
      }),
      loadMeshTransforms(base),
    ]).then(async ([preview, transforms]) => {
      if (disposed) return
      if (preview.format !== 'g1-joints-v2' || !preview.positions?.length || !preview.global_rot_mats?.length) {
        throw new Error('Generated preview needs G1 pose rotations. Restart the Aura library server once to upgrade old previews.')
      }
      data = preview
      setDuration(preview.positions.length / preview.fps)
      const entries = [...FILE_TO_JOINT.entries()]
      rig = (await Promise.all(entries.map(async ([meshFile, joint]) => {
        const transform = transforms.get(meshFile) || { pos: new THREE.Vector3(), quat: new THREE.Quaternion() }
        const geometry = (await loadGeometry(`${base}meshes/${meshFile}`)).clone()
        const mesh = new THREE.Mesh(geometry, materialFor(meshFile, marbleTexture))
        mesh.name = `generated_${meshFile}`
        addComicOutline(mesh)
        mesh.castShadow = true; mesh.receiveShadow = true; mesh.frustumCulled = false
        motionRoot.add(mesh)
        return { mesh, joint, geomPos: transform.pos.clone(), geomQuat: transform.quat.clone() }
      }))).filter(Boolean)

      const firstRoot = preview.positions[0][0]
      const pathPoints = preview.positions.map(frame => new THREE.Vector3(frame[0][0] - firstRoot[0], .012, frame[0][2] - firstRoot[2]))
      const pathGeometry = new THREE.BufferGeometry().setFromPoints(pathPoints)
      pathLine = new THREE.Line(pathGeometry, pathMaterial); scene.add(pathLine)

      const xs = pathPoints.map(p => p.x), zs = pathPoints.map(p => p.z)
      const span = Math.max(1.3, Math.max(...xs) - Math.min(...xs), Math.max(...zs) - Math.min(...zs))
      camera.position.set(span * .85 + 1.4, 1.55, span * 1.15 + 2.0)
      controls.target.set((Math.min(...xs) + Math.max(...xs)) / 2, .72, (Math.min(...zs) + Math.max(...zs)) / 2)
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
      const dt = Math.min((now - last) / 1000, .08); last = now
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
      controls.update(); renderer.render(scene, camera)
    }
    raf = requestAnimationFrame(tick)

    return () => {
      disposed = true; cancelAnimationFrame(raf); observer.disconnect(); controls.dispose()
      rig.forEach(item => { item.mesh.geometry.dispose(); (item.mesh.material as THREE.Material).dispose() })
      pathLine?.geometry.dispose(); pathMaterial.dispose(); marbleTexture.dispose(); renderer.dispose(); renderer.domElement.remove()
    }
  }, [file])

  return <div className={compact ? 'generated-g1-player compact' : 'generated-g1-player'}>
    <div className="generated-g1-canvas" ref={mount} />
    <div className="generated-g1-overlay"><span><i /> AURA → G1</span><small>{status}</small></div>
    <div className="generated-g1-controls">
      <button type="button" onClick={() => setPaused(value => !value)}>{paused ? '▶ Play' : 'Ⅱ Pause'}</button>
      <div><span style={{ width: `${progress * 100}%` }} /></div>
      <small>{duration ? `${duration.toFixed(1)} s · loops automatically` : 'loading…'}</small>
    </div>
  </div>
}
