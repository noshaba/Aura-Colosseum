import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { FILE_TO_JOINT, loadGeometry, loadMeshTransforms, matrixToQuat, type G1Preview } from '../three/g1Rig'
import { createLightweightSplat } from '../three/lightweightSplat'

type ScenePlan = { scene: { floor_y: number }; path: number[][] }
type Props = {
  previewFile?: string | null
  splatBuffer?: ArrayBuffer | null
  plan?: ScenePlan | null
}

type RigPart = { mesh: THREE.Mesh; joint: number; geomPos: THREE.Vector3; geomQuat: THREE.Quaternion }

export function SplatG1SceneViewer({ previewFile, splatBuffer, plan }: Props) {
  const mount = useRef<HTMLDivElement>(null)
  const [status, setStatus] = useState('Load a .splat scene or generate a scene-aware motion.')

  useEffect(() => {
    const el = mount.current
    if (!el) return
    let disposed = false
    let raf = 0
    let data: G1Preview | null = null
    let rig: RigPart[] = []
    let elapsed = 0
    let last = performance.now()
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x101516)
    const camera = new THREE.PerspectiveCamera(52, 1, 0.02, 100)
    camera.position.set(4.4, 3.2, 5.2)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.shadowMap.enabled = true
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    el.appendChild(renderer.domElement)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.target.set(0, 0.8, 0)
    scene.add(new THREE.HemisphereLight(0xffffff, 0x243036, 2.0))
    const key = new THREE.DirectionalLight(0xffffff, 2.8); key.position.set(-3, 6, 4); key.castShadow = true; scene.add(key)

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(18, 18), new THREE.MeshStandardMaterial({ color: 0x202829, roughness: 0.92 }))
    floor.rotation.x = -Math.PI / 2
    floor.position.y = plan?.scene.floor_y ?? 0
    floor.receiveShadow = true
    scene.add(floor)

    let splat: ReturnType<typeof createLightweightSplat> | null = null
    if (splatBuffer) {
      try {
        splat = createLightweightSplat(splatBuffer)
        scene.add(splat.object)
        setStatus(`Gaussian-splat preview · ${splat.renderedSplats.toLocaleString()} / ${splat.totalSplats.toLocaleString()} splats`)
      } catch (error) {
        setStatus(error instanceof Error ? error.message : 'Could not read .splat scene')
      }
    }

    if (plan?.path?.length) {
      const points = plan.path.map(([x, z]) => new THREE.Vector3(x, (plan.scene.floor_y ?? 0) + 0.025, z))
      const geometry = new THREE.BufferGeometry().setFromPoints(points)
      const material = new THREE.LineBasicMaterial({ color: 0xe0b459 })
      scene.add(new THREE.Line(geometry, material))
      const xs = plan.path.map(p => p[0]), zs = plan.path.map(p => p[1])
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2
      const cz = (Math.min(...zs) + Math.max(...zs)) / 2
      controls.target.set(cx, 0.8, cz)
    }

    const resize = () => {
      const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight)
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize); ro.observe(el); resize()

    const base = `${import.meta.env.BASE_URL || '/'}models/g1-native/`
    if (previewFile) {
      Promise.all([
        fetch(previewFile, { cache: 'no-store' }).then(async r => {
          if (!r.ok) throw new Error(`G1 preview request failed (${r.status})`)
          return r.json() as Promise<G1Preview>
        }),
        loadMeshTransforms(base),
      ]).then(async ([preview, transforms]) => {
        if (disposed) return
        if (preview.format !== 'g1-joints-v2') throw new Error('Scene Lab needs a g1-joints-v2 preview')
        data = preview
        const matIvory = new THREE.MeshStandardMaterial({ color: 0xe9e4da, roughness: 0.5, metalness: 0.15 })
        const matGold = new THREE.MeshStandardMaterial({ color: 0xb59657, roughness: 0.42, metalness: 0.28 })
        rig = await Promise.all([...FILE_TO_JOINT.entries()].map(async ([meshFile, joint]) => {
          const transform = transforms.get(meshFile) || { pos: new THREE.Vector3(), quat: new THREE.Quaternion() }
          const geometry = (await loadGeometry(`${base}meshes/${meshFile}`)).clone()
          const gold = /pelvis|hip_pitch|ankle_roll|logo_link|head_link/i.test(meshFile)
          const mesh = new THREE.Mesh(geometry, gold ? matGold : matIvory)
          mesh.castShadow = true; mesh.receiveShadow = true; mesh.frustumCulled = false
          scene.add(mesh)
          return { mesh, joint, geomPos: transform.pos.clone(), geomQuat: transform.quat.clone() }
        }))
        setStatus(splat ? `Scene-aware G1 · ${splat.renderedSplats.toLocaleString()} splats` : 'Scene-aware G1 motion · geometry proxy view')
      }).catch(error => { if (!disposed) setStatus(error instanceof Error ? error.message : 'Could not load scene-aware G1 motion') })
    }

    const tmpOffset = new THREE.Vector3(), tmpQuat = new THREE.Quaternion()
    const tick = (now: number) => {
      if (disposed) return
      raf = requestAnimationFrame(tick)
      const dt = Math.min(0.08, Math.max(0, (now - last) / 1000)); last = now
      if (data) {
        elapsed = (elapsed + dt) % (data.positions.length / data.fps)
        const frame = Math.min(data.positions.length - 1, Math.floor(elapsed * data.fps))
        for (const part of rig) {
          const p = data.positions[frame][part.joint]
          const q = matrixToQuat(data.global_rot_mats[frame][part.joint])
          tmpOffset.copy(part.geomPos).applyQuaternion(q)
          part.mesh.position.set(p[0], p[1], p[2]).add(tmpOffset)
          tmpQuat.copy(q).multiply(part.geomQuat)
          part.mesh.quaternion.copy(tmpQuat)
        }
      }
      controls.update()
      renderer.render(scene, camera)
    }
    raf = requestAnimationFrame(tick)

    return () => {
      disposed = true; cancelAnimationFrame(raf); ro.disconnect(); controls.dispose(); splat?.dispose()
      rig.forEach(part => part.mesh.geometry.dispose())
      scene.traverse(obj => {
        if (obj instanceof THREE.Mesh && obj !== floor) {
          const material = obj.material
          if (Array.isArray(material)) material.forEach(m => m.dispose())
        }
      })
      floor.geometry.dispose(); (floor.material as THREE.Material).dispose()
      renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove()
    }
  }, [plan, previewFile, splatBuffer])

  return <div className="scene-splat-viewer"><div ref={mount} className="scene-splat-canvas" /><div className="scene-splat-status">{status}</div></div>
}
