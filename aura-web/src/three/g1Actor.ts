/**
 * A G1 STL robot driven by NVIDIA Kimodo · Aura integration global joint rotations, played IN PLACE:
 * horizontal root travel is removed per frame, the clip is turned so its first
 * frame faces +z (the camera), and a windowed floor estimate keeps terrain clips
 * (stairs) on the pedestal. Root height and every rotation are otherwise kept.
 */
import * as THREE from 'three'
import { FILE_TO_JOINT, loadGeometry, loadMeshTransforms, type G1Preview } from './g1Rig'

const J = 34

export type PreparedClip = {
  frames: number
  fps: number
  duration: number
  /** frames * 34 * 3, in-place positions */
  pos: Float32Array
  /** frames * 34 * 4 (x, y, z, w), heading-normalised global rotations */
  quat: Float32Array
}

export function prepareClip(preview: G1Preview): PreparedClip {
  if (preview.format !== 'g1-joints-v2' || !preview.positions?.length || !preview.global_rot_mats?.length) {
    throw new Error('Hero clip needs g1-joints-v2 rotations')
  }
  const frames = preview.positions.length
  const fps = preview.fps || 30
  const pos = new Float32Array(frames * J * 3)
  const quat = new Float32Array(frames * J * 4)

  // Heading of the first frame: pelvis forward (+z local) projected on the ground.
  const r0 = preview.global_rot_mats[0][0]
  const yaw0 = Math.atan2(r0[0][2], r0[2][2])
  const unYaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -yaw0)
  const m = new THREE.Matrix4()
  const q = new THREE.Quaternion()
  const v = new THREE.Vector3()

  // Per-frame lowest joint, then a +-0.5 s windowed minimum as the "floor".
  const minY = new Float32Array(frames)
  for (let t = 0; t < frames; t++) {
    let lo = Infinity
    for (const p of preview.positions[t]) lo = Math.min(lo, p[1])
    minY[t] = lo
  }
  const half = Math.round(fps * 0.5)
  const floor = new Float32Array(frames)
  for (let t = 0; t < frames; t++) {
    let lo = Infinity
    for (let k = Math.max(0, t - half); k <= Math.min(frames - 1, t + half); k++) lo = Math.min(lo, minY[k])
    floor[t] = lo
  }

  for (let t = 0; t < frames; t++) {
    const frame = preview.positions[t]
    const rots = preview.global_rot_mats[t]
    const rx = frame[0][0], rz = frame[0][2]
    for (let j = 0; j < J; j++) {
      const p = frame[j]
      v.set(p[0] - rx, p[1] - floor[t], p[2] - rz).applyQuaternion(unYaw)
      pos.set([v.x, v.y, v.z], (t * J + j) * 3)
      const r = rots[j]
      m.set(r[0][0], r[0][1], r[0][2], 0, r[1][0], r[1][1], r[1][2], 0, r[2][0], r[2][1], r[2][2], 0, 0, 0, 0, 1)
      q.setFromRotationMatrix(m).premultiply(unYaw).normalize()
      quat.set([q.x, q.y, q.z, q.w], (t * J + j) * 4)
    }
  }
  return { frames, fps, duration: frames / fps, pos, quat }
}

export type ActorPalette = { body: number; joint: number; outline: number }

type ActorPart = { mesh: THREE.Mesh; joint: number; geomPos: THREE.Vector3; geomQuat: THREE.Quaternion }

const JOINT_PARTS = /pelvis\.STL|pelvis_contour|hip_pitch|ankle_roll|logo_link|head_link/i

/** Loads every STL once per arena; both actors share these clones. */
export async function loadArenaGeometries(base: string) {
  // The rig XML and the STL meshes are independent downloads: fetch them together.
  const [transforms, geoms] = await Promise.all([
    loadMeshTransforms(base),
    Promise.all([...FILE_TO_JOINT.keys()].map(async file => [file, (await loadGeometry(`${base}meshes/${file}`)).clone()] as const)),
  ])
  return { transforms, geometries: new Map(geoms) }
}

export class G1Actor {
  static RIM = new THREE.Color(0xfff2dc)
  readonly group = new THREE.Group()
  readonly meshes: THREE.Mesh[] = []
  private parts: ActorPart[] = []
  private bodyMat: THREE.MeshToonMaterial
  private jointMat: THREE.MeshToonMaterial
  private outlineMat: THREE.MeshBasicMaterial
  private baseBody = new THREE.Color()
  private baseJoint = new THREE.Color()
  private dimTint = new THREE.Color()
  private clip: PreparedClip | null = null
  private tmpA = new THREE.Quaternion()
  private tmpB = new THREE.Quaternion()
  private tmpV = new THREE.Vector3()
  private tmpP = new THREE.Vector3()

  constructor(
    assets: { transforms: Map<string, { pos: THREE.Vector3; quat: THREE.Quaternion }>; geometries: Map<string, THREE.BufferGeometry> },
    palette: ActorPalette,
    marble: THREE.Texture | null,
    dimColor: number,
  ) {
    this.baseBody.set(palette.body)
    this.baseJoint.set(palette.joint)
    this.dimTint.set(dimColor)
    this.bodyMat = new THREE.MeshToonMaterial({ color: palette.body, map: marble, emissive: 0x14100c, emissiveIntensity: 0.025, side: THREE.DoubleSide })
    this.jointMat = new THREE.MeshToonMaterial({ color: palette.joint, emissive: 0x1f1912, emissiveIntensity: 0.06, side: THREE.DoubleSide })
    this.outlineMat = new THREE.MeshBasicMaterial({ color: palette.outline, side: THREE.BackSide, transparent: true, opacity: 0.72 })
    for (const [file, joint] of FILE_TO_JOINT) {
      const geometry = assets.geometries.get(file)
      if (!geometry) continue
      const t = assets.transforms.get(file) || { pos: new THREE.Vector3(), quat: new THREE.Quaternion() }
      const mesh = new THREE.Mesh(geometry, JOINT_PARTS.test(file) ? this.jointMat : this.bodyMat)
      mesh.name = file // lets effects find limbs (e.g. robotFlare effectorsByName)
      mesh.castShadow = true
      mesh.frustumCulled = false
      const outline = new THREE.Mesh(geometry, this.outlineMat)
      outline.scale.setScalar(1.013)
      outline.renderOrder = -1
      outline.frustumCulled = false
      outline.raycast = () => {}
      mesh.add(outline)
      this.group.add(mesh)
      this.meshes.push(mesh)
      this.parts.push({ mesh, joint, geomPos: t.pos.clone(), geomQuat: t.quat.clone() })
    }
  }

  get duration() { return this.clip?.duration ?? 0 }

  setClip(clip: PreparedClip | null) {
    this.clip = clip
    this.group.visible = !!clip
    if (clip) this.setTime(0)
  }

  /** Pose at time t (seconds, wraps), linearly interpolated between frames. */
  setTime(t: number) {
    const c = this.clip
    if (!c) return
    const f = ((t % c.duration) + c.duration) % c.duration * c.fps
    const i0 = Math.min(c.frames - 1, Math.floor(f))
    const i1 = (i0 + 1) % c.frames
    const a = i1 === 0 ? 0 : f - i0
    for (const part of this.parts) {
      const o0 = (i0 * J + part.joint), o1 = (i1 * J + part.joint)
      this.tmpA.set(c.quat[o0 * 4], c.quat[o0 * 4 + 1], c.quat[o0 * 4 + 2], c.quat[o0 * 4 + 3])
      this.tmpB.set(c.quat[o1 * 4], c.quat[o1 * 4 + 1], c.quat[o1 * 4 + 2], c.quat[o1 * 4 + 3])
      this.tmpA.slerp(this.tmpB, a)
      this.tmpP.set(c.pos[o0 * 3], c.pos[o0 * 3 + 1], c.pos[o0 * 3 + 2])
        .lerp(this.tmpV.set(c.pos[o1 * 3], c.pos[o1 * 3 + 1], c.pos[o1 * 3 + 2]), a)
      this.tmpV.copy(part.geomPos).applyQuaternion(this.tmpA)
      part.mesh.position.copy(this.tmpP).add(this.tmpV)
      part.mesh.quaternion.copy(this.tmpA).multiply(part.geomQuat)
    }
  }

  /** highlight 0..1 (hover rim), dim 0..1 (lost the vote). */
  setLook(highlight: number, dim: number) {
    this.bodyMat.color.copy(this.baseBody).lerp(this.dimTint, dim * 0.55)
    this.jointMat.color.copy(this.baseJoint).lerp(this.dimTint, dim * 0.45)
    this.bodyMat.emissive.setHex(0x14100c).lerp(G1Actor.RIM, highlight)
    this.jointMat.emissive.setHex(0x1f1912).lerp(G1Actor.RIM, highlight)
    this.bodyMat.emissiveIntensity = 0.025 + highlight * 0.22
    this.jointMat.emissiveIntensity = 0.06 + highlight * 0.18
    this.outlineMat.opacity = 0.72 + highlight * 0.28 - dim * 0.4
  }

  dispose() {
    this.bodyMat.dispose(); this.jointMat.dispose(); this.outlineMat.dispose()
    this.group.removeFromParent()
  }
}
