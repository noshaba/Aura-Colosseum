/**
 * The Blossom Fairy character (public/models/custom/blossom-fairy.rigged.web.glb: blossom-fairy.web.glb, a lighter copy of
 * `aura-web/models-src/Blossom Fairy Running.glb`: unused metallic-roughness map dropped, textures re-encoded as WebP, plus finger and wing bones)
 * driven by G1Skeleton34 motion.
 *
 * Every Aura motion is recorded on the Unitree G1 skeleton (per-joint world
 * positions + global rotations, Y-up, +Z forward, character-left = +X). The fairy
 * is a Mixamo-rigged skinned mesh (`mixamorig:*` bones) whose bind pose is a
 * running lean (pelvis and spine pitched forward, arms in an A-pose), so rotations
 * are retargeted through a "neutral" pose:
 *
 *   world_bone(t) = R_g1(t) * inverse(R_g1_rest) * N_bone
 *
 * N_bone is the fairy bone's world rotation once it is laid along the G1 rest pose:
 * pelvis, spine, neck and head upright; legs straight down; upper arms down;
 * forearms and hands forward; feet flat facing +Z. It is computed once at load
 * from the bind pose, so the G1 rest pose maps exactly onto that neutral pose.
 * Limb twist is kept by matching the G1 joint's hinge axis (local +X, the pitch
 * axis of every G1 hinge) with the fairy limb's lateral axis.
 *
 * Forearms take only part of the G1 wrist roll (the hand takes all of it), so the
 * twist is spread over the forearm instead of wringing the elbow. Clavicles keep their
 * bind pose relative to Spine2 (the auto-rig skins chest and hair to them, so lifting
 * them shears the torso).
 *
 * Root: Hips follow the G1 pelvis scaled by the leg-length ratio, then a vertical
 * correction keeps the lower foot at the height the G1's lower ankle has (scaled),
 * so feet neither float nor sink.
 *
 * The rigged GLB (blossom-fairy.rigged.web.glb, built from blossom-fairy.web.glb by
 * adding bones and re-skinning; geometry, materials and textures unchanged) adds:
 * - fingers: `mixamorig:{Left,Right}Hand{Thumb,Index,Middle,Ring,Pinky}{1,2,3}`. G1 has
 *   no finger joints, so the curl is procedural from the arm's context (relaxed, open
 *   when raised, fist when pumping, grip when held still in front): updateHandContext.
 * - wings: `wing_{L,R}` + `wing_{L,R}_tip` under Spine2 (the wing verts were skinned to
 *   the Head before). They beat on the skeleton from the mesh's onBeforeRender, so every
 *   viewer flutters; reduced motion holds them still: animateWings.
 * - smoothed weights across the knee / elbow / hip / shoulder bands.
 * The GLB's `Running` clip is not used. The extra bones (Bone_019/020/024/025 skin the
 * wrist cuffs; Bone_029/030 and headfront carry no weights) keep their bind pose
 * relative to their parents.
 */
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { clone as cloneSkinned } from 'three/examples/jsm/utils/SkeletonUtils.js'
import { prefersReducedMotion } from './auraWorld'
import { markWingWeights, toonStylizeMaterial } from './toonStylize'
import { CHARACTERS, getCurrentCharacter, type MixamoCharacter } from './characters'

/**
 * The Mixamo character used when a caller passes none: the selected one if it is a
 * Mixamo body, else the first Mixamo entry of the registry.
 */
function fallbackMixamo(): MixamoCharacter {
  const cur = getCurrentCharacter()
  if (cur.kind === 'mixamo-retarget') return cur
  const first = CHARACTERS.find((c): c is MixamoCharacter => c.kind === 'mixamo-retarget')
  if (!first) throw new Error('characters.ts lists no mixamo-retarget character')
  return first
}

/** Bone name without the Mixamo prefix, in whatever form the loader left it (`mixamorig:Hips`, `mixamorigHips`, `mixamorig_Hips`). */
const canonicalBone = (name: string) => name.replace(/^mixamorig[:_]?/i, '')

const G1_JOINTS = 34

/**
 * G1Skeleton34 rest pose in Three axes, generated from public/models/g1-native/g1.xml
 * (toe_base / hand_roll endpoints from clip data). Positions in metres.
 */
const G1_REST_POS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 0.793, 0], [0.06445, 0.6903, 0], [0.11645, 0.65984, 0], [0.11645, 0.54196, 0.04622], [0.1186, 0.3537, 0], [0.11851, 0.05369, 0], [0.11851, 0.03614, 0], [0.11851, 0.00114, 0.14],
  [-0.06445, 0.6903, 0], [-0.11645, 0.65984, 0], [-0.11645, 0.54196, 0.04622], [-0.1186, 0.3537, 0], [-0.11851, 0.05369, 0], [-0.11851, 0.03614, 0], [-0.11851, 0.00114, 0.14],
  [0, 0.793, 0], [0, 0.837, -0.00396], [0, 0.837, -0.00396],
  [0.10022, 1.08478, 0], [0.14056, 1.08196, 0], [0.14681, 0.97876, 0], [0.14681, 0.89824, 0.01577], [0.14868, 0.88824, 0.11577], [0.14867, 0.88824, 0.15377], [0.14866, 0.88823, 0.19977], [0.14864, 0.88822, 0.29976],
  [-0.10021, 1.08478, 0], [-0.14055, 1.08196, 0], [-0.1468, 0.97876, 0], [-0.1468, 0.89824, 0.01577], [-0.14867, 0.88824, 0.11577], [-0.14866, 0.88824, 0.15377], [-0.14865, 0.88823, 0.19977], [-0.14863, 0.88822, 0.29977],
]
/** Non-identity G1 rest rotations (x, y, z, w): hip roll/yaw carry a -10 deg pitch, shoulder pitch a +-16 deg roll. */
const G1_REST_QUAT: Record<number, readonly [number, number, number, number]> = {
  2: [-0.08734, 0, 0, 0.99618], 3: [-0.08734, 0, 0, 0.99618],
  9: [-0.08734, 0, 0, 0.99618], 10: [-0.08734, 0, 0, 0.99618],
  18: [0, 0, 0.1392, 0.99026], 26: [0, 0, -0.1392, 0.99026],
}
const G1_ANKLE = { L: 6, R: 13 }


/**
 * Fairy bone -> G1 joint whose global rotation drives it, and the G1 rest segment
 * (from joint -> to joint) the fairy bone is laid along in the neutral pose.
 * Spine/Spine1/Spine2 blend pelvis -> torso (the three G1 waist joints composed);
 * Neck/Head follow the torso. Shoulders (clavicles), toes and the extra bones keep
 * their bind pose relative to their parent.
 */
type LimbSpec = { bone: string; joint: number; seg?: [number, number]; foot?: boolean; chainRoot?: boolean; twistFrom?: number; twistK?: number }
/** Share of the G1 wrist roll the forearm takes (the rest twists at the wrist, under the cuff). */
const FOREARM_TWIST = 0.4
const LIMBS: LimbSpec[] = [
  { bone: 'LeftUpLeg', joint: 3, seg: [2, 4], chainRoot: true },
  { bone: 'LeftLeg', joint: 4, seg: [4, 5] },
  { bone: 'LeftFoot', joint: 6, foot: true },
  { bone: 'RightUpLeg', joint: 10, seg: [9, 11], chainRoot: true },
  { bone: 'RightLeg', joint: 11, seg: [11, 12] },
  { bone: 'RightFoot', joint: 13, foot: true },
  { bone: 'LeftArm', joint: 20, seg: [19, 21], chainRoot: true },
  { bone: 'LeftForeArm', joint: 22, seg: [21, 24], twistFrom: 21, twistK: FOREARM_TWIST },
  { bone: 'LeftHand', joint: 24, seg: [24, 25] },
  { bone: 'RightArm', joint: 28, seg: [27, 29], chainRoot: true },
  { bone: 'RightForeArm', joint: 30, seg: [29, 32], twistFrom: 29, twistK: FOREARM_TWIST },
  { bone: 'RightHand', joint: 32, seg: [32, 33] },
]
/** Upright in the neutral pose; blend factor 0 = pelvis, 1 = torso. */
const AXIAL: [string, number][] = [['Spine', 1 / 3], ['Spine1', 2 / 3], ['Spine2', 1], ['Neck', 1], ['Head', 1]]
const PELVIS = 0
const TORSO = 17

/** One G1 pose sample: 34 x (x, y, z) positions and 34 x (x, y, z, w) global rotations, Y-up. */
export type G1PoseSample = { pos: Float32Array; quat: Float32Array }
export const createPoseSample = (): G1PoseSample => ({ pos: new Float32Array(G1_JOINTS * 3), quat: new Float32Array(G1_JOINTS * 4) })

/** Fill a sample from raw g1-joints-v2 arrays (rotation matrices), optionally subtracting a root xz offset. */
export function sampleFromPreview(positions: number[][], rotations: number[][][], out: G1PoseSample, rootX = 0, rootZ = 0) {
  for (let j = 0; j < G1_JOINTS; j++) {
    const p = positions[j]
    out.pos[j * 3] = p[0] - rootX; out.pos[j * 3 + 1] = p[1]; out.pos[j * 3 + 2] = p[2] - rootZ
    const r = rotations[j]
    tmpM.set(r[0][0], r[0][1], r[0][2], 0, r[1][0], r[1][1], r[1][2], 0, r[2][0], r[2][1], r[2][2], 0, 0, 0, 0, 1)
    tmpQ.setFromRotationMatrix(tmpM).normalize()
    out.quat[j * 4] = tmpQ.x; out.quat[j * 4 + 1] = tmpQ.y; out.quat[j * 4 + 2] = tmpQ.z; out.quat[j * 4 + 3] = tmpQ.w
  }
  return out
}

/**
 * Positions-only G1 data (g1-joints-v1 has no rotations): estimate the global
 * rotations the fairy retarget reads. Pelvis and torso frames come from the hip /
 * shoulder lines; each limb joint gets the rotation that lays its rest segment along
 * the current one, with twist from the knee / elbow bend plane (falling back to the
 * parent's frame when the limb is nearly straight).
 */
type PosSeg = { joint: number; parent: number; from: number; to: number; hinge?: 'knee' | 'elbow'; next?: [number, number] }
const POS_SEGS: PosSeg[] = [
  { joint: 3, parent: 0, from: 2, to: 4, hinge: 'knee', next: [4, 5] },
  { joint: 4, parent: 3, from: 4, to: 5, hinge: 'knee', next: [4, 5] },
  { joint: 6, parent: 4, from: 6, to: 7 },
  { joint: 10, parent: 0, from: 9, to: 11, hinge: 'knee', next: [11, 12] },
  { joint: 11, parent: 10, from: 11, to: 12, hinge: 'knee', next: [11, 12] },
  { joint: 13, parent: 11, from: 13, to: 14 },
  { joint: 20, parent: 17, from: 19, to: 21, hinge: 'elbow', next: [21, 24] },
  { joint: 22, parent: 20, from: 21, to: 24, hinge: 'elbow', next: [21, 24] },
  { joint: 24, parent: 22, from: 24, to: 25 },
  { joint: 28, parent: 17, from: 27, to: 29, hinge: 'elbow', next: [29, 32] },
  { joint: 30, parent: 28, from: 29, to: 32, hinge: 'elbow', next: [29, 32] },
  { joint: 32, parent: 30, from: 32, to: 33 },
]
const posRestFrameInv = new Map<number, THREE.Quaternion>()
const pv = (positions: number[][], j: number, out: THREE.Vector3) => out.set(positions[j][0], positions[j][1], positions[j][2])
const g1RestQuat = (j: number) => { const q = G1_REST_QUAT[j]; return q ? new THREE.Quaternion(q[0], q[1], q[2], q[3]) : new THREE.Quaternion() }
// Reusable scratch for sampleFromPositions (called every frame): the solved rotation of
// each joint it estimates (pelvis, torso, limb joints; the rest stay identity), and vectors.
const posSolved = new Map<number, THREE.Quaternion>([0, 17, ...POS_SEGS.map(s => s.joint)].map(j => [j, new THREE.Quaternion()]))
const pA = new THREE.Vector3(), pB = new THREE.Vector3(), pC = new THREE.Vector3(), pD = new THREE.Vector3(), pL = new THREE.Vector3()
/** Last solve (input frame + output sample): replaying the same frame (paused, or several rAFs per data frame) is free. */
let lastPositions: number[][] | null = null
let lastOut: G1PoseSample | null = null

export function sampleFromPositions(positions: number[][], out: G1PoseSample) {
  if (positions === lastPositions && out === lastOut) return out
  const R = posSolved
  const a = pA, b = pB, c = pC, d = pD, l = pL
  // Pelvis (the hip pitch joints are rigid with it): x = right hip -> left hip, y = hip midpoint -> pelvis.
  pv(positions, 1, a).sub(pv(positions, 8, b))
  pv(positions, 0, c).sub(pv(positions, 1, b).add(pv(positions, 8, d)).multiplyScalar(0.5))
  frameQuat(c, a, R.get(0)!)
  // Torso: x = right shoulder -> left shoulder, y = waist -> shoulder midpoint.
  pv(positions, 18, a).sub(pv(positions, 26, b))
  pv(positions, 18, c).add(pv(positions, 26, d)).multiplyScalar(0.5).sub(pv(positions, 16, b))
  frameQuat(c, a, R.get(17)!)
  for (const s of POS_SEGS) {
    let restInv = posRestFrameInv.get(s.joint)
    if (!restInv) {
      const rd = new THREE.Vector3(...G1_REST_POS[s.to]).sub(new THREE.Vector3(...G1_REST_POS[s.from]))
      restInv = frameQuat(rd, X.clone().applyQuaternion(g1RestQuat(s.joint)), new THREE.Quaternion()).invert().multiply(g1RestQuat(s.joint))
      posRestFrameInv.set(s.joint, restInv)
    }
    pv(positions, s.to, d).sub(pv(positions, s.from, a)).normalize()
    // Lateral from the parent's frame (no twist) ...
    l.copy(X).applyQuaternion(R.get(s.parent)!)
    // ... replaced by the hinge axis when the knee / elbow is bent.
    if (s.hinge) {
      // upper segment = chain root -> hinge joint, lower = hinge joint -> end of the next segment
      const [h0, h1] = s.next!
      const chainFrom = s.hinge === 'knee' ? (s.joint === 3 || s.joint === 4 ? 2 : 9) : (s.joint === 20 || s.joint === 22 ? 19 : 27)
      pv(positions, h0, a).sub(pv(positions, chainFrom, b)).normalize()
      pv(positions, h1, c).sub(pv(positions, h0, b)).normalize()
      const n = s.hinge === 'knee' ? b.crossVectors(a, c) : b.crossVectors(c, a)
      const w = THREE.MathUtils.smoothstep(n.length(), 0.12, 0.35)
      if (w > 0) l.lerp(n.normalize(), w)
    }
    frameQuat(d, l, R.get(s.joint)!).multiply(restInv)
  }
  for (let j = 0; j < G1_JOINTS; j++) {
    const p = positions[j]
    out.pos[j * 3] = p[0]; out.pos[j * 3 + 1] = p[1]; out.pos[j * 3 + 2] = p[2]
    const q = R.get(j)
    if (q) { out.quat[j * 4] = q.x; out.quat[j * 4 + 1] = q.y; out.quat[j * 4 + 2] = q.z; out.quat[j * 4 + 3] = q.w }
    else { out.quat[j * 4] = 0; out.quat[j * 4 + 1] = 0; out.quat[j * 4 + 2] = 0; out.quat[j * 4 + 3] = 1 }
  }
  lastPositions = positions; lastOut = out
  return out
}

const tmpM = new THREE.Matrix4()
const tmpQ = new THREE.Quaternion()
const tmpQ2 = new THREE.Quaternion()
const tmpQ3 = new THREE.Quaternion()
const tmpV = new THREE.Vector3()
const tmpV2 = new THREE.Vector3()
const tmpZ = new THREE.Vector3()
const X = new THREE.Vector3(1, 0, 0)
const Y = new THREE.Vector3(0, 1, 0)

/** Rotation whose Y axis is `dir` and X axis is `lateral` (orthogonalised). */
function frameQuat(dir: THREE.Vector3, lateral: THREE.Vector3, out: THREE.Quaternion) {
  const y = tmpV.copy(dir).normalize()
  const x = tmpV2.copy(lateral).addScaledVector(y, -lateral.dot(y))
  if (x.lengthSq() < 1e-8) x.set(1, 0, 0).addScaledVector(y, -y.x)
  x.normalize()
  const z = tmpZ.crossVectors(x, y)
  tmpM.makeBasis(x, y, z)
  return out.setFromRotationMatrix(tmpM)
}


/** A parsed Mixamo-rigged GLB, shared by every viewer (they clone it with `new FairyRig(asset, def)`). */
export type FairyAsset = { scene: THREE.Group; height: number; baseMaterial: THREE.MeshStandardMaterial | null }
export type MixamoAsset = FairyAsset

/** Resolved URL -> one download/parse, kept for the session (entries are dropped on failure so a retry refetches). */
const assetCache = new Map<string, Promise<MixamoAsset>>()

/** One cached download/parse of a mixamo-retarget character's GLB (`def.url`, relative to `base`). */
export function loadMixamoAsset(def: MixamoCharacter, base = import.meta.env.BASE_URL || '/'): Promise<MixamoAsset> {
  const url = `${base.endsWith('/') ? base : `${base}/`}${def.url}`
  const cached = assetCache.get(url)
  if (cached) return cached
  const promise = new GLTFLoader().loadAsync(url).then(gltf => {
    const scene = gltf.scene
    scene.updateMatrixWorld(true)
    let height = 1.64
    let baseMaterial: THREE.MeshStandardMaterial | null = null
    scene.traverse(o => {
      const mesh = o as THREE.SkinnedMesh
      if (!mesh.isSkinnedMesh) return
      mesh.geometry.computeBoundingBox()
      const bb = mesh.geometry.boundingBox!
      height = bb.max.y - bb.min.y
      const m = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material
      if ((m as THREE.MeshStandardMaterial).isMeshStandardMaterial) baseMaterial = m as THREE.MeshStandardMaterial
    })
    return { scene, height, baseMaterial }
  })
  assetCache.set(url, promise)
  promise.catch(() => { if (assetCache.get(url) === promise) assetCache.delete(url) })
  return promise
}

/**
 * Per-viewer fairy material: Aura's toon shading over her baked base colour and
 * normal map, so she reads in the comic world while keeping her own colours.
 * The textures belong to the cached asset; dispose only the material.
 * Wings and fingers are skeletal now, so the material carries no shader patch.
 */
export function createFairyMaterial(asset: FairyAsset, def: MixamoCharacter = fallbackMixamo()) {
  const material = new THREE.MeshToonMaterial({
    color: 0xffffff,
    map: asset.baseMaterial?.map ?? null, // her normal map is not used: the painted pass drops normal maps
    emissive: 0x14100c,
    emissiveIntensity: 0.025,
    side: THREE.DoubleSide,
  })
  // Painted-character pass (toonStylize.ts): palette bake of the photo texture (cached and shared,
  // kept for the session like the GLB itself), two-tone sun + hue-kept shade + rims, wing panes
  // from the wing bones' skin weights, and the composite's character tag. Chains onBeforeCompile /
  // customProgramCacheKey.
  let wings = false
  const wingRe = def.features.wings
  if (wingRe) asset.scene.traverse(o => { const sm = o as THREE.SkinnedMesh; if (sm.isSkinnedMesh) wings = markWingWeights(sm, wingRe) || wings })
  toonStylizeMaterial(material, { wings, keepBake: true })
  return material
}

/**
 * Dress a viewer's fairy clone: the toon body material on every skinned mesh, cast shadows,
 * and the ink inverted-hull outline. Returns the per-viewer materials, which the viewer
 * disposes (geometry and textures belong to the cached asset).
 */
export function dressFairy(rig: FairyRig, asset: FairyAsset, opts: { ink?: THREE.ColorRepresentation; receiveShadow?: boolean } = {}) {
  const body = createFairyMaterial(asset, rig.def)
  const ink = new THREE.MeshBasicMaterial({ color: opts.ink ?? 0x2a2c40, transparent: true, opacity: 0.72 })
  for (const mesh of rig.meshes) {
    mesh.material = body
    mesh.castShadow = true
    if (opts.receiveShadow) mesh.receiveShadow = true
    mesh.add(createSkinnedOutline(mesh, ink, 0.006))
  }
  return { body, ink }
}

type DrivenLimb = { bone: THREE.Bone; joint: number; twistFrom: number; twistK: number; restInv: THREE.Quaternion; neutral: THREE.Quaternion }
type Finger = { bone: THREE.Bone; restLocal: THREE.Quaternion; seg: number; finger: number }
type Hand = {
  side: 'L' | 'R'; hand: number; shoulder: number; fingers: Finger[]
  curl: Float32Array // smoothed curl per finger slot x segment (5 x 3)
  prev: THREE.Vector3; hasPrev: boolean
}
type Wing = { root: THREE.Bone; tip: THREE.Bone | null; rootRest: THREE.Quaternion; tipRest: THREE.Quaternion; axis: THREE.Vector3; sign: number }

const FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'] as const
/** Curl (rad, about each finger bone's local X, + = toward the palm) per segment: proximal, middle, distal. */
const CURL = {
  relaxed: [0.32, 0.42, 0.28],
  open: [0.06, 0.08, 0.04],
  fist: [1.25, 1.45, 0.95],
  grip: [0.95, 1.1, 0.75],
}
/** Per finger multiplier (thumb, index ... pinky): pinky side curls a little more, like a real hand at rest. */
const FINGER_K = [0.45, 0.85, 1, 1.1, 1.22]
const WING_HZ = 2.4

/**
 * One fairy instance: its own bones (SkeletonUtils clone), shared geometry and
 * textures, and the retarget from a G1PoseSample.
 */
export class FairyRig {
  /** The cloned glTF scene, in rig space (metres, feet at y = 0 in bind pose). */
  readonly root: THREE.Object3D
  readonly meshes: THREE.SkinnedMesh[] = []
  /** Bones by name without the Mixamo prefix (`Hips`, `LeftHand`, `LeftHandIndex1`, `wing_L`, ...). */
  readonly bones = new Map<string, THREE.Bone>()
  /** Fairy leg length / G1 leg length. */
  readonly scale: number
  readonly height: number
  /** The registry entry this instance was built for (features switch fingers / wings on). */
  readonly def: MixamoCharacter
  private order: THREE.Bone[] = []
  private world = new Map<THREE.Object3D, THREE.Quaternion>()
  private driven = new Map<THREE.Bone, DrivenLimb>()
  private axial = new Map<THREE.Bone, { neutral: THREE.Quaternion; k: number }>()
  private hands: Hand[] = []
  private wings: Wing[] = []
  private hips: THREE.Bone
  private hipsNeutral: THREE.Quaternion
  private hipsParentInv = new THREE.Matrix4()
  private footRestY: number
  private legs: { side: 'L' | 'R'; up: THREE.Bone; leg: THREE.Bone; foot: THREE.Bone }[] = []
  private reducedMotion = prefersReducedMotion()
  /** Motion energy 0..1 (pelvis speed), drives the wing beat; smoothed. */
  private energy = 0
  private prevPelvis = new THREE.Vector3()
  private lastPoseT = -1
  private lastWingT = -1
  private qa = new THREE.Quaternion()
  private qb = new THREE.Quaternion()
  private qc = new THREE.Quaternion()
  private va = new THREE.Vector3()
  private vb = new THREE.Vector3()
  private vc = new THREE.Vector3()

  constructor(asset: FairyAsset, def: MixamoCharacter = fallbackMixamo()) {
    this.def = def
    this.root = cloneSkinned(asset.scene)
    this.root.name = def.label.replace(/\W+/g, '') || def.id
    this.height = asset.height
    this.root.position.set(0, 0, 0); this.root.quaternion.identity(); this.root.scale.setScalar(1)
    this.root.updateMatrixWorld(true)
    this.root.traverse(o => {
      if ((o as THREE.Bone).isBone) this.bones.set(canonicalBone(o.name), o as THREE.Bone)
      const mesh = o as THREE.SkinnedMesh
      if (mesh.isSkinnedMesh) {
        mesh.frustumCulled = false // the bind-pose bounds don't follow the retargeted pose
        mesh.userData.sharedGeometry = true // owned by the cached asset: viewers must not dispose it
        mesh.onBeforeRender = () => this.animateWings(performance.now() / 1000) // wing beat clock (every viewer)
        this.meshes.push(mesh)
      }
    })
    const need = (name: string) => {
      const b = this.bones.get(name)
      if (!b) throw new Error(`Fairy rig is missing bone ${name}`)
      return b
    }
    this.hips = need('Hips')
    // Hierarchy order (parents first) for the world-rotation pass.
    const top = [...this.bones.values()].filter(b => !(b.parent as THREE.Bone | null)?.isBone)
    for (const b of top) {
      b.traverse(o => { if ((o as THREE.Bone).isBone) this.order.push(o as THREE.Bone) })
      // Non-bone ancestors never move: keep their rig-space rotation as the chain's base.
      const q = new THREE.Quaternion()
      b.parent?.matrixWorld.decompose(new THREE.Vector3(), q, new THREE.Vector3())
      if (b.parent) this.world.set(b.parent, q)
    }

    // Bind-pose world (rig space) rotations / positions.
    const restQ = new Map<THREE.Bone, THREE.Quaternion>()
    const restP = new Map<THREE.Bone, THREE.Vector3>()
    for (const b of this.order) {
      const p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3()
      b.matrixWorld.decompose(p, q, s)
      restQ.set(b, q); restP.set(b, p)
    }
    const g1Pos = (j: number) => new THREE.Vector3(...G1_REST_POS[j])
    const g1RestQ = (j: number) => { const q = G1_REST_QUAT[j]; return q ? new THREE.Quaternion(q[0], q[1], q[2], q[3]) : new THREE.Quaternion() }
    /** Bind rotation with the bone's Y axis swung to vertical (removes the running lean). */
    const upright = (bone: THREE.Bone) => {
      const qr = restQ.get(bone)!
      return new THREE.Quaternion().setFromUnitVectors(Y.clone().applyQuaternion(qr), Y).multiply(qr)
    }

    // Neutral pose for each limb bone; lateral axes are parallel-transported down each chain.
    let chainLateral = new THREE.Vector3()
    let chainDir = new THREE.Vector3()
    const fA = new THREE.Quaternion(), fB = new THREE.Quaternion()
    for (const spec of LIMBS) {
      const bone = need(spec.bone)
      const qr = restQ.get(bone)!
      const d0 = Y.clone().applyQuaternion(qr)
      let neutral: THREE.Quaternion
      if (spec.foot) {
        // Feet are flat in the bind pose; only remove the toe-out yaw.
        const toe = bone.children.find(c => (c as THREE.Bone).isBone)
        const h = toe ? restP.get(toe as THREE.Bone)!.clone().sub(restP.get(bone)!) : d0.clone()
        const yaw = Math.atan2(h.x, h.z)
        neutral = new THREE.Quaternion().setFromAxisAngle(Y, -yaw).multiply(qr)
      } else {
        let l0: THREE.Vector3
        if (spec.chainRoot) {
          // A limb abducted from hanging straight down (no axial twist) has its hinge
          // along +Z x dir: +X for legs and arms down, tilted with an A-pose.
          l0 = new THREE.Vector3(0, 0, 1).cross(d0)
          if (l0.lengthSq() < 0.09) l0 = X.clone()
        } else {
          l0 = chainLateral.clone().applyQuaternion(new THREE.Quaternion().setFromUnitVectors(chainDir, d0))
        }
        l0.addScaledVector(d0, -l0.dot(d0)).normalize()
        chainLateral = l0.clone(); chainDir = d0.clone()
        const [a, b] = spec.seg!
        const dN = g1Pos(b).sub(g1Pos(a)).normalize()
        const lN = X.clone().applyQuaternion(g1RestQ(spec.joint))
        frameQuat(dN, lN, fA)
        frameQuat(d0, l0, fB)
        neutral = fA.clone().multiply(fB.invert()).multiply(qr).normalize()
      }
      // Twist split (forearms): the G1 wrist roll is spread between forearm and hand instead of all at the elbow.
      this.driven.set(bone, { bone, joint: spec.joint, twistFrom: spec.twistFrom ?? -1, twistK: spec.twistK ?? 1, restInv: g1RestQ(spec.joint).invert(), neutral })
    }
    for (const [name, k] of AXIAL) this.axial.set(need(name), { neutral: upright(need(name)), k })
    this.hipsNeutral = upright(this.hips)
    this.hipsParentInv.copy(this.hips.parent!.matrixWorld).invert()

    // Hands: finger bones (def.features.fingers; without them the hand stays a single bone).
    if (def.features.fingers) for (const [side, pre, hand, shoulder] of [['L', 'Left', 24, 18], ['R', 'Right', 32, 26]] as const) {
      const fingers: Finger[] = []
      FINGERS.forEach((f, fi) => {
        for (let s = 0; s < 3; s++) {
          const bone = this.bones.get(`${pre}Hand${f}${s + 1}`)
          if (bone) fingers.push({ bone, restLocal: bone.quaternion.clone(), seg: s, finger: fi })
        }
      })
      const curl = new Float32Array(15)
      for (let i = 0; i < 15; i++) curl[i] = CURL.relaxed[i % 3] * FINGER_K[Math.floor(i / 3)]
      this.hands.push({ side, hand, shoulder, fingers, curl, prev: new THREE.Vector3(), hasPrev: false })
    }
    this.poseFingers() // relaxed hands even in viewers that never call applyPose

    // Wings (def.features.wings): root bones matching the pattern, each with an optional `<root>_tip`
    // child, flapping about the root's local Y (the hinge along the back). Side from the name (L / Left).
    const wingRe = def.features.wings
    if (wingRe) for (const [name, root] of this.bones) {
      if (!wingRe.test(name) || /_tip$/i.test(name)) continue
      const side: 'L' | 'R' = /(^|[^a-z])(l|left)([^a-z]|$)|left/i.test(name) ? 'L' : 'R'
      const tip = this.bones.get(`${name}_tip`) ?? null
      // Which way round sweeps the tip backward (-Z in rig space)? Test a small turn on the outward vector.
      const qw = restQ.get(root)!
      const out = new THREE.Vector3(side === 'L' ? 1 : -1, 0, 0)
      const axisW = Y.clone().applyQuaternion(qw)
      const turned = out.clone().applyAxisAngle(axisW, 0.3)
      this.wings.push({ root, tip, rootRest: root.quaternion.clone(), tipRest: tip ? tip.quaternion.clone() : new THREE.Quaternion(), axis: Y.clone(), sign: turned.z < 0 ? 1 : -1 })
    }

    // Leg-length ratio and foot height reference.
    for (const side of ['L', 'R'] as const) {
      const pre = side === 'L' ? 'Left' : 'Right'
      this.legs.push({ side, up: need(`${pre}UpLeg`), leg: need(`${pre}Leg`), foot: need(`${pre}Foot`) })
    }
    const fairyLeg = this.legs.reduce((sum, l) => sum + l.leg.position.length() + l.foot.position.length(), 0) / 2
    const g1Leg = g1Pos(4).distanceTo(g1Pos(2)) + g1Pos(5).distanceTo(g1Pos(4))
    this.scale = fairyLeg / g1Leg
    this.footRestY = this.legs.reduce((sum, l) => sum + restP.get(l.foot)!.y, 0) / 2
    for (const b of this.order) this.world.set(b, new THREE.Quaternion())
  }

  /** Pose the body from a G1 sample. Positions in the sample are G1 metres; output is rig space. */
  applyPose(sample: G1PoseSample) {
    const q = sample.quat, pos = sample.pos
    const g1 = (j: number, out: THREE.Quaternion) => out.set(q[j * 4], q[j * 4 + 1], q[j * 4 + 2], q[j * 4 + 3])
    const pelvisD = g1(PELVIS, this.qa) // pelvis/torso rest rotations are identity
    const torsoD = g1(TORSO, this.qb)
    for (const bone of this.order) {
      const parentW = this.world.get(bone.parent!)!
      const w = this.world.get(bone)!
      let target: THREE.Quaternion | null = null
      if (bone === this.hips) {
        target = tmpQ3.copy(pelvisD).multiply(this.hipsNeutral)
      } else {
        const limb = this.driven.get(bone)
        if (limb) {
          g1(limb.joint, tmpQ3)
          if (limb.twistFrom >= 0) tmpQ3.copy(g1(limb.twistFrom, this.qc).slerp(tmpQ3, limb.twistK))
          target = tmpQ3.multiply(limb.restInv).multiply(limb.neutral)
        } else {
          const ax = this.axial.get(bone)
          if (ax) target = tmpQ3.copy(pelvisD).slerp(torsoD, ax.k).multiply(ax.neutral)
        }
      }
      if (target) {
        target.normalize()
        w.copy(target)
        bone.quaternion.copy(tmpQ2.copy(parentW).invert().multiply(target))
      } else {
        w.copy(parentW).multiply(bone.quaternion)
      }
    }
    // Root: scaled pelvis, then lift/lower so the lower foot matches the lower G1 ankle.
    const s = this.scale
    const hipsP = this.va.set(pos[0] * s, pos[1] * s, pos[2] * s)
    let fairyLow = Infinity, targetLow = Infinity
    const hipsW = this.world.get(this.hips)!
    for (const leg of this.legs) {
      // FK: hips -> up leg -> leg -> foot (bone local translations are unscaled).
      const p = tmpV.copy(leg.up.position).applyQuaternion(hipsW).add(hipsP)
      p.add(tmpV2.copy(leg.leg.position).applyQuaternion(this.world.get(leg.up)!))
      p.add(tmpV2.copy(leg.foot.position).applyQuaternion(this.world.get(leg.leg)!))
      fairyLow = Math.min(fairyLow, p.y)
      const a = G1_ANKLE[leg.side]
      targetLow = Math.min(targetLow, this.footRestY + s * (pos[a * 3 + 1] - G1_REST_POS[a][1]))
    }
    hipsP.y += targetLow - fairyLow
    this.hips.position.copy(hipsP).applyMatrix4(this.hipsParentInv)

    // Motion context for hands and wings (G1 metres, real time between poses).
    const now = performance.now() / 1000
    const dt = this.lastPoseT < 0 ? 0 : THREE.MathUtils.clamp(now - this.lastPoseT, 0, 0.1)
    const jump = dt === 0 || now - this.lastPoseT > 0.25 // first pose or a seek: no velocity
    this.lastPoseT = now
    const pel = this.vb.set(pos[0], pos[1], pos[2])
    if (!jump && dt > 0) {
      const v = this.vc.copy(pel).sub(this.prevPelvis).length() / dt
      const e = THREE.MathUtils.smoothstep(v, 0.15, 1.6)
      this.energy += (e - this.energy) * Math.min(1, dt * 3)
    }
    this.prevPelvis.copy(pel)
    for (const hand of this.hands) this.updateHandContext(hand, sample, torsoD, dt, jump)
    this.poseFingers()
  }

  /**
   * Finger curl target from what the arm is doing (G1 has no finger joints): open when
   * the hand is raised (wave, reach up), a loose fist when the arms pump fast, a grip
   * when the hand is held in front of the body and nearly still (carrying), relaxed
   * otherwise. Smoothed so the hand never snaps.
   */
  private updateHandContext(hand: Hand, sample: G1PoseSample, torsoD: THREE.Quaternion, dt: number, jump: boolean) {
    const pos = sample.pos
    const hp = this.va.set(pos[hand.hand * 3] - pos[0], pos[hand.hand * 3 + 1] - pos[1], pos[hand.hand * 3 + 2] - pos[2])
    let speed = 0
    if (!jump && hand.hasPrev && dt > 0) speed = this.vb.copy(hp).sub(hand.prev).length() / dt
    hand.prev.copy(hp); hand.hasPrev = true
    // In the torso frame: z = forward, y = up.
    const local = this.vb.copy(hp).applyQuaternion(this.qc.copy(torsoD).invert())
    const raised = THREE.MathUtils.smoothstep(pos[hand.hand * 3 + 1] - pos[hand.shoulder * 3 + 1], -0.08, 0.1)
    const pump = THREE.MathUtils.smoothstep(speed, 1.2, 2.6) * (1 - raised)
    const front = THREE.MathUtils.smoothstep(local.z, 0.16, 0.3) * THREE.MathUtils.smoothstep(-(pos[hand.hand * 3 + 1] - pos[hand.shoulder * 3 + 1]), 0.05, 0.2)
    const grip = front * (1 - THREE.MathUtils.smoothstep(speed, 0.5, 1.2)) * (1 - raised)
    const rate = jump ? 1 : Math.min(1, dt * 6)
    for (let i = 0; i < 15; i++) {
      const s = i % 3, f = Math.floor(i / 3)
      let c = CURL.relaxed[s]
      c += (CURL.open[s] - c) * raised
      c += (CURL.fist[s] - c) * pump
      c += (CURL.grip[s] - c) * grip
      const target = c * FINGER_K[f]
      hand.curl[i] += (target - hand.curl[i]) * rate
    }
  }

  private poseFingers() {
    for (const hand of this.hands) for (const f of hand.fingers) {
      const a = hand.curl[f.finger * 3 + f.seg]
      f.bone.quaternion.copy(f.restLocal).multiply(tmpQ.setFromAxisAngle(X, a))
    }
  }

  /**
   * Wing beat on the skeleton (called from the mesh's onBeforeRender, so every viewer
   * flutters, even those that never call applyPose; the new angles land on the next
   * frame's matrix update). Tips sweep back and return, never forward into the body;
   * the tip bone lags the root (membrane follow-through); motion energy beats faster
   * and wider. Reduced motion: wings hold their rest pose.
   */
  private animateWings(t: number) {
    if (t === this.lastWingT || !this.wings.length) return
    this.lastWingT = t
    const e = this.reducedMotion ? 0 : this.energy
    const amp = this.reducedMotion ? 0 : 0.32 + 0.22 * e
    const w = Math.PI * 2 * (WING_HZ + 1.2 * e)
    const phase = t * w
    const root = amp * (0.5 + 0.5 * Math.sin(phase))
    const tip = amp * 0.55 * (0.5 + 0.5 * Math.sin(phase - 0.9))
    for (const wing of this.wings) {
      wing.root.quaternion.copy(wing.rootRest).multiply(tmpQ.setFromAxisAngle(wing.axis, root * wing.sign))
      wing.tip?.quaternion.copy(wing.tipRest).multiply(tmpQ.setFromAxisAngle(wing.axis, tip * wing.sign))
    }
  }

  /** Per-instance resources only (skeleton bone textures); geometry and textures are shared through the cached asset. */
  dispose() {
    // The skeletons are per clone (SkeletonUtils.clone): free their bone textures (the ink outlines share them).
    const skeletons = new Set<THREE.Skeleton>()
    for (const mesh of this.meshes) { mesh.onBeforeRender = () => {}; skeletons.add(mesh.skeleton) }
    skeletons.forEach(sk => sk.dispose())
    this.root.removeFromParent()
  }
}

/** Ink inverted-hull outline for a skinned mesh: same geometry and skeleton, pushed along the bind normal. */
export function createSkinnedOutline(mesh: THREE.SkinnedMesh, material: THREE.MeshBasicMaterial, width: number) {
  material.side = THREE.BackSide
  const prevCompile = material.onBeforeCompile
  const prevKey = material.customProgramCacheKey
  material.onBeforeCompile = (shader, renderer) => {
    prevCompile.call(material, shader, renderer)
    shader.uniforms.outlineWidth = { value: width }
    shader.vertexShader = 'uniform float outlineWidth;\n' + shader.vertexShader
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n  transformed += normalize( objectNormal ) * outlineWidth;')
  }
  material.customProgramCacheKey = () => `${prevKey.call(material)}|fairy-outline-${width}`
  const outline = new THREE.SkinnedMesh(mesh.geometry, material)
  outline.bind(mesh.skeleton, mesh.bindMatrix)
  outline.bindMode = mesh.bindMode
  outline.frustumCulled = false
  outline.renderOrder = -1
  outline.raycast = () => {}
  outline.name = `${mesh.name}_outline`
  outline.userData.sharedGeometry = true
  return outline
}
