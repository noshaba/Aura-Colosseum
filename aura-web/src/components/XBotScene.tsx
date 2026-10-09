import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { AURA_OVERLAY_LAYER, createAuraRenderer, createAuraWorld, prefersReducedMotion } from '../three/auraWorld'
import { RobotFlare } from '../three/robotFlare'
import { toonStylize, type ToonStylizeHandle } from '../three/toonStylize'
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { createBvhPoseBuffers, loadBVH, sampleBVHWorldPose } from '../bvh'
import type { BvhMotion, BvhPoseBuffers } from '../bvh'
import type { MotionQuality, MotionSide } from '../aistReferenceMotions'
import { STUDY_CLIP_SECONDS } from '../aistReferenceMotions'
import { FairyRig, ROBOT_MODEL, createFairyMaterial, loadFairyAsset } from '../three/fairyRig'

/** 'g1' resolves to the Blossom Fairy while ROBOT_MODEL === 'fairy' (see three/fairyRig.ts). */
export type Embodiment = 'g1' | 'xbot' | 'fairy'

const BODY_LABEL: Record<Embodiment, { short: string; loading: string; chip: string }> = {
  g1: { short: 'G1', loading: 'Unitree G1 geometry', chip: 'UNITREE G1' },
  xbot: { short: 'XBot', loading: 'XBot', chip: 'XBOT' },
  fairy: { short: 'Fairy', loading: 'Blossom Fairy', chip: 'BLOSSOM FAIRY' },
}

export type ViewPose = {
  azimuth: number
  polar: number
  distance: number
}

export const DEFAULT_VIEW_POSE: ViewPose = {
  azimuth: 0,
  polar: Math.PI * 0.49,
  distance: 5.35,
}

type Props = {
  embodiment?: Embodiment
  side: MotionSide
  quality: MotionQuality
  motionFile: string
  motionUrl: string
  degradationSeed: number
  startOffsetSeconds: number
  paused?: boolean
  playhead?: number
  autoRotate?: boolean
  showTrails?: boolean
  showLandmarks?: boolean
  resetViewSignal?: number
  viewPose: ViewPose
  onViewPoseChange: (pose: ViewPose) => void
}

type LoadState = 'loading' | 'ready' | 'error'
type RestPose = {
  quaternion: THREE.Quaternion
  position: THREE.Vector3
  worldQuaternion: THREE.Quaternion
  parentWorldQuaternion: THREE.Quaternion
  worldPosition: THREE.Vector3
}
type Rig = Record<string, THREE.Bone | undefined>
type Trail = {
  endEffector: THREE.Object3D
  color: THREE.Color
  segments: THREE.InstancedMesh<THREE.CylinderGeometry, THREE.MeshBasicMaterial>
  glow: THREE.InstancedMesh<THREE.CylinderGeometry, THREE.MeshBasicMaterial>
  marker: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>
  points: THREE.Vector3[]
  lastSample: number
}

type LegSide = 'Left' | 'Right'
type LegCalibration = {
  sourceHip: string
  sourceKnee: string
  sourceAnkle: string
  sourceToe: string
  targetThigh: TargetBoneName
  targetShin: TargetBoneName
  targetFoot: TargetBoneName
  correction: THREE.Quaternion
  targetRestAxis: THREE.Vector3
  targetRestPole: THREE.Vector3
  targetRestPlaneNormal: THREE.Vector3
  targetThighLength: number
  targetShinLength: number
  targetRestReachRatio: number
  sourceMaxLength: number
}
type RetargetCalibration = {
  sourceToTargetWorld: THREE.Quaternion
  sourceToTargetWorldInverse: THREE.Quaternion
  segmentCorrections: Map<TargetBoneName, THREE.Quaternion>
  embodiment: Embodiment
  legs: Record<LegSide, LegCalibration>
}

const TARGET_BONES = [
  'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
  'LeftUpLeg', 'LeftLeg', 'LeftFoot',
  'RightUpLeg', 'RightLeg', 'RightFoot',
] as const

type TargetBoneName = (typeof TARGET_BONES)[number]

/**
 * We map each XBot bone to the AIST++ joint whose WORLD transform contains the
 * equivalent motion. This intentionally folds AIST++ helper joints into the
 * nearest XBot joint (LHipJoint/RHipJoint and Neck/Neck1).
 */
const SOURCE_FOR_TARGET: Record<TargetBoneName, string> = {
  Hips: 'Hips',
  Spine: 'LowerBack',
  Spine1: 'Spine',
  Spine2: 'Spine1',
  Neck: 'Neck1',
  Head: 'Head',
  LeftShoulder: 'LeftShoulder',
  LeftArm: 'LeftArm',
  LeftForeArm: 'LeftForeArm',
  LeftHand: 'LeftHand',
  RightShoulder: 'RightShoulder',
  RightArm: 'RightArm',
  RightForeArm: 'RightForeArm',
  RightHand: 'RightHand',
  LeftUpLeg: 'LeftUpLeg',
  LeftLeg: 'LeftLeg',
  LeftFoot: 'LeftFoot',
  RightUpLeg: 'RightUpLeg',
  RightLeg: 'RightLeg',
  RightFoot: 'RightFoot',
}


/**
 * Long humanoid limbs are much more stable when we retarget the direction of
 * each physical segment instead of copying joint Euler/quaternion axes between
 * two rigs. AIST++ and Mixamo use different local bone frames, especially in
 * the hips and legs; raw orientation transfer is what caused the crossed,
 * corkscrewed knees in v1.4.
 */
const SWING_SEGMENTS: Partial<Record<TargetBoneName, { sourceChild: string; targetChild: string }>> = {
  LeftShoulder: { sourceChild: 'LeftArm', targetChild: 'LeftArm' },
  LeftArm: { sourceChild: 'LeftForeArm', targetChild: 'LeftForeArm' },
  LeftForeArm: { sourceChild: 'LeftHand', targetChild: 'LeftHand' },
  RightShoulder: { sourceChild: 'RightArm', targetChild: 'RightArm' },
  RightArm: { sourceChild: 'RightForeArm', targetChild: 'RightForeArm' },
  RightForeArm: { sourceChild: 'RightHand', targetChild: 'RightHand' },
  Neck: { sourceChild: 'Head', targetChild: 'Head' },
  LeftUpLeg: { sourceChild: 'LeftLeg', targetChild: 'LeftLeg' },
  LeftLeg: { sourceChild: 'LeftFoot', targetChild: 'LeftFoot' },
  LeftFoot: { sourceChild: 'LeftToeBase', targetChild: 'LeftToeBase' },
  RightUpLeg: { sourceChild: 'RightLeg', targetChild: 'RightLeg' },
  RightLeg: { sourceChild: 'RightFoot', targetChild: 'RightFoot' },
  RightFoot: { sourceChild: 'RightToeBase', targetChild: 'RightToeBase' },
}

const IK_LEG_BONES = new Set<TargetBoneName>([
  'LeftUpLeg', 'LeftLeg', 'LeftFoot',
  'RightUpLeg', 'RightLeg', 'RightFoot',
])

const identityQuaternion = new THREE.Quaternion()
const qParentInv = new THREE.Quaternion()
const qDesiredWorld = new THREE.Quaternion()
const qDesiredLocal = new THREE.Quaternion()
const qRestInv = new THREE.Quaternion()
const qDelta = new THREE.Quaternion()
const qAdjusted = new THREE.Quaternion()
const qNudge = new THREE.Quaternion()
const qFinalWorld = new THREE.Quaternion()
const qSwing = new THREE.Quaternion()
const qHeadStartLocal = new THREE.Quaternion()
const qHeadCurrentLocal = new THREE.Quaternion()
const qHeadParentInv = new THREE.Quaternion()
const qUpperBodyDeltaLocal = new THREE.Quaternion()
const qUpperBodyDeltaWorld = new THREE.Quaternion()
const qUpperBodyMappedWorld = new THREE.Quaternion()
const qUpperBodyTargetDeltaLocal = new THREE.Quaternion()
const qUpperBodyParentStartInv = new THREE.Quaternion()
const qUpperBodyTargetParentRestInv = new THREE.Quaternion()
const qUpperBodyDamped = new THREE.Quaternion()
const sourceAnimatedDirection = new THREE.Vector3()
const targetBaselineDirection = new THREE.Vector3()
const qBaselineWorld = new THREE.Quaternion()
const tempEuler = new THREE.Euler()
const tempVector = new THREE.Vector3()
const tempVector2 = new THREE.Vector3()
const spherical = new THREE.Spherical()

/**
 * FBXLoader sanitizes node names for Three.js PropertyBinding. In particular,
 * a Mixamo bone stored in the FBX as `mixamorig:Hips` is commonly exposed at
 * runtime as `mixamorigHips` (the colon is removed). Some exporters also use
 * underscores or an Armature prefix. Normalize all of those forms before
 * matching so the rig lookup works across Mixamo FBX variants.
 */
function canonicalRigBoneName(name: string) {
  return name
    .trim()
    .replace(/^Armature[\s_:\-|]*/i, '')
    .replace(/^mixamorig[\s_:\-|]*/i, '')
    .replace(/[^a-z0-9]/gi, '')
    .toLowerCase()
}

function findBone(root: THREE.Object3D, targetName: string) {
  const wanted = canonicalRigBoneName(targetName)
  let found: THREE.Bone | undefined

  root.traverse((object) => {
    if (found || !(object instanceof THREE.Bone)) return
    if (canonicalRigBoneName(object.name) === wanted) found = object
  })

  return found
}

function listRigBoneNames(root: THREE.Object3D) {
  const names: string[] = []
  root.traverse((object) => {
    if (object instanceof THREE.Bone) names.push(object.name)
  })
  return names
}

function makeRig(root: THREE.Object3D): Rig {
  return Object.fromEntries(TARGET_BONES.map((name) => [name, findBone(root, name)])) as Rig
}

function captureRestPose(root: THREE.Object3D, rig: Rig) {
  root.updateMatrixWorld(true)
  const rest = new Map<THREE.Bone, RestPose>()
  for (const bone of Object.values(rig)) {
    if (!bone) continue
    const parentWorldQuaternion = new THREE.Quaternion()
    if (bone.parent) bone.parent.getWorldQuaternion(parentWorldQuaternion)
    rest.set(bone, {
      quaternion: bone.quaternion.clone(),
      position: bone.position.clone(),
      worldQuaternion: bone.getWorldQuaternion(new THREE.Quaternion()),
      parentWorldQuaternion,
      worldPosition: bone.getWorldPosition(new THREE.Vector3()),
    })
  }
  return rest
}

function resetRig(rig: Rig, rest: Map<THREE.Bone, RestPose>) {
  for (const bone of Object.values(rig)) {
    if (!bone) continue
    const base = rest.get(bone)
    if (!base) continue
    bone.position.copy(base.position)
    bone.quaternion.copy(base.quaternion)
  }
}

function makeMaterial(name: string, source: THREE.Material | undefined, marbleTexture?: THREE.Texture) {
  const isJoint = /joint/i.test(name)
  const isSurface = /surface/i.test(name)
  const isDarkDetail = /eye|visor|inner|socket|under|rubber/i.test(name)
  const standard = source instanceof THREE.MeshStandardMaterial ? source : undefined

  // Monochrome beige palette: cream shell, taupe mechanics and warm
  // near-black details, all on one hue. Highlights come from the warm
  // studio lighting rather than changing A/B appearance independently.
  const material = new THREE.MeshPhysicalMaterial({
    color: isJoint
      ? new THREE.Color(0x9c8a72)
      : isDarkDetail
        ? new THREE.Color(0x16120e)
        : isSurface
          ? new THREE.Color(0xf1e8d9)
          : standard?.color ?? new THREE.Color(0xe6dccb),
    map: isSurface ? marbleTexture ?? standard?.map ?? null : standard?.map ?? null,
    normalMap: standard?.normalMap ?? null,
    roughness: isJoint ? 0.28 : isDarkDetail ? 0.48 : isSurface ? 0.44 : 0.38,
    metalness: isJoint ? 0.82 : isDarkDetail ? 0.58 : 0.06,
    clearcoat: isJoint ? 0.26 : isDarkDetail ? 0.10 : isSurface ? 0.58 : 0.72,
    clearcoatRoughness: isJoint ? 0.24 : isSurface ? 0.46 : 0.34,
    sheen: isSurface ? 0.16 : 0,
    sheenColor: new THREE.Color(0xe6d9c4),
    sheenRoughness: 0.72,
    emissive: isDarkDetail ? new THREE.Color(0x070605) : new THREE.Color(0x000000),
    emissiveIntensity: isDarkDetail ? 0.08 : 0,
    side: THREE.DoubleSide,
  })
  if (isSurface && marbleTexture) {
    marbleTexture.colorSpace = THREE.SRGBColorSpace
    material.map = marbleTexture
  }
  material.name = `${name || source?.name || 'XBot'}_renaissance_display`
  return material
}


const G1_LINK_TARGET: Record<string, TargetBoneName> = {
  pelvis: 'Hips',
  waist_yaw_link: 'Spine',
  waist_roll_link: 'Spine1',
  torso_link: 'Spine2',
  logo_link: 'Spine2',
  head_link: 'Spine2',
  left_hip_pitch_link: 'LeftUpLeg',
  left_hip_roll_link: 'LeftUpLeg',
  left_hip_yaw_link: 'LeftUpLeg',
  left_knee_link: 'LeftLeg',
  left_ankle_pitch_link: 'LeftFoot',
  left_ankle_roll_link: 'LeftFoot',
  right_hip_pitch_link: 'RightUpLeg',
  right_hip_roll_link: 'RightUpLeg',
  right_hip_yaw_link: 'RightUpLeg',
  right_knee_link: 'RightLeg',
  right_ankle_pitch_link: 'RightFoot',
  right_ankle_roll_link: 'RightFoot',
  // The three physical shoulder axes share one semantic upper-arm driver.
  // Keeping them together preserves the rigid G1 shoulder assembly instead of
  // pulling the pitch cover away from the roll/yaw links during human motion.
  left_shoulder_pitch_link: 'LeftArm',
  left_shoulder_roll_link: 'LeftArm',
  left_shoulder_yaw_link: 'LeftArm',
  left_elbow_link: 'LeftForeArm',
  left_wrist_roll_link: 'LeftHand',
  left_wrist_pitch_link: 'LeftHand',
  left_wrist_yaw_link: 'LeftHand',
  left_rubber_hand: 'LeftHand',
  right_shoulder_pitch_link: 'RightArm',
  right_shoulder_roll_link: 'RightArm',
  right_shoulder_yaw_link: 'RightArm',
  right_elbow_link: 'RightForeArm',
  right_wrist_roll_link: 'RightHand',
  right_wrist_pitch_link: 'RightHand',
  right_wrist_yaw_link: 'RightHand',
  right_rubber_hand: 'RightHand',
}

const g1AxisConversion = new THREE.Matrix4().set(
  0, 1, 0, 0,
  0, 0, 1, 0,
  1, 0, 0, 0,
  0, 0, 0, 1,
)

function parseUrdfVector(raw: string | null | undefined, fallback = new THREE.Vector3()) {
  if (!raw) return fallback.clone()
  const values = raw.trim().split(/\s+/).map(Number)
  return new THREE.Vector3(values[0] ?? fallback.x, values[1] ?? fallback.y, values[2] ?? fallback.z)
}

function urdfOriginMatrix(origin: Element | null) {
  const xyz = parseUrdfVector(origin?.getAttribute('xyz'))
  const rpy = parseUrdfVector(origin?.getAttribute('rpy'))
  const quaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(rpy.x, rpy.y, rpy.z, 'XYZ'))
  return new THREE.Matrix4().compose(xyz, quaternion, new THREE.Vector3(1, 1, 1))
}

function makeG1OfficialMaterial(name: string, marbleTexture?: THREE.Texture) {
  const isGold = /dark/i.test(name)
  if (!isGold && marbleTexture) {
    marbleTexture.colorSpace = THREE.SRGBColorSpace
    marbleTexture.wrapS = THREE.RepeatWrapping
    marbleTexture.wrapT = THREE.RepeatWrapping
    marbleTexture.repeat.set(0.62, 0.62)
    marbleTexture.center.set(0.5, 0.5)
    marbleTexture.rotation = -0.08
  }
  return new THREE.MeshToonMaterial({
    color: new THREE.Color(isGold ? 0xa8957a : 0xf2e9da),
    map: !isGold ? marbleTexture ?? null : null,
    emissive: new THREE.Color(isGold ? 0x1f1912 : 0x14100c),
    emissiveIntensity: isGold ? 0.06 : 0.025,
    side: THREE.DoubleSide,
  })
}


/**
 * Build Aura's semantic retarget skeleton directly from the zero-pose G1 URDF
 * joint frames. v4.3 reused the earlier T-pose proxy as a hidden driver, so the
 * official arms-down G1 geometry rotated around incompatible shoulder/spine
 * pivots and visibly separated. These bones now sit on the robot's real hip,
 * knee, ankle, waist, shoulder, elbow and wrist frames.
 *
 * Multi-axis robot joints are intentionally collapsed into the nearest semantic
 * human segment; the official rigid links stay grouped on that segment. This is
 * still a kinematic visualization, but its rest morphology and rotation pivots
 * are now G1-native.
 */
function buildG1SemanticDriver(
  linkWorld: Map<string, THREE.Matrix4>,
  rootAlignment: THREE.Matrix4,
) {
  const group = new THREE.Group()
  group.name = 'Unitree-G1-semantic-URDF-driver'

  const worldPoint = (linkName: string) => {
    const matrix = linkWorld.get(linkName)
    if (!matrix) throw new Error(`G1 URDF is missing link frame: ${linkName}`)
    return new THREE.Vector3().setFromMatrixPosition(
      rootAlignment.clone().multiply(g1AxisConversion).multiply(matrix),
    )
  }

  const addBoneAt = (
    name: string,
    parent: THREE.Object3D,
    worldPosition: THREE.Vector3,
    parentWorldPosition: THREE.Vector3,
  ) => {
    const bone = new THREE.Bone()
    bone.name = name
    bone.position.copy(worldPosition).sub(parentWorldPosition)
    parent.add(bone)
    return bone
  }

  const origin = new THREE.Vector3()
  const hipsPoint = worldPoint('pelvis')
  const hips = addBoneAt('Hips', group, hipsPoint, origin)

  const spinePoint = worldPoint('waist_yaw_link')
  const spine = addBoneAt('Spine', hips, spinePoint, hipsPoint)
  const spine1Point = worldPoint('waist_roll_link')
  const spine1 = addBoneAt('Spine1', spine, spine1Point, spinePoint)
  const spine2Point = worldPoint('torso_link')
  const spine2 = addBoneAt('Spine2', spine1, spine2Point, spine1Point)

  // G1's head link is fixed to the torso and its STL is authored with the head
  // volume above the torso frame. Synthetic neck/head bones are used only for
  // the anatomical calibration basis; no official geometry is attached to them.
  const neckPoint = hipsPoint.clone().add(new THREE.Vector3(0, 0.34, 0))
  const headPoint = hipsPoint.clone().add(new THREE.Vector3(0, 0.45, 0))
  const neck = addBoneAt('Neck', spine2, neckPoint, spine2Point)
  addBoneAt('Head', neck, headPoint, neckPoint)

  const makeArm = (side: 'Left' | 'Right') => {
    const shoulderFrame = worldPoint(`${side.toLowerCase()}_shoulder_pitch_link`)
    const elbowFrame = worldPoint(`${side.toLowerCase()}_elbow_link`)
    const wristFrame = worldPoint(`${side.toLowerCase()}_wrist_roll_link`)
    const wristYawFrame = worldPoint(`${side.toLowerCase()}_wrist_yaw_link`)

    // Shoulder and upper-arm share the exact physical shoulder pivot. The zero
    // offset prevents a human clavicle rotation from translating the robot's
    // bolted shoulder joint away from the torso.
    const shoulder = addBoneAt(`${side}Shoulder`, spine2, shoulderFrame, spine2Point)
    const arm = addBoneAt(`${side}Arm`, shoulder, shoulderFrame, shoulderFrame)
    const foreArm = addBoneAt(`${side}ForeArm`, arm, elbowFrame, shoulderFrame)
    const hand = addBoneAt(`${side}Hand`, foreArm, wristFrame, elbowFrame)
    const handEnd = wristYawFrame.clone().add(new THREE.Vector3(0, -0.015, 0.085))
    addBoneAt(`${side}HandMiddle4`, hand, handEnd, wristFrame)
  }

  const makeLeg = (side: 'Left' | 'Right') => {
    const prefix = side.toLowerCase()
    const hipFrame = worldPoint(`${prefix}_hip_pitch_link`)
    const kneeFrame = worldPoint(`${prefix}_knee_link`)
    const ankleFrame = worldPoint(`${prefix}_ankle_pitch_link`)
    const ankleRollFrame = worldPoint(`${prefix}_ankle_roll_link`)

    const thigh = addBoneAt(`${side}UpLeg`, hips, hipFrame, hipsPoint)
    const shin = addBoneAt(`${side}Leg`, thigh, kneeFrame, hipFrame)
    const foot = addBoneAt(`${side}Foot`, shin, ankleFrame, kneeFrame)
    const toePoint = ankleRollFrame.clone().add(new THREE.Vector3(0, -0.035, 0.17))
    const toe = addBoneAt(`${side}ToeBase`, foot, toePoint, ankleFrame)
    addBoneAt(`${side}Toe_End`, toe, toePoint.clone().add(new THREE.Vector3(0, 0, 0.08)), toePoint)
  }

  makeArm('Left')
  makeArm('Right')
  makeLeg('Left')
  makeLeg('Right')
  group.updateMatrixWorld(true)
  return group
}

/**
 * Load the vendor G1 URDF/STL geometry supplied under public/models/g1 and bind
 * each rigid link to a semantic skeleton whose bind pivots come directly from
 * the G1 URDF. This keeps the stable pelvis-space AIST++ leg retarget while
 * rotating the real G1 exterior geometry around G1-native joint locations.
 *
 * The result is still a kinematic visualization, not Unitree's controller or a
 * dynamics simulation. The official 29-DoF link geometry is used; unsupported
 * sub-joint rotations are folded into the closest semantic human segment.
 */
async function buildG1OfficialModel(assetBase: string, marbleTexture?: THREE.Texture) {
  const urdfUrl = `${assetBase}g1_29dof_mode_15.urdf`
  const response = await fetch(urdfUrl)
  if (!response.ok) throw new Error(`G1 URDF request failed (${response.status})`)
  const xml = new DOMParser().parseFromString(await response.text(), 'application/xml')
  if (xml.querySelector('parsererror')) throw new Error('Could not parse the bundled G1 URDF.')

  const linkWorld = new Map<string, THREE.Matrix4>()
  const childJoints = new Map<string, Array<{ child: string; origin: THREE.Matrix4 }>>()
  const childLinks = new Set<string>()

  xml.querySelectorAll('joint').forEach((joint) => {
    const parent = joint.querySelector('parent')?.getAttribute('link')
    const child = joint.querySelector('child')?.getAttribute('link')
    if (!parent || !child) return
    const origin = urdfOriginMatrix(joint.querySelector('origin'))
    const list = childJoints.get(parent) ?? []
    list.push({ child, origin })
    childJoints.set(parent, list)
    childLinks.add(child)
  })

  const allLinks = Array.from(xml.querySelectorAll('link')).map((link) => link.getAttribute('name')).filter((name): name is string => Boolean(name))
  const rootLink = allLinks.find((name) => !childLinks.has(name)) ?? 'pelvis'
  linkWorld.set(rootLink, new THREE.Matrix4())
  const visit = (parent: string) => {
    const parentMatrix = linkWorld.get(parent)
    if (!parentMatrix) return
    for (const joint of childJoints.get(parent) ?? []) {
      linkWorld.set(joint.child, parentMatrix.clone().multiply(joint.origin))
      visit(joint.child)
    }
  }
  visit(rootLink)

  // Place the pelvis at a convenient scene height, then derive every semantic
  // driver pivot from the actual URDF zero-pose frames. The model is normalized
  // to the stage later in finishModel(), so this is only a stable bind offset.
  const g1RootAlignment = new THREE.Matrix4().makeTranslation(0, 0.76, 0)
  const group = buildG1SemanticDriver(linkWorld, g1RootAlignment)
  group.name = 'Unitree-G1-official-URDF-geometry'
  const rig = makeRig(group)

  const stlLoader = new STLLoader()
  const darkMaterial = makeG1OfficialMaterial('dark')
  const whiteMaterial = makeG1OfficialMaterial('white', marbleTexture)
  const loadedMeshes: THREE.Mesh[] = []

  const visualJobs = Array.from(xml.querySelectorAll('link')).flatMap((link) => {
    const linkName = link.getAttribute('name') ?? ''
    const targetName = G1_LINK_TARGET[linkName]
    const targetBone = targetName ? rig[targetName] : undefined
    const world = linkWorld.get(linkName)
    if (!targetBone || !world) return []

    const visuals = Array.from(link.children).filter((child) => child.tagName.toLowerCase() === 'visual')
    return visuals.map(async (visual) => {
      const meshNode = visual.querySelector('geometry > mesh')
      const filename = meshNode?.getAttribute('filename')
      if (!filename) return
      const materialName = visual.querySelector('material')?.getAttribute('name') ?? 'white'
      const geometry = await stlLoader.loadAsync(`${assetBase}${filename}`)
      geometry.computeVertexNormals()

      const visualWorldUrdf = world.clone().multiply(urdfOriginMatrix(visual.querySelector('origin')))
      const visualWorldThree = g1RootAlignment.clone().multiply(g1AxisConversion).multiply(visualWorldUrdf)
      targetBone.updateWorldMatrix(true, false)
      const localMatrix = targetBone.matrixWorld.clone().invert().multiply(visualWorldThree)

      const mesh = new THREE.Mesh(geometry, /dark/i.test(materialName) ? darkMaterial : whiteMaterial)
      mesh.name = `g1_${linkName}_surface`
      const outline = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0x17130e, side: THREE.BackSide, transparent: true, opacity: 0.70 }))
      outline.name = `${mesh.name}_comic_outline`
      outline.scale.setScalar(1.012)
      outline.renderOrder = -1
      mesh.add(outline)
      localMatrix.decompose(mesh.position, mesh.quaternion, mesh.scale)
      mesh.castShadow = true
      mesh.receiveShadow = true
      mesh.frustumCulled = false
      targetBone.add(mesh)
      loadedMeshes.push(mesh)
    })
  })

  await Promise.all(visualJobs)
  if (loadedMeshes.length < 20) throw new Error(`Only ${loadedMeshes.length} G1 visual links loaded.`)
  group.updateMatrixWorld(true)
  return group
}

const MAX_TRAIL_POINTS = 14
// Luminous end-effector traces: a bright colored core inside a much softer,
// additive outer tube. This creates a bloom-like glow without requiring a
// heavyweight post-processing pass and stays reliable across mobile GPUs.
const TRAIL_RADIUS = 0.0065
const TRAIL_BORDER_RADIUS = 0.021
const trailAxis = new THREE.Vector3(0, 1, 0)
const trailDirection = new THREE.Vector3()
const trailMidpoint = new THREE.Vector3()
const trailScale = new THREE.Vector3()
const trailQuaternion = new THREE.Quaternion()
const trailMatrix = new THREE.Matrix4()
const trailBorderColor = new THREE.Color()
const trailCoreColor = new THREE.Color()
const trailWhite = new THREE.Color(0xffffff)

function createTrail(endEffector: THREE.Object3D, color: number) {
  // Use instanced 3D cylinders instead of Line2. This is deliberately a little
  // more geometry, but it is much more reliable across browsers/GPUs and makes
  // the trajectory visibly continuous instead of collapsing to an endpoint dot.
  const segmentGeometry = new THREE.CylinderGeometry(1, 1, 1, 7, 1, true)
  const trailColor = new THREE.Color(color)
  const segmentMaterial = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.92,
    depthWrite: false,
    depthTest: false,
    toneMapped: false,
    blending: THREE.AdditiveBlending,
  })
  const glowMaterial = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.22,
    depthWrite: false,
    depthTest: false,
    toneMapped: false,
    blending: THREE.AdditiveBlending,
  })

  const segments = new THREE.InstancedMesh(segmentGeometry, segmentMaterial, MAX_TRAIL_POINTS)
  segments.count = 0
  segments.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  segments.frustumCulled = false
  segments.renderOrder = 5

  const glow = new THREE.InstancedMesh(segmentGeometry, glowMaterial, MAX_TRAIL_POINTS)
  glow.count = 0
  glow.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  glow.frustumCulled = false
  glow.renderOrder = 4

  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.032, 16, 16),
    new THREE.MeshStandardMaterial({
      color: trailColor,
      emissive: trailColor,
      emissiveIntensity: 1.8,
      roughness: 0.22,
      metalness: 0.0,
      depthWrite: false,
      transparent: true,
      opacity: 0.82,
      depthTest: false,
    }),
  )
  marker.renderOrder = 6

  return { endEffector, color: trailColor, segments, glow, marker, points: [], lastSample: -1 } satisfies Trail
}

function setTrailSegments(trail: Trail, current?: THREE.Vector3) {
  const points = current ? [...trail.points, current] : trail.points
  const count = Math.min(Math.max(0, points.length - 1), MAX_TRAIL_POINTS - 1)
  const first = Math.max(0, points.length - 1 - count)

  let written = 0
  for (let i = first; i < points.length - 1 && written < count; i += 1) {
    const a = points[i]
    const b = points[i + 1]
    trailDirection.copy(b).sub(a)
    const length = trailDirection.length()
    if (length < 1e-5) continue

    trailDirection.multiplyScalar(1 / length)
    trailMidpoint.copy(a).add(b).multiplyScalar(0.5)
    trailQuaternion.setFromUnitVectors(trailAxis, trailDirection)

    // Older samples taper and dim, while the newest samples flare toward white.
    // Instance color multiplies the additive material, producing a bright core
    // with a softer colored halo and a natural luminous fade behind the limb.
    const age01 = count <= 1 ? 1 : written / Math.max(1, count - 1)
    const taper = THREE.MathUtils.lerp(0.18, 1, age01 * age01 * age01)
    const coreStrength = THREE.MathUtils.lerp(0.10, 1.0, age01 * age01)
    const glowStrength = THREE.MathUtils.lerp(0.025, 0.60, age01 * age01)

    trailScale.set(TRAIL_RADIUS * taper, length, TRAIL_RADIUS * taper)
    trailMatrix.compose(trailMidpoint, trailQuaternion, trailScale)
    trail.segments.setMatrixAt(written, trailMatrix)
    trailCoreColor.copy(trail.color).lerp(trailWhite, 0.48 * age01).multiplyScalar(coreStrength)
    trail.segments.setColorAt(written, trailCoreColor)

    trailScale.set(TRAIL_BORDER_RADIUS * taper, length, TRAIL_BORDER_RADIUS * taper)
    trailMatrix.compose(trailMidpoint, trailQuaternion, trailScale)
    trail.glow.setMatrixAt(written, trailMatrix)
    trailBorderColor.copy(trail.color).lerp(trailWhite, 0.14 * age01).multiplyScalar(glowStrength)
    trail.glow.setColorAt(written, trailBorderColor)
    written += 1
  }

  trail.segments.count = written
  trail.glow.count = written
  trail.segments.instanceMatrix.needsUpdate = true
  trail.glow.instanceMatrix.needsUpdate = true
  if (trail.segments.instanceColor) trail.segments.instanceColor.needsUpdate = true
  if (trail.glow.instanceColor) trail.glow.instanceColor.needsUpdate = true
}

function hash01(text: string, seed: number) {
  let h = (2166136261 ^ seed) >>> 0
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return (h >>> 0) / 4294967295
}

// Deliberately visible study degradation: the lower-quality side should read as
// hesitant and less coordinated at a glance, while remaining the same choreography.
function warpedLocalTime(localSeconds: number, seed: number) {
  const u = THREE.MathUtils.clamp(localSeconds / STUDY_CLIP_SECONDS, 0, 1)
  const envelope = Math.sin(Math.PI * u) ** 2
  const p1 = hash01('time-a', seed) * Math.PI * 2
  const p2 = hash01('time-b', seed) * Math.PI * 2
  const shift = envelope * (
    0.28 * Math.sin(Math.PI * 2 * u + p1) +
    0.12 * Math.sin(Math.PI * 4 * u + p2)
  )
  return THREE.MathUtils.clamp(localSeconds + shift, 0, STUDY_CLIP_SECONDS)
}

function spatialScale(targetName: string, localSeconds: number, seed: number) {
  const phase = hash01(`amp-${targetName}`, seed) * Math.PI * 2
  const wobble = 0.5 + 0.5 * Math.sin(localSeconds * 0.86 + phase)
  const h = hash01(`bias-${targetName}`, seed)

  if (/Hand|ForeArm|Arm|Foot|Leg|UpLeg/.test(targetName)) {
    if (h > 0.90) return 1.08 + 0.10 * wobble
    return 0.48 + 0.22 * wobble
  }
  if (/Shoulder|Spine/.test(targetName)) return 0.65 + 0.16 * wobble
  if (/Neck|Head/.test(targetName)) return 0.72 + 0.12 * wobble
  return 0.66 + 0.16 * wobble
}

function degradationNudge(targetName: string, localSeconds: number, seed: number) {
  const limb = /(Arm|ForeArm|Hand|UpLeg|Leg|Foot)/.test(targetName)
  const max = limb ? 0.13 : 0.055
  const px = hash01(`nx-${targetName}`, seed) * Math.PI * 2
  const py = hash01(`ny-${targetName}`, seed) * Math.PI * 2
  const pz = hash01(`nz-${targetName}`, seed) * Math.PI * 2
  tempEuler.set(
    max * Math.sin(localSeconds * 1.07 + px),
    max * 0.72 * Math.sin(localSeconds * 0.83 + py),
    max * 0.58 * Math.sin(localSeconds * 1.31 + pz),
    'XYZ',
  )
  return qNudge.setFromEuler(tempEuler)
}

function getSourceRestPosition(motion: BvhMotion, name: string) {
  const index = motion.nameToIndex.get(name)
  if (index === undefined) throw new Error(`BVH calibration is missing ${name}.`)
  return motion.restWorldPositions[index]
}

function getTargetRestPosition(rig: Rig, rest: Map<THREE.Bone, RestPose>, name: TargetBoneName) {
  const bone = rig[name]
  const pose = bone ? rest.get(bone) : undefined
  if (!bone || !pose) throw new Error(`XBot calibration is missing ${name}.`)
  return pose.worldPosition
}

function projectPole(
  hip: THREE.Vector3,
  knee: THREE.Vector3,
  ankle: THREE.Vector3,
  out: THREE.Vector3,
) {
  out.copy(ankle).sub(hip)
  const lengthSq = out.lengthSq()
  if (lengthSq < 1e-10) return out.set(0, 0, 1)
  out.multiplyScalar(1 / Math.sqrt(lengthSq))
  tempVector2.copy(knee).sub(hip)
  tempVector2.addScaledVector(out, -tempVector2.dot(out))
  if (tempVector2.lengthSq() < 1e-10) {
    // A deterministic perpendicular fallback. It is only used if the rest pose
    // is perfectly straight and therefore has no meaningful knee pole.
    tempVector2.set(0, 0, 1).addScaledVector(out, -out.z)
    if (tempVector2.lengthSq() < 1e-10) tempVector2.set(1, 0, 0).addScaledVector(out, -out.x)
  }
  return out.copy(tempVector2).normalize()
}

function makeAnatomicalBasis(
  leftHip: THREE.Vector3,
  rightHip: THREE.Vector3,
  leftShoulder: THREE.Vector3,
  rightShoulder: THREE.Vector3,
  hips: THREE.Vector3,
  head: THREE.Vector3,
) {
  const up = head.clone().sub(hips).normalize()
  const hipRight = rightHip.clone().sub(leftHip).normalize()
  const shoulderRight = rightShoulder.clone().sub(leftShoulder).normalize()
  const right = hipRight.add(shoulderRight).normalize()
  right.addScaledVector(up, -right.dot(up)).normalize()
  const forward = right.clone().cross(up).normalize()
  right.copy(up).cross(forward).normalize()
  const matrix = new THREE.Matrix4().makeBasis(right, up, forward)
  return new THREE.Quaternion().setFromRotationMatrix(matrix).normalize()
}

function buildLegCalibration(
  side: LegSide,
  motion: BvhMotion,
  rig: Rig,
  rest: Map<THREE.Bone, RestPose>,
  sourceToTargetWorld: THREE.Quaternion,
) {
  const sourceHip = `${side}UpLeg`
  const sourceKnee = `${side}Leg`
  const sourceAnkle = `${side}Foot`
  const sourceToe = `${side}ToeBase`
  const targetThigh = `${side}UpLeg` as TargetBoneName
  const targetShin = `${side}Leg` as TargetBoneName
  const targetFoot = `${side}Foot` as TargetBoneName

  const srcHip = getSourceRestPosition(motion, sourceHip)
  const srcKnee = getSourceRestPosition(motion, sourceKnee)
  const srcAnkle = getSourceRestPosition(motion, sourceAnkle)
  const tgtHip = getTargetRestPosition(rig, rest, targetThigh)
  const tgtKnee = getTargetRestPosition(rig, rest, targetShin)
  const tgtAnkle = getTargetRestPosition(rig, rest, targetFoot)

  const sourceAxis = srcAnkle.clone().sub(srcHip)
  const targetAxis = tgtAnkle.clone().sub(tgtHip)
  const mappedAxis = sourceAxis.clone().normalize().applyQuaternion(sourceToTargetWorld).normalize()
  const targetAxisNormal = targetAxis.clone().normalize()
  const directionCorrection = new THREE.Quaternion().setFromUnitVectors(mappedAxis, targetAxisNormal)

  const sourcePole = new THREE.Vector3()
  const targetPole = new THREE.Vector3()
  projectPole(srcHip, srcKnee, srcAnkle, sourcePole)
  projectPole(tgtHip, tgtKnee, tgtAnkle, targetPole)

  // Only align the source leg's PRIMARY hip→ankle direction. Do not derive
  // an axial twist from XBot's rest knee pole: its bind leg is almost straight,
  // so that pole is numerically unstable. In earlier builds that unstable twist
  // rotated AIST++'s forward axis toward ±X and made the kneecaps face inward.
  targetPole.addScaledVector(targetAxisNormal, -targetPole.dot(targetAxisNormal)).normalize()

  const targetThighBone = rig[targetThigh]
  const targetShinBone = rig[targetShin]
  const targetThighBase = targetThighBone ? rest.get(targetThighBone) : undefined
  const targetShinBase = targetShinBone ? rest.get(targetShinBone) : undefined
  if (!targetThighBase || !targetShinBase) {
    throw new Error(`XBot calibration is missing ${side} leg bind-pose data.`)
  }
  const targetRestPlaneNormal = targetAxisNormal.clone().cross(targetPole).normalize()
  const correction = directionCorrection.clone().normalize()

  const targetThighLength = tgtKnee.distanceTo(tgtHip)
  const targetShinLength = tgtAnkle.distanceTo(tgtKnee)
  const sourceMaxLength = srcKnee.distanceTo(srcHip) + srcAnkle.distanceTo(srcKnee)
  const targetMaxLength = targetThighLength + targetShinLength

  return {
    sourceHip,
    sourceKnee,
    sourceAnkle,
    sourceToe,
    targetThigh,
    targetShin,
    targetFoot,
    correction,
    targetRestAxis: targetAxis,
    targetRestPole: targetPole,
    targetRestPlaneNormal,
    targetThighLength,
    targetShinLength,
    targetRestReachRatio: THREE.MathUtils.clamp(targetAxis.length() / Math.max(targetMaxLength, 1e-6), 0, 1),
    sourceMaxLength,
  } satisfies LegCalibration
}

function buildRetargetCalibration(
  motion: BvhMotion,
  rig: Rig,
  rest: Map<THREE.Bone, RestPose>,
  targetSegmentRestDirections: Map<TargetBoneName, THREE.Vector3>,
  embodiment: Embodiment,
) {
  const sourceBasis = makeAnatomicalBasis(
    getSourceRestPosition(motion, 'LeftUpLeg'),
    getSourceRestPosition(motion, 'RightUpLeg'),
    getSourceRestPosition(motion, 'LeftArm'),
    getSourceRestPosition(motion, 'RightArm'),
    getSourceRestPosition(motion, 'Hips'),
    getSourceRestPosition(motion, 'Head'),
  )
  const targetBasis = makeAnatomicalBasis(
    getTargetRestPosition(rig, rest, 'LeftUpLeg'),
    getTargetRestPosition(rig, rest, 'RightUpLeg'),
    getTargetRestPosition(rig, rest, 'LeftArm'),
    getTargetRestPosition(rig, rest, 'RightArm'),
    getTargetRestPosition(rig, rest, 'Hips'),
    getTargetRestPosition(rig, rest, 'Head'),
  )
  const sourceToTargetWorld = targetBasis.clone().multiply(sourceBasis.clone().invert()).normalize()
  const sourceToTargetWorldInverse = sourceToTargetWorld.clone().invert()

  // One global anatomical basis handles facing direction and, crucially, the
  // semantic left/right axis. Per-segment corrections then only compensate for
  // different rest-pose proportions. This avoids accidentally mirroring one leg
  // independently of the other.
  const segmentCorrections = new Map<TargetBoneName, THREE.Quaternion>()
  for (const targetName of TARGET_BONES) {
    const segment = SWING_SEGMENTS[targetName]
    const targetRestDirection = targetSegmentRestDirections.get(targetName)
    if (!segment || !targetRestDirection) continue
    const sourceIndex = motion.nameToIndex.get(SOURCE_FOR_TARGET[targetName])
    const childIndex = motion.nameToIndex.get(segment.sourceChild)
    if (sourceIndex === undefined || childIndex === undefined) continue
    const mappedSourceRestDirection = motion.restWorldPositions[childIndex]
      .clone()
      .sub(motion.restWorldPositions[sourceIndex])
      .normalize()
      .applyQuaternion(sourceToTargetWorld)
      .normalize()
    segmentCorrections.set(
      targetName,
      new THREE.Quaternion().setFromUnitVectors(mappedSourceRestDirection, targetRestDirection.clone().normalize()),
    )
  }

  return {
    sourceToTargetWorld,
    sourceToTargetWorldInverse,
    segmentCorrections,
    embodiment,
    legs: {
      Left: buildLegCalibration('Left', motion, rig, rest, sourceToTargetWorld),
      Right: buildLegCalibration('Right', motion, rig, rest, sourceToTargetWorld),
    },
  } satisfies RetargetCalibration
}

function aimBoneAlongWorldDirection(
  bone: THREE.Bone,
  base: RestPose,
  parentWorld: THREE.Quaternion,
  localChildDirection: THREE.Vector3,
  desiredWorldDirection: THREE.Vector3,
  outWorld: THREE.Quaternion,
) {
  qBaselineWorld.copy(parentWorld).multiply(base.quaternion).normalize()
  targetBaselineDirection.copy(localChildDirection).applyQuaternion(qBaselineWorld).normalize()
  qSwing.setFromUnitVectors(targetBaselineDirection, desiredWorldDirection.clone().normalize())
  outWorld.copy(qSwing).multiply(qBaselineWorld).normalize()
  qParentInv.copy(parentWorld).invert()
  bone.quaternion.copy(qParentInv).multiply(outWorld).normalize()
}

const sourcePelvisLeftAxis = new THREE.Vector3()
const sourcePelvisCenter = new THREE.Vector3()
const targetPelvisLeftAxis = new THREE.Vector3()
const targetPelvisCenter = new THREE.Vector3()
const directThighDirection = new THREE.Vector3()
const directShinDirection = new THREE.Vector3()
const directFootDirection = new THREE.Vector3()
const segmentBaselineDirection = new THREE.Vector3()

/**
 * Change only the lateral component of a direction while preserving as much of
 * its vertical/depth direction as possible. `requiredLateral` is a signed unit
 * direction component along `lateralAxis`.
 */
function enforceLateralComponent(
  direction: THREE.Vector3,
  lateralAxis: THREE.Vector3,
  requiredLateral: number,
) {
  const lateral = THREE.MathUtils.clamp(requiredLateral, -0.94, 0.94)
  direction.normalize()
  const currentLateral = direction.dot(lateralAxis)
  tempVector.copy(direction).addScaledVector(lateralAxis, -currentLateral)
  if (tempVector.lengthSq() < 1e-8) {
    tempVector.set(0, -1, 0).addScaledVector(lateralAxis, lateralAxis.y)
    if (tempVector.lengthSq() < 1e-8) tempVector.set(0, 0, 1).addScaledVector(lateralAxis, -lateralAxis.z)
  }
  tempVector.normalize().multiplyScalar(Math.sqrt(Math.max(0, 1 - lateral * lateral)))
  direction.copy(tempVector).addScaledVector(lateralAxis, lateral).normalize()
}

function softenSegmentDirection(
  bone: THREE.Bone,
  base: RestPose,
  parentWorld: THREE.Quaternion,
  localChildDirection: THREE.Vector3,
  desiredWorldDirection: THREE.Vector3,
  amount: number,
) {
  qBaselineWorld.copy(parentWorld).multiply(base.quaternion).normalize()
  segmentBaselineDirection.copy(localChildDirection).applyQuaternion(qBaselineWorld).normalize()
  qSwing.setFromUnitVectors(segmentBaselineDirection, desiredWorldDirection)
  qAdjusted.identity().slerp(qSwing, amount)
  desiredWorldDirection.copy(segmentBaselineDirection).applyQuaternion(qAdjusted).normalize()
}

/**
 * Retarget a leg from the *actual source segment directions* and preserve the
 * source knee/ankle's lateral position relative to the pelvis.
 *
 * Previous builds applied a fixed correction that mapped AIST++'s splayed BVH
 * rest legs onto XBot's straight bind legs. During real motion the thighs are
 * already much closer to vertical, so that fixed correction overshot neutral:
 * left rotated inward/right and right rotated inward/left, producing the
 * persistent X-legged silhouette.
 *
 * This solver deliberately does NOT use that per-leg rest correction. It maps
 * the live AIST++ thigh/shin directions through the one global anatomical basis,
 * then constrains the resulting target knee/ankle to the same normalized
 * left/right pelvis coordinate as the source. Axial roll stays XBot-native via
 * aimBoneAlongWorldDirection's minimal swing.
 */
function solveLegIK(
  side: LegSide,
  rig: Rig,
  targetRest: Map<THREE.Bone, RestPose>,
  targetAnimatedWorld: Map<THREE.Bone, THREE.Quaternion>,
  motion: BvhMotion,
  sourcePose: BvhPoseBuffers,
  calibration: RetargetCalibration,
  targetSegmentLocalDirections: Map<TargetBoneName, THREE.Vector3>,
  quality: MotionQuality,
  localSeconds: number,
  seed: number,
) {
  const leg = calibration.legs[side]
  const thigh = rig[leg.targetThigh]
  const shin = rig[leg.targetShin]
  const foot = rig[leg.targetFoot]
  const thighBase = thigh ? targetRest.get(thigh) : undefined
  const shinBase = shin ? targetRest.get(shin) : undefined
  const footBase = foot ? targetRest.get(foot) : undefined
  const thighLocalDir = targetSegmentLocalDirections.get(leg.targetThigh)
  const shinLocalDir = targetSegmentLocalDirections.get(leg.targetShin)
  const footLocalDir = targetSegmentLocalDirections.get(leg.targetFoot)
  if (!thigh || !shin || !foot || !thighBase || !shinBase || !footBase || !thighLocalDir || !shinLocalDir || !footLocalDir) return

  const srcHipIndex = motion.nameToIndex.get(leg.sourceHip)
  const srcKneeIndex = motion.nameToIndex.get(leg.sourceKnee)
  const srcAnkleIndex = motion.nameToIndex.get(leg.sourceAnkle)
  const srcToeIndex = motion.nameToIndex.get(leg.sourceToe)
  const srcLeftHipIndex = motion.nameToIndex.get('LeftUpLeg')
  const srcRightHipIndex = motion.nameToIndex.get('RightUpLeg')
  const targetLeftHip = rig.LeftUpLeg
  const targetRightHip = rig.RightUpLeg
  if (
    srcHipIndex === undefined || srcKneeIndex === undefined || srcAnkleIndex === undefined || srcToeIndex === undefined ||
    srcLeftHipIndex === undefined || srcRightHipIndex === undefined || !targetLeftHip || !targetRightHip
  ) return

  const srcHip = sourcePose.worldPositions[srcHipIndex]
  const srcKnee = sourcePose.worldPositions[srcKneeIndex]
  const srcAnkle = sourcePose.worldPositions[srcAnkleIndex]
  const srcToe = sourcePose.worldPositions[srcToeIndex]
  const srcLeftHip = sourcePose.worldPositions[srcLeftHipIndex]
  const srcRightHip = sourcePose.worldPositions[srcRightHipIndex]

  sourcePelvisCenter.copy(srcLeftHip).add(srcRightHip).multiplyScalar(0.5)
  sourcePelvisLeftAxis.copy(srcLeftHip).sub(srcRightHip)
  const sourceHipWidth = Math.max(sourcePelvisLeftAxis.length(), 1e-6)
  sourcePelvisLeftAxis.multiplyScalar(1 / sourceHipWidth)
  const sourceHalfWidth = sourceHipWidth * 0.5
  const sourceKneeRatio = THREE.MathUtils.clamp(
    tempVector.copy(srcKnee).sub(sourcePelvisCenter).dot(sourcePelvisLeftAxis) / sourceHalfWidth,
    -3.0,
    3.0,
  )
  const sourceAnkleRatio = THREE.MathUtils.clamp(
    tempVector.copy(srcAnkle).sub(sourcePelvisCenter).dot(sourcePelvisLeftAxis) / sourceHalfWidth,
    -4.0,
    4.0,
  )

  targetLeftHip.updateWorldMatrix(true, false)
  targetRightHip.updateWorldMatrix(true, false)
  const targetLeftHipWorld = targetLeftHip.getWorldPosition(new THREE.Vector3())
  const targetRightHipWorld = targetRightHip.getWorldPosition(new THREE.Vector3())
  targetPelvisCenter.copy(targetLeftHipWorld).add(targetRightHipWorld).multiplyScalar(0.5)
  targetPelvisLeftAxis.copy(targetLeftHipWorld).sub(targetRightHipWorld)
  const targetHipWidth = Math.max(targetPelvisLeftAxis.length(), 1e-6)
  targetPelvisLeftAxis.multiplyScalar(1 / targetHipWidth)
  const targetHalfWidth = targetHipWidth * 0.5

  // --- thigh ---------------------------------------------------------------
  directThighDirection.copy(srcKnee).sub(srcHip).normalize()
    .applyQuaternion(calibration.sourceToTargetWorld).normalize()

  const thighParentWorld = thigh.parent
    ? thigh.parent.getWorldQuaternion(new THREE.Quaternion())
    : new THREE.Quaternion()

  if (quality === 'degraded') {
    softenSegmentDirection(
      thigh,
      thighBase,
      thighParentWorld,
      thighLocalDir,
      directThighDirection,
      THREE.MathUtils.clamp(spatialScale(leg.targetThigh, localSeconds, seed), 0.48, 0.86),
    )
  }

  const targetHipWorld = thigh.getWorldPosition(new THREE.Vector3())
  const targetHipLateral = tempVector.copy(targetHipWorld).sub(targetPelvisCenter).dot(targetPelvisLeftAxis)
  const targetKneeLateral = sourceKneeRatio * targetHalfWidth
  const requiredThighLateral = (targetKneeLateral - targetHipLateral) / Math.max(leg.targetThighLength, 1e-6)
  enforceLateralComponent(directThighDirection, targetPelvisLeftAxis, requiredThighLateral)

  const thighWorld = new THREE.Quaternion()
  aimBoneAlongWorldDirection(
    thigh,
    thighBase,
    thighParentWorld,
    thighLocalDir,
    directThighDirection,
    thighWorld,
  )
  const existingThigh = targetAnimatedWorld.get(thigh)
  if (existingThigh) existingThigh.copy(thighWorld)
  else targetAnimatedWorld.set(thigh, thighWorld.clone())
  thigh.updateWorldMatrix(true, true)

  // --- shin ----------------------------------------------------------------
  directShinDirection.copy(srcAnkle).sub(srcKnee).normalize()
    .applyQuaternion(calibration.sourceToTargetWorld).normalize()

  const kneeWorld = shin.getWorldPosition(new THREE.Vector3())
  const shinParentWorld = thigh.getWorldQuaternion(new THREE.Quaternion())

  if (quality === 'degraded') {
    softenSegmentDirection(
      shin,
      shinBase,
      shinParentWorld,
      shinLocalDir,
      directShinDirection,
      THREE.MathUtils.clamp(spatialScale(leg.targetShin, localSeconds, seed), 0.48, 0.86),
    )
  }

  const targetKneeActualLateral = tempVector.copy(kneeWorld).sub(targetPelvisCenter).dot(targetPelvisLeftAxis)
  const targetAnkleLateral = sourceAnkleRatio * targetHalfWidth
  const requiredShinLateral = (targetAnkleLateral - targetKneeActualLateral) / Math.max(leg.targetShinLength, 1e-6)
  enforceLateralComponent(directShinDirection, targetPelvisLeftAxis, requiredShinLateral)

  const shinWorld = new THREE.Quaternion()
  aimBoneAlongWorldDirection(
    shin,
    shinBase,
    shinParentWorld,
    shinLocalDir,
    directShinDirection,
    shinWorld,
  )
  const existingShin = targetAnimatedWorld.get(shin)
  if (existingShin) existingShin.copy(shinWorld)
  else targetAnimatedWorld.set(shin, shinWorld.clone())
  shin.updateWorldMatrix(true, true)

  // --- foot ----------------------------------------------------------------
  directFootDirection.copy(srcToe).sub(srcAnkle).normalize()
    .applyQuaternion(calibration.sourceToTargetWorld).normalize()
  const footParentWorld = shin.getWorldQuaternion(new THREE.Quaternion())
  if (quality === 'degraded') {
    softenSegmentDirection(
      foot,
      footBase,
      footParentWorld,
      footLocalDir,
      directFootDirection,
      THREE.MathUtils.clamp(spatialScale(leg.targetFoot, localSeconds, seed), 0.52, 0.90),
    )
  }
  const footWorld = new THREE.Quaternion()
  aimBoneAlongWorldDirection(
    foot,
    footBase,
    footParentWorld,
    footLocalDir,
    directFootDirection,
    footWorld,
  )
  const existingFoot = targetAnimatedWorld.get(foot)
  if (existingFoot) existingFoot.copy(footWorld)
  else targetAnimatedWorld.set(foot, footWorld.clone())
}

function applyRetargetedPose(
  rig: Rig,
  targetRest: Map<THREE.Bone, RestPose>,
  targetAnimatedWorld: Map<THREE.Bone, THREE.Quaternion>,
  motion: BvhMotion,
  sourcePose: BvhPoseBuffers,
  sourceStartPose: BvhPoseBuffers,
  localSeconds: number,
  startOffsetSeconds: number,
  quality: MotionQuality,
  seed: number,
  rootScale: number,
  calibration: RetargetCalibration,
  targetSegmentLocalDirections: Map<TargetBoneName, THREE.Vector3>,
) {
  resetRig(rig, targetRest)
  targetAnimatedWorld.clear()

  const safeStart = Math.min(startOffsetSeconds, Math.max(0, motion.duration - STUDY_CLIP_SECONDS))
  const sampledLocal = quality === 'degraded' ? warpedLocalTime(localSeconds, seed) : localSeconds
  const sampleTime = THREE.MathUtils.clamp(safeStart + sampledLocal, 0, motion.duration)
  sampleBVHWorldPose(motion, sampleTime, sourcePose)

  // Root translation must use the same anatomical basis as rotations. Otherwise
  // a source forward step can become a target sideways step when rigs face in
  // different coordinate directions.
  const hips = rig.Hips
  const hipsRest = hips ? targetRest.get(hips) : undefined
  const sourceHipsIndex = motion.nameToIndex.get('Hips')
  if (hips && hipsRest && sourceHipsIndex !== undefined) {
    tempVector.copy(sourcePose.worldPositions[sourceHipsIndex]).sub(sourceStartPose.worldPositions[sourceHipsIndex])
    tempVector.applyQuaternion(calibration.sourceToTargetWorld)
    if (quality === 'degraded') {
      tempVector.x *= 0.68
      tempVector.y *= 0.84
      tempVector.z *= 0.68
      const phase = hash01('root-noise', seed) * Math.PI * 2
      const envelope = Math.sin(Math.PI * THREE.MathUtils.clamp(localSeconds / STUDY_CLIP_SECONDS, 0, 1)) ** 2
      tempVector.x += envelope * 0.18 * Math.sin(localSeconds * 0.82 + phase)
      tempVector.z += envelope * 0.14 * Math.sin(localSeconds * 1.14 + phase * 0.7)
    }
    hips.position.copy(hipsRest.position).addScaledVector(tempVector, rootScale)
  }

  for (const targetName of TARGET_BONES) {
    if (IK_LEG_BONES.has(targetName)) continue
    const targetBone = rig[targetName]
    if (!targetBone) continue
    const targetBase = targetRest.get(targetBone)
    if (!targetBase) continue

    const sourceName = SOURCE_FOR_TARGET[targetName]
    const sourceIndex = motion.nameToIndex.get(sourceName)
    if (sourceIndex === undefined) continue

    const parent = targetBone.parent
    const animatedParent = parent instanceof THREE.Bone ? targetAnimatedWorld.get(parent) : undefined
    const parentWorld = animatedParent ?? targetBase.parentWorldQuaternion

    const swing = SWING_SEGMENTS[targetName]
    const targetLocalDir = targetSegmentLocalDirections.get(targetName)
    const sourceChildIndex = swing ? motion.nameToIndex.get(swing.sourceChild) : undefined

    if (targetName === 'Neck' || targetName === 'Head') {
      // Neck/head need their relative motion transferred through the rigs' local
      // coordinate frames. Applying AIST++ local quaternions directly to Mixamo
      // makes pitch axes disagree and leaves the avatar looking down. Start from
      // XBot's native bind pose, compute only the source delta from the clip's
      // first frame, conjugate that delta through the anatomical world mapping,
      // then express it in XBot's parent-rest frame.
      const sourceJoint = motion.joints[sourceIndex]
      const sourceParentIndex = sourceJoint?.parent ?? -1
      if (sourceParentIndex >= 0) {
        qUpperBodyParentStartInv.copy(sourceStartPose.worldQuaternions[sourceParentIndex]).invert()
        qHeadStartLocal
          .copy(qUpperBodyParentStartInv)
          .multiply(sourceStartPose.worldQuaternions[sourceIndex])
          .normalize()

        qHeadParentInv.copy(sourcePose.worldQuaternions[sourceParentIndex]).invert()
        qHeadCurrentLocal
          .copy(qHeadParentInv)
          .multiply(sourcePose.worldQuaternions[sourceIndex])
          .normalize()

        // Delta in the source parent's coordinate system.
        qUpperBodyDeltaLocal
          .copy(qHeadCurrentLocal)
          .multiply(qHeadStartLocal.clone().invert())
          .normalize()

        // Re-express that local delta in source world axes at the calibration
        // frame, map those axes to XBot world, then bring it into XBot's parent
        // rest frame. This is the part v3.5 was missing.
        qUpperBodyDeltaWorld
          .copy(sourceStartPose.worldQuaternions[sourceParentIndex])
          .multiply(qUpperBodyDeltaLocal)
          .multiply(sourceStartPose.worldQuaternions[sourceParentIndex].clone().invert())
          .normalize()

        qUpperBodyMappedWorld
          .copy(calibration.sourceToTargetWorld)
          .multiply(qUpperBodyDeltaWorld)
          .multiply(calibration.sourceToTargetWorldInverse)
          .normalize()

        qUpperBodyTargetParentRestInv.copy(targetBase.parentWorldQuaternion).invert()
        qUpperBodyTargetDeltaLocal
          .copy(qUpperBodyTargetParentRestInv)
          .multiply(qUpperBodyMappedWorld)
          .multiply(targetBase.parentWorldQuaternion)
          .normalize()

        // Human dance BVHs often contain much larger neck/head excursion than
        // reads naturally on XBot's spherical head. Damp only this secondary
        // motion while keeping the native forward-facing bind orientation.
        const upperBodyWeight = targetName === 'Head' ? 0.48 : 0.58
        qUpperBodyDamped
          .copy(identityQuaternion)
          .slerp(qUpperBodyTargetDeltaLocal, upperBodyWeight)
          .normalize()

        qDesiredLocal
          .copy(qUpperBodyDamped)
          .multiply(targetBase.quaternion)
          .normalize()
      } else {
        qDesiredLocal.copy(targetBase.quaternion)
      }
    } else if (swing) {
      if (targetLocalDir && sourceChildIndex !== undefined) {
        sourceAnimatedDirection
          .copy(sourcePose.worldPositions[sourceChildIndex])
          .sub(sourcePose.worldPositions[sourceIndex])
          .normalize()
          .applyQuaternion(calibration.sourceToTargetWorld)
        const correction = calibration.segmentCorrections.get(targetName)
        const g1ArmSegment =
          calibration.embodiment === 'g1' &&
          (targetName === 'LeftArm' || targetName === 'LeftForeArm' ||
            targetName === 'RightArm' || targetName === 'RightForeArm')

        // The G1's native bind pose has both arms hanging down, while AIST++'s
        // BVH rest skeleton is essentially a T-pose. A constant quaternion that
        // rotates the source rest arm into the target rest arm is therefore NOT
        // an anatomical frame conversion: it also rotates real dance motion. In
        // practice, source "down/front" motion became robot "inward" motion
        // and sent both arms through the torso. For G1 arms, the global
        // sourceToTarget anatomical basis already gives the correct physical
        // direction. Aim the G1 segments directly along that direction and let
        // aimBoneAlongWorldDirection handle the arms-down bind orientation.
        if (correction && !g1ArmSegment) sourceAnimatedDirection.applyQuaternion(correction)
        sourceAnimatedDirection.normalize()
        aimBoneAlongWorldDirection(
          targetBone,
          targetBase,
          parentWorld,
          targetLocalDir,
          sourceAnimatedDirection,
          qDesiredWorld,
        )
        qDesiredLocal.copy(targetBone.quaternion)
      } else {
        // Some robot morphologies intentionally collapse multiple physical axes
        // onto one semantic joint. In the G1 driver, Shoulder and Arm occupy the
        // same physical shoulder pivot, so Shoulder→Arm has zero length. Falling
        // through to the generic full-orientation transfer here applied AIST++
        // clavicle rotations to that helper bone and pulled both arms inward into
        // an X. A zero-length swing helper has no segment direction to retarget,
        // so keep its native bind orientation and let the real Arm/ForeArm
        // segments carry the motion instead. XBot is unaffected because its
        // shoulder segments have a non-zero local direction.
        qDesiredLocal.copy(targetBase.quaternion)
      }
    } else {
      // Convert the full source-world rotation through one anatomical coordinate
      // basis before applying it to the target rest orientation. The old code
      // transferred source axes directly and could mirror the lower body.
      qDesiredWorld
        .copy(calibration.sourceToTargetWorld)
        .multiply(sourcePose.worldQuaternions[sourceIndex])
        .multiply(calibration.sourceToTargetWorldInverse)
        .multiply(targetBase.worldQuaternion)
        .normalize()
      qParentInv.copy(parentWorld).invert()
      qDesiredLocal.copy(qParentInv).multiply(qDesiredWorld).normalize()
    }

    if (quality === 'degraded') {
      qRestInv.copy(targetBase.quaternion).invert()
      qDelta.copy(qRestInv).multiply(qDesiredLocal).normalize()
      qAdjusted.copy(identityQuaternion).slerp(qDelta, spatialScale(targetName, localSeconds, seed))
      qAdjusted.multiply(degradationNudge(targetName, localSeconds, seed)).normalize()
      targetBone.quaternion.copy(targetBase.quaternion).multiply(qAdjusted).normalize()
    } else {
      targetBone.quaternion.copy(qDesiredLocal)
    }

    qFinalWorld.copy(parentWorld).multiply(targetBone.quaternion).normalize()
    const stored = targetAnimatedWorld.get(targetBone)
    if (stored) stored.copy(qFinalWorld)
    else targetAnimatedWorld.set(targetBone, qFinalWorld.clone())
  }

  // Lower body maps the live AIST++ thigh/shin segment directions directly and
  // preserves each knee/ankle lateral coordinate relative to the pelvis. This
  // avoids the fixed rest-pose correction that was pushing both knees inward.
  solveLegIK('Left', rig, targetRest, targetAnimatedWorld, motion, sourcePose, calibration, targetSegmentLocalDirections, quality, localSeconds, seed)
  solveLegIK('Right', rig, targetRest, targetAnimatedWorld, motion, sourcePose, calibration, targetSegmentLocalDirections, quality, localSeconds, seed)
}

function applyViewPose(camera: THREE.PerspectiveCamera, controls: OrbitControls, pose: ViewPose) {
  spherical.set(
    THREE.MathUtils.clamp(pose.distance, controls.minDistance, controls.maxDistance),
    THREE.MathUtils.clamp(pose.polar, controls.minPolarAngle, controls.maxPolarAngle),
    pose.azimuth,
  )
  tempVector.setFromSpherical(spherical).add(controls.target)
  camera.position.copy(tempVector)
  camera.lookAt(controls.target)
}

export function XBotScene({
  embodiment: requestedEmbodiment = 'g1',
  side,
  quality,
  motionFile,
  motionUrl,
  degradationSeed,
  startOffsetSeconds,
  paused = false,
  playhead = 0,
  autoRotate = true,
  showTrails = true,
  showLandmarks = true,
  resetViewSignal = 0,
  viewPose,
  onViewPoseChange,
}: Props) {
  const embodiment: Embodiment = requestedEmbodiment === 'g1' && ROBOT_MODEL === 'fairy' ? 'fairy' : requestedEmbodiment
  const mountRef = useRef<HTMLDivElement | null>(null)
  const pausedRef = useRef(paused)
  const playheadRef = useRef(playhead)
  const autoRotateRef = useRef(autoRotate)
  const showTrailsRef = useRef(showTrails)
  const showLandmarksRef = useRef(showLandmarks)
  const resetViewRef = useRef(resetViewSignal)
  const viewPoseRef = useRef(viewPose)
  const onViewPoseChangeRef = useRef(onViewPoseChange)
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [loadError, setLoadError] = useState('')
  const [motionMeta, setMotionMeta] = useState('')

  pausedRef.current = paused
  playheadRef.current = playhead
  autoRotateRef.current = autoRotate
  showTrailsRef.current = showTrails
  showLandmarksRef.current = showLandmarks
  resetViewRef.current = resetViewSignal
  viewPoseRef.current = viewPose
  onViewPoseChangeRef.current = onViewPoseChange

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    let disposed = false
    let raf = 0
    let model: THREE.Group | undefined
    let presentation: THREE.Group | undefined
    let turntable: THREE.Group | undefined
    let motion: BvhMotion | undefined
    let sourcePose: BvhPoseBuffers | undefined
    let sourceStartPose: BvhPoseBuffers | undefined
    let rig: Rig = {}
    let targetRest = new Map<THREE.Bone, RestPose>()
    let retargetCalibration: RetargetCalibration | undefined
    const targetSegmentRestDirections = new Map<TargetBoneName, THREE.Vector3>()
    const targetSegmentLocalDirections = new Map<TargetBoneName, THREE.Vector3>()
    const targetAnimatedWorld = new Map<THREE.Bone, THREE.Quaternion>()
    let rootScale = 1
    let rawModelHeight = 1
    let trails: Trail[] = []
    let lastPlayhead = -1
    let lastResetSignal = resetViewRef.current
    let modelLoaded = false
    let motionLoaded = false
    let userControlling = false

    const scene = new THREE.Scene()

    const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 220) // far covers the Aura world's distant mesas
    camera.position.set(0, 1.4, 5.35)

    const renderer = createAuraRenderer({ alpha: false, powerPreference: 'high-performance' }) // canvas MSAA only without the world composite
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.02

    // Marble skin of the XBot and G1 bodies; the fairy keeps her own textures, so it is not loaded for her.
    let marbleTexture: THREE.Texture | undefined
    if (embodiment !== 'fairy') {
      marbleTexture = new THREE.TextureLoader().load('/textures/marble-gold.png')
      marbleTexture.colorSpace = THREE.SRGBColorSpace
      marbleTexture.wrapS = THREE.RepeatWrapping
      marbleTexture.wrapT = THREE.RepeatWrapping
      marbleTexture.repeat.set(0.55, 0.55)
      // Larger-scale veining: repeat below 1 enlarges the marble pattern across the shell.
      marbleTexture.center.set(0.5, 0.5)
      marbleTexture.rotation = -0.08
      marbleTexture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy())
    }

    mount.appendChild(renderer.domElement)

    // Stylised Aura world (sky, ground, mesas, ink pass). The stage robot is
    // ~2.7 units tall, so the world is scaled ~2x relative to a real G1.
    const world = createAuraWorld(scene, renderer, { scale: 2.1, ring: false, props: { density: 0.8, keepOut: 3.3 } })
    world.observe(mount)
    const worldClock = new THREE.Clock()
    let robotFlare: RobotFlare | null = null
    let stylized: ToonStylizeHandle | null = null
    let fairyRig: FairyRig | null = null

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.075
    controls.enablePan = false
    controls.minDistance = 3.65
    controls.maxDistance = 7.4
    controls.minPolarAngle = Math.PI * 0.28
    controls.maxPolarAngle = Math.PI * 0.65
    controls.target.set(0, 1.28, 0)

    controls.addEventListener('start', () => { userControlling = true })
    controls.addEventListener('end', () => { userControlling = false })
    controls.addEventListener('change', () => {
      if (!userControlling) return
      onViewPoseChangeRef.current({
        azimuth: controls.getAzimuthalAngle(),
        polar: controls.getPolarAngle(),
        distance: camera.position.distanceTo(controls.target),
      })
    })

    const hemi = new THREE.HemisphereLight(0xfff4e4, 0x2a231c, 1.72)
    scene.add(hemi)

    // Warm gallery key: reads as antique gilt on metallic joints.
    const key = new THREE.DirectionalLight(0xfff0dc, 4.2)
    key.position.set(-4.2, 6.6, 4.4)
    key.castShadow = true
    key.shadow.mapSize.set(1024, 1024)
    key.shadow.camera.near = 0.1
    key.shadow.camera.far = 18
    scene.add(key)

    // Cool powder-blue fill keeps the ivory shell aligned with the website palette.
    const fill = new THREE.DirectionalLight(0xd8c7ad, 2.10)
    fill.position.set(4.8, 3.1, 4.2)
    scene.add(fill)

    const rim = new THREE.DirectionalLight(0xe6d6bd, 3.25)
    rim.position.set(3.8, 5.8, -4.8)
    scene.add(rim)

    const goldRim = new THREE.DirectionalLight(0xcdb898, 1.65)
    goldRim.position.set(-4.2, 3.8, -3.6)
    scene.add(goldRim)

    const top = new THREE.PointLight(0xfff4e4, 9.5, 8, 2)
    top.position.set(0, 4.8, 0.8)
    scene.add(top)

    // Renaissance turntable: parchment stone, enamel blue inlays, and antique-gold trim.
    // The character and stage rotate together when auto-rotate is enabled so
    // the presentation feels like a museum pedestal instead of a generic demo.
    const stageTopY = 0.665

    // The world ground (createAuraWorld) replaces the old dark floor disc and halos.

    turntable = new THREE.Group()
    turntable.name = `${BODY_LABEL[embodiment].short}-${side}-turntable`
    scene.add(turntable)

    const stageBase = new THREE.Mesh(
      new THREE.CylinderGeometry(2.1, 2.24, 0.34, 96, 1),
      new THREE.MeshPhysicalMaterial({
        color: 0x3d405b, // navy
        roughness: 0.58,
        metalness: 0.03,
        clearcoat: 0.24,
        clearcoatRoughness: 0.62,
      }),
    )
    stageBase.position.y = 0.17
    stageBase.castShadow = true
    stageBase.receiveShadow = true
    turntable.add(stageBase)

    const stageMid = new THREE.Mesh(
      new THREE.CylinderGeometry(1.76, 1.9, 0.15, 96, 1),
      new THREE.MeshPhysicalMaterial({
        color: 0x4f5268,
        roughness: 0.46,
        metalness: 0.02,
        clearcoat: 0.30,
        clearcoatRoughness: 0.54,
      }),
    )
    stageMid.position.y = 0.415
    stageMid.castShadow = true
    stageMid.receiveShadow = true
    turntable.add(stageMid)

    const stageTop = new THREE.Mesh(
      new THREE.CylinderGeometry(1.38, 1.46, 0.17, 96, 1),
      new THREE.MeshPhysicalMaterial({
        color: 0x81b29a, // sage
        roughness: 0.34,
        metalness: 0.01,
        clearcoat: 0.42,
        clearcoatRoughness: 0.38,
      }),
    )
    stageTop.position.y = 0.57
    stageTop.castShadow = true
    stageTop.receiveShadow = true
    turntable.add(stageTop)

    const topInlay = new THREE.Mesh(
      new THREE.CircleGeometry(1.26, 96),
      new THREE.MeshPhysicalMaterial({
        color: 0x6f9a8a,
        roughness: 0.44,
        metalness: 0.05,
        clearcoat: 0.52,
        clearcoatRoughness: 0.24,
      }),
    )
    topInlay.rotation.x = -Math.PI / 2
    topInlay.position.y = stageTopY + 0.002
    topInlay.receiveShadow = true
    turntable.add(topInlay)

    const topGoldRing = new THREE.Mesh(
      new THREE.RingGeometry(1.08, 1.18, 96),
      new THREE.MeshBasicMaterial({ color: 0xf2cc8f, transparent: true, opacity: 0.8, side: THREE.DoubleSide }),
    )
    topGoldRing.rotation.x = -Math.PI / 2
    topGoldRing.position.y = stageTopY + 0.004
    turntable.add(topGoldRing)

    const topBlueRing = new THREE.Mesh(
      new THREE.RingGeometry(0.54, 0.72, 96),
      new THREE.MeshBasicMaterial({ color: 0xe07a5f, transparent: true, opacity: 0.6, side: THREE.DoubleSide }),
    )
    topBlueRing.rotation.x = -Math.PI / 2
    topBlueRing.position.y = stageTopY + 0.005
    turntable.add(topBlueRing)

    const gildedTrimMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xe2c089,
      roughness: 0.26,
      metalness: 0.88,
      clearcoat: 0.22,
      clearcoatRoughness: 0.28,
    })
    const enamelTrimMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x5e6073,
      roughness: 0.34,
      metalness: 0.24,
      clearcoat: 0.34,
      clearcoatRoughness: 0.26,
    })

    const trimOuter = new THREE.Mesh(new THREE.TorusGeometry(1.46, 0.028, 18, 96), gildedTrimMaterial)
    trimOuter.rotation.x = Math.PI / 2
    trimOuter.position.y = 0.61
    turntable.add(trimOuter)

    const trimLower = new THREE.Mesh(new THREE.TorusGeometry(1.90, 0.036, 18, 96), enamelTrimMaterial)
    trimLower.rotation.x = Math.PI / 2
    trimLower.position.y = 0.41
    turntable.add(trimLower)

    const trimBase = new THREE.Mesh(new THREE.TorusGeometry(2.08, 0.042, 18, 96), gildedTrimMaterial)
    trimBase.rotation.x = Math.PI / 2
    trimBase.position.y = 0.18
    turntable.add(trimBase)

    const ribGeometry = new THREE.BoxGeometry(0.06, 0.20, 0.045)
    for (let i = 0; i < 20; i += 1) {
      const rib = new THREE.Mesh(ribGeometry, gildedTrimMaterial)
      const angle = (i / 20) * Math.PI * 2
      rib.position.set(Math.cos(angle) * 1.98, 0.17, Math.sin(angle) * 1.98)
      rib.rotation.y = -angle
      rib.castShadow = true
      turntable.add(rib)
    }

    const stageShadow = new THREE.Mesh(
      new THREE.CircleGeometry(2.28, 96),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.10 }),
    )
    stageShadow.rotation.x = -Math.PI / 2
    stageShadow.position.y = 0.003
    turntable.add(stageShadow)

    const base = import.meta.env.BASE_URL || '/'
    const normalizedBase = base.endsWith('/') ? base : `${base}/`
    const modelUrl = `${normalizedBase}models/xbot.fbx`

    setLoadState('loading')
    setLoadError('')
    setMotionMeta('')

    const maybeReady = () => {
      if (disposed || !modelLoaded || !motion) return
      if (!retargetCalibration) {
        try {
          retargetCalibration = buildRetargetCalibration(motion, rig, targetRest, targetSegmentRestDirections, embodiment)
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error)
          setLoadError(`Retarget calibration: ${message}`)
          setLoadState('error')
          return
        }
      }
      if (motionLoaded) setLoadState('ready')
    }

    const finishModel = (obj: THREE.Group) => {
      if (disposed) return
      model = obj
      presentation = new THREE.Group()
      presentation.name = `${BODY_LABEL[embodiment].short}-${side}-presentation`
      presentation.position.y = stageTopY
      presentation.add(obj)
      ;(turntable ?? scene).add(presentation)

      obj.traverse((child) => {
        const mesh = child as THREE.Mesh
        if (!mesh.isMesh) return
        mesh.castShadow = true
        mesh.receiveShadow = true
        mesh.frustumCulled = false
        // XBot receives the Renaissance marble skin. G1 uses the bundled official
        // URDF/STL geometry and its own white-and-gold robot materials, with the marble texture applied to the ivory shell parts.
        if (embodiment === 'xbot') {
          const originals = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
          const replacements = originals.map((material) => makeMaterial(`${mesh.name}_${material?.name ?? ''}`, material, marbleTexture))
          mesh.material = Array.isArray(mesh.material) ? replacements : replacements[0]
        }
      })

      // Paint the body (G1 or XBot) into the world's palette language (toonStylize.ts);
      // the fairy already gets it in createFairyMaterial (fairyRig.ts).
      stylized?.dispose()
      stylized = embodiment === 'fairy' ? null : toonStylize(obj)
      obj.updateMatrixWorld(true)
      const rawBox = new THREE.Box3().setFromObject(obj)
      rawModelHeight = Math.max(rawBox.max.y - rawBox.min.y, 0.0001)
      obj.scale.multiplyScalar(2.72 / rawModelHeight)
      obj.updateMatrixWorld(true)

      const box = new THREE.Box3().setFromObject(obj)
      const center = box.getCenter(new THREE.Vector3())
      obj.position.x -= center.x
      obj.position.z -= center.z
      obj.position.y -= box.min.y
      obj.updateMatrixWorld(true)

      const fitted = new THREE.Box3().setFromObject(obj)
      const fittedSize = fitted.getSize(new THREE.Vector3())
      controls.target.set(0, stageTopY + fittedSize.y * 0.47, 0)
      applyViewPose(camera, controls, viewPoseRef.current)
      controls.update()

      rig = makeRig(obj)
      const missingTargetBones = TARGET_BONES.filter((name) => !rig[name])
      if (missingTargetBones.length > 0) {
        const discovered = listRigBoneNames(obj)
        const preview = discovered.slice(0, 24).join(', ')
        setLoadError(
          `${BODY_LABEL[embodiment].short} rig mapping failed for: ${missingTargetBones.join(', ')}. ` +
          `Runtime bones (${discovered.length}): ${preview}${discovered.length > 24 ? ', …' : ''}`,
        )
        setLoadState('error')
        return
      }

      targetRest = captureRestPose(obj, rig)
      for (const bone of Object.values(rig)) {
        if (bone) targetAnimatedWorld.set(bone, new THREE.Quaternion())
      }

      for (const targetName of TARGET_BONES) {
        const segment = SWING_SEGMENTS[targetName]
        const bone = rig[targetName]
        if (!segment || !bone) continue
        const child = findBone(obj, segment.targetChild)
        const base = targetRest.get(bone)
        if (!child || !base) continue
        const childWorld = child.getWorldPosition(new THREE.Vector3())
        const direction = childWorld.sub(base.worldPosition)
        if (direction.lengthSq() > 1e-8) targetSegmentRestDirections.set(targetName, direction.normalize())
        if (child.position.lengthSq() > 1e-8) targetSegmentLocalDirections.set(targetName, child.position.clone().normalize())
      }

      const trailDefinitions: Array<[THREE.Object3D | undefined, number]> = [
        [findBone(obj, 'LeftHandMiddle4') ?? rig.LeftHand, 0xe6d6bd],
        [findBone(obj, 'RightHandMiddle4') ?? rig.RightHand, 0xf4ecdf],
        [findBone(obj, 'LeftToe_End') ?? findBone(obj, 'LeftToeBase') ?? rig.LeftFoot, 0xbfae94],
        [findBone(obj, 'RightToe_End') ?? findBone(obj, 'RightToeBase') ?? rig.RightFoot, 0xd8c7ad],
      ]
      trails = trailDefinitions
        .filter((entry): entry is [THREE.Object3D, number] => Boolean(entry[0]))
        .map(([endEffector, color]) => createTrail(endEffector, color))
      for (const trail of trails) {
        presentation.add(trail.glow)
        presentation.add(trail.segments)
        presentation.add(trail.marker)
      }
      // Fairy flare riding the fastest of hands, feet and head (none under reduced motion).
      if (!prefersReducedMotion()) {
        const limbs = [...trails.map((trail) => trail.endEffector), rig.Head].filter((o): o is THREE.Object3D => Boolean(o))
        robotFlare?.dispose()
        robotFlare = new RobotFlare(limbs.map((object) => ({ object })), { scale: 2.1, strength: 0.65, layer: AURA_OVERLAY_LAYER })
        scene.add(robotFlare.points)
      }

      modelLoaded = true
      if (motion) rootScale = rawModelHeight / motion.sourceHeight
      maybeReady()
    }

    if (embodiment === 'fairy') {
      // Shared cached GLB; this viewer gets its own skinned clone and toon material over her baked textures.
      loadFairyAsset()
        .then((asset) => {
          if (disposed) return
          fairyRig = new FairyRig(asset)
          const body = createFairyMaterial(asset) // one material for every fairy mesh of this viewer
          for (const mesh of fairyRig.meshes) mesh.material = body
          const holder = new THREE.Group()
          holder.add(fairyRig.root)
          finishModel(holder)
        })
        .catch((error: unknown) => {
          if (disposed) return
          const message = error instanceof Error ? error.message : String(error)
          setLoadError(`Blossom Fairy: ${message || 'Could not load the GLB.'}`)
          setLoadState('error')
        })
    } else if (embodiment === 'g1') {
      buildG1OfficialModel(`${normalizedBase}models/g1/`, marbleTexture)
        .then((obj) => finishModel(obj))
        .catch((error: unknown) => {
          if (disposed) return
          const message = error instanceof Error ? error.message : String(error)
          setLoadError(`Unitree G1: ${message || 'Could not load the bundled URDF/STL geometry.'}`)
          setLoadState('error')
        })
    } else {
      new FBXLoader().load(
        modelUrl,
        (obj) => finishModel(obj),
        undefined,
        (error) => {
          if (disposed) return
          const message = error instanceof Error ? error.message : String(error)
          setLoadError(`XBot: ${message || 'FBXLoader could not parse the model.'}`)
          setLoadState('error')
        },
      )
    }

    loadBVH(motionUrl)
      .then((loadedMotion) => {
        if (disposed) return
        const missingSourceBones = TARGET_BONES
          .map((name) => SOURCE_FOR_TARGET[name])
          .filter((name) => !loadedMotion.nameToIndex.has(name))
        if (missingSourceBones.length > 0) {
          throw new Error(`BVH is missing mapped joints: ${[...new Set(missingSourceBones)].join(', ')}`)
        }

        motion = loadedMotion
        sourcePose = createBvhPoseBuffers(loadedMotion)
        sourceStartPose = createBvhPoseBuffers(loadedMotion)
        const safeStart = Math.min(startOffsetSeconds, Math.max(0, loadedMotion.duration - STUDY_CLIP_SECONDS))
        sampleBVHWorldPose(loadedMotion, safeStart, sourceStartPose)
        motionLoaded = true
        if (modelLoaded) rootScale = rawModelHeight / loadedMotion.sourceHeight
        setMotionMeta(`${loadedMotion.frameCount} frames · ${(1 / loadedMotion.frameTime).toFixed(0)} fps · ${TARGET_BONES.length}/${TARGET_BONES.length} mapped`)
        maybeReady()
      })
      .catch((error: unknown) => {
        if (disposed) return
        const message = error instanceof Error ? error.message : String(error)
        setLoadError(`Motion: ${message || 'Could not parse the bundled AIST++ BVH.'}`)
        setLoadState('error')
      })

    const resize = () => {
      const width = Math.max(mount.clientWidth, 1)
      const height = Math.max(mount.clientHeight, 1)
      renderer.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
    }

    const observer = new ResizeObserver(resize)
    observer.observe(mount)
    resize()

    const sampleWorld = new THREE.Vector3()
    const sampleLocal = new THREE.Vector3()

    const clearTrails = () => {
      trails.forEach((trail) => {
        trail.points = []
        trail.lastSample = -1
        setTrailSegments(trail)
      })
    }

    const tick = () => {
      const worldDt = worldClock.getDelta()
      if (!world.visible) {
        raf = requestAnimationFrame(tick)
        return
      }
      const normalized = THREE.MathUtils.clamp(playheadRef.current / 100, 0, 1)
      const seconds = normalized * STUDY_CLIP_SECONDS

      if (model && motion && sourcePose && sourceStartPose && retargetCalibration) {
        applyRetargetedPose(
          rig,
          targetRest,
          targetAnimatedWorld,
          motion,
          sourcePose,
          sourceStartPose,
          seconds,
          startOffsetSeconds,
          quality,
          degradationSeed,
          rootScale,
          retargetCalibration,
          targetSegmentLocalDirections,
        )
        model.updateMatrixWorld(true)
      }

      if (turntable) {
        turntable.rotation.y = autoRotateRef.current
          ? THREE.MathUtils.lerp(-0.65, 0.65, normalized)
          : 0
        turntable.updateMatrixWorld(true)
      }

      if (presentation) {
        presentation.rotation.y = 0
        presentation.updateMatrixWorld(true)
      }

      if (!userControlling) applyViewPose(camera, controls, viewPoseRef.current)

      if (lastResetSignal !== resetViewRef.current) {
        lastResetSignal = resetViewRef.current
        clearTrails()
      }

      if (lastPlayhead >= 0 && playheadRef.current + 1 < lastPlayhead) clearTrails()
      lastPlayhead = playheadRef.current

      if (presentation) {
        for (const trail of trails) {
          trail.segments.visible = showTrailsRef.current
          trail.glow.visible = showTrailsRef.current
          trail.marker.visible = showLandmarksRef.current
          trail.endEffector.getWorldPosition(sampleWorld)
          sampleLocal.copy(sampleWorld)
          presentation.worldToLocal(sampleLocal)
          trail.marker.position.copy(sampleLocal)

          if (showTrailsRef.current) {
            if (seconds - trail.lastSample > 1 / 30) {
              trail.lastSample = seconds
              trail.points.push(sampleLocal.clone())
              // Keep only ~0.8 seconds at 30 samples/sec. The shorter history
              // makes the trajectory read as a fading gesture rather than a
              // persistent scribble over the whole viewport.
              if (trail.points.length > MAX_TRAIL_POINTS) trail.points.shift()
            }
            // Always append the current effector as the visual head of the
            // polyline, even between history samples, so the trace never
            // visibly detaches from the hand/foot.
            setTrailSegments(trail, sampleLocal)
          }
        }
      }

      controls.update()
      if (robotFlare) {
        camera.updateMatrixWorld()
        // the turntable / presentation updateMatrixWorld above already refreshed the posed body
        robotFlare.update(worldDt, renderer.getPixelRatio(), camera, Math.max(mount.clientHeight, 1), 1, true)
      }
      world.render(camera, worldDt)
      raf = requestAnimationFrame(tick)
    }
    tick()

    return () => {
      disposed = true
      cancelAnimationFrame(raf)
      observer.disconnect()
      controls.dispose()
      robotFlare?.dispose()
      stylized?.dispose()
      world.dispose()

      scene.traverse((object) => {
        const mesh = object as THREE.Mesh
        if (!mesh.userData?.sharedGeometry) mesh.geometry?.dispose?.() // the fairy's geometry belongs to the cached GLB
        if (!mesh.material) return
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
        materials.forEach((material) => material?.dispose?.())
      })
      fairyRig?.dispose() // after the traverse, so its per-viewer materials were disposed above

      marbleTexture?.dispose()
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [embodiment, side, quality, motionFile, motionUrl, degradationSeed, startOffsetSeconds])

  return (
    <div className="xbot-scene" ref={mountRef}>
      {loadState === 'loading' && <div className="xbot-status">Loading {BODY_LABEL[embodiment].loading} + bundled AIST++ BVH…</div>}
      {loadState === 'ready' && (
        <div className="motion-source-chip" title={motionFile}>
          AIST++ → {BODY_LABEL[embodiment].chip} · morphology retarget · {motionMeta}
        </div>
      )}
      {loadState === 'error' && (
        <div className="xbot-status xbot-error">
          <strong>Motion viewer failed to load</strong>
          <span>{loadError}</span>
          <small>BVHs are bundled under src/assets/motions/. G1 uses the bundled Unitree URDF/STL geometry with Aura’s semantic kinematic retarget; XBot uses public/models/xbot.fbx.</small>
        </div>
      )}
    </div>
  )
}
