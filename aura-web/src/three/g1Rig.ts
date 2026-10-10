/**
 * Shared Unitree G1 STL rig loading (MuJoCo g1.xml geom transforms + STL meshes).
 * Used by GeneratedG1RobotPreview and the hero arena; the geometry and transform
 * caches are module-level so both share one download/parse of every mesh.
 */
import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'

export type G1Preview = {
  format: 'g1-joints-v2'
  fps: number
  positions: number[][][]
  global_rot_mats: number[][][][]
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

export const FILE_TO_JOINT = new Map<string, number>()
Object.entries(G1_MESH_JOINTS).forEach(([joint, files]) => files.forEach(file => FILE_TO_JOINT.set(file, Number(joint))))

// MuJoCo (z-up, x-forward) -> NVIDIA Kimodo · Aura integration / Three.js (y-up, z-forward).
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

export function matrixToQuat(matrix: number[][]) {
  const m = new THREE.Matrix4().set(
    matrix[0][0], matrix[0][1], matrix[0][2], 0,
    matrix[1][0], matrix[1][1], matrix[1][2], 0,
    matrix[2][0], matrix[2][1], matrix[2][2], 0,
    0, 0, 0, 1,
  )
  return new THREE.Quaternion().setFromRotationMatrix(m).normalize()
}

function mujocoQuatToText2MotionAura(wxyz: number[]) {
  const q = new THREE.Quaternion(wxyz[1], wxyz[2], wxyz[3], wxyz[0]).normalize()
  const r = new THREE.Matrix4().makeRotationFromQuaternion(q)
  const converted = MUJOCO_TO_TEXT2MOTION_AURA.clone().multiply(r).multiply(TEXT2MOTION_AURA_TO_MUJOCO)
  return new THREE.Quaternion().setFromRotationMatrix(converted).normalize()
}

export async function loadMeshTransforms(base: string) {
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
      byFile.set(file, { pos, quat: mujocoQuatToText2MotionAura([w, qx, qy, qz]) })
    })
    return byFile
  })()
  // Do not cache a failed request forever.
  transformCache.catch(() => { transformCache = null })
  return transformCache
}

/** Cached, converted STL geometry. Callers that dispose must clone() first. */
export function loadGeometry(url: string) {
  let promise = geometryCache.get(url)
  if (!promise) {
    promise = new STLLoader().loadAsync(url).then(geometry => {
      geometry.applyMatrix4(MUJOCO_TO_TEXT2MOTION_AURA)
      geometry.computeVertexNormals()
      return geometry
    })
    promise.catch(() => geometryCache.delete(url))
    geometryCache.set(url, promise)
  }
  return promise
}
