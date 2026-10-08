import * as THREE from 'three'

export type BvhJoint = {
  name: string
  parent: number
  offset: THREE.Vector3
  channels: string[]
  channelStart: number
}

export type BvhMotion = {
  joints: BvhJoint[]
  nameToIndex: Map<string, number>
  frameTime: number
  frameCount: number
  channelCount: number
  frames: Float32Array
  duration: number
  restWorldPositions: THREE.Vector3[]
  sourceHeight: number
}

export type BvhPoseBuffers = {
  worldPositions: THREE.Vector3[]
  worldQuaternions: THREE.Quaternion[]
}

const axisX = new THREE.Vector3(1, 0, 0)
const axisY = new THREE.Vector3(0, 1, 0)
const axisZ = new THREE.Vector3(0, 0, 1)
const qAxis = new THREE.Quaternion()
const qA = new THREE.Quaternion()
const qB = new THREE.Quaternion()
const qLocal = new THREE.Quaternion()
const pA = new THREE.Vector3()
const pB = new THREE.Vector3()
const pLocal = new THREE.Vector3()

function expect(tokens: string[], cursor: { value: number }, wanted: string) {
  const got = tokens[cursor.value++]
  if (got !== wanted) throw new Error(`BVH parse error: expected “${wanted}”, got “${got ?? 'EOF'}”.`)
}

function numberToken(tokens: string[], cursor: { value: number }) {
  const token = tokens[cursor.value++]
  const value = Number(token)
  if (!Number.isFinite(value)) throw new Error(`BVH parse error: expected number, got “${token ?? 'EOF'}”.`)
  return value
}

export function parseBVH(text: string): BvhMotion {
  const tokens = text.match(/[{}]|[^\s{}]+/g) ?? []
  const cursor = { value: 0 }
  const joints: BvhJoint[] = []
  let channelCount = 0

  expect(tokens, cursor, 'HIERARCHY')

  function parseEndSite() {
    expect(tokens, cursor, 'Site')
    expect(tokens, cursor, '{')
    expect(tokens, cursor, 'OFFSET')
    numberToken(tokens, cursor)
    numberToken(tokens, cursor)
    numberToken(tokens, cursor)
    expect(tokens, cursor, '}')
  }

  function parseJoint(keyword: 'ROOT' | 'JOINT', parent: number) {
    expect(tokens, cursor, keyword)
    const name = tokens[cursor.value++]
    if (!name) throw new Error('BVH parse error: joint name missing.')
    expect(tokens, cursor, '{')

    const index = joints.length
    const joint: BvhJoint = {
      name,
      parent,
      offset: new THREE.Vector3(),
      channels: [],
      channelStart: channelCount,
    }
    joints.push(joint)

    while (cursor.value < tokens.length && tokens[cursor.value] !== '}') {
      const token = tokens[cursor.value]
      if (token === 'OFFSET') {
        cursor.value += 1
        joint.offset.set(
          numberToken(tokens, cursor),
          numberToken(tokens, cursor),
          numberToken(tokens, cursor),
        )
      } else if (token === 'CHANNELS') {
        cursor.value += 1
        const count = numberToken(tokens, cursor)
        joint.channelStart = channelCount
        joint.channels = []
        for (let i = 0; i < count; i += 1) {
          const channel = tokens[cursor.value++]
          if (!channel) throw new Error(`BVH parse error: channel missing for ${name}.`)
          joint.channels.push(channel)
          channelCount += 1
        }
      } else if (token === 'JOINT') {
        parseJoint('JOINT', index)
      } else if (token === 'End') {
        cursor.value += 1
        parseEndSite()
      } else {
        throw new Error(`BVH parse error in ${name}: unexpected token “${token}”.`)
      }
    }

    expect(tokens, cursor, '}')
  }

  parseJoint('ROOT', -1)
  expect(tokens, cursor, 'MOTION')
  expect(tokens, cursor, 'Frames:')
  const frameCount = numberToken(tokens, cursor)
  expect(tokens, cursor, 'Frame')
  expect(tokens, cursor, 'Time:')
  const frameTime = numberToken(tokens, cursor)

  if (frameCount < 1 || channelCount < 1 || frameTime <= 0) {
    throw new Error('BVH parse error: invalid motion header.')
  }

  const expectedValues = frameCount * channelCount
  const frames = new Float32Array(expectedValues)
  for (let i = 0; i < expectedValues; i += 1) {
    frames[i] = numberToken(tokens, cursor)
  }

  const nameToIndex = new Map<string, number>()
  joints.forEach((joint, index) => nameToIndex.set(joint.name, index))

  const restWorldPositions = joints.map(() => new THREE.Vector3())
  for (let i = 0; i < joints.length; i += 1) {
    const joint = joints[i]
    if (joint.parent < 0) restWorldPositions[i].copy(joint.offset)
    else restWorldPositions[i].copy(restWorldPositions[joint.parent]).add(joint.offset)
  }

  let minY = Infinity
  let maxY = -Infinity
  for (const position of restWorldPositions) {
    minY = Math.min(minY, position.y)
    maxY = Math.max(maxY, position.y)
  }

  return {
    joints,
    nameToIndex,
    frameTime,
    frameCount,
    channelCount,
    frames,
    duration: Math.max(0, (frameCount - 1) * frameTime),
    restWorldPositions,
    sourceHeight: Math.max(maxY - minY, 0.001),
  }
}

export function createBvhPoseBuffers(motion: BvhMotion): BvhPoseBuffers {
  return {
    worldPositions: motion.joints.map(() => new THREE.Vector3()),
    worldQuaternions: motion.joints.map(() => new THREE.Quaternion()),
  }
}

function readJointLocalAtFrame(
  motion: BvhMotion,
  joint: BvhJoint,
  frame: number,
  outPosition: THREE.Vector3,
  outQuaternion: THREE.Quaternion,
) {
  outPosition.copy(joint.offset)
  outQuaternion.identity()

  const frameBase = frame * motion.channelCount + joint.channelStart
  for (let i = 0; i < joint.channels.length; i += 1) {
    const channel = joint.channels[i]
    const value = motion.frames[frameBase + i]

    if (channel === 'Xposition') outPosition.x += value
    else if (channel === 'Yposition') outPosition.y += value
    else if (channel === 'Zposition') outPosition.z += value
    else if (channel === 'Xrotation') outQuaternion.multiply(qAxis.setFromAxisAngle(axisX, THREE.MathUtils.degToRad(value)))
    else if (channel === 'Yrotation') outQuaternion.multiply(qAxis.setFromAxisAngle(axisY, THREE.MathUtils.degToRad(value)))
    else if (channel === 'Zrotation') outQuaternion.multiply(qAxis.setFromAxisAngle(axisZ, THREE.MathUtils.degToRad(value)))
  }
  outQuaternion.normalize()
}

export function sampleBVHWorldPose(motion: BvhMotion, timeSeconds: number, out: BvhPoseBuffers) {
  const frameFloat = THREE.MathUtils.clamp(timeSeconds / motion.frameTime, 0, motion.frameCount - 1)
  const frameA = Math.floor(frameFloat)
  const frameB = Math.min(frameA + 1, motion.frameCount - 1)
  const alpha = frameFloat - frameA

  for (let i = 0; i < motion.joints.length; i += 1) {
    const joint = motion.joints[i]
    readJointLocalAtFrame(motion, joint, frameA, pA, qA)
    readJointLocalAtFrame(motion, joint, frameB, pB, qB)
    pLocal.copy(pA).lerp(pB, alpha)
    qLocal.slerpQuaternions(qA, qB, alpha).normalize()

    if (joint.parent < 0) {
      out.worldPositions[i].copy(pLocal)
      out.worldQuaternions[i].copy(qLocal)
    } else {
      const parentPosition = out.worldPositions[joint.parent]
      const parentQuaternion = out.worldQuaternions[joint.parent]
      out.worldQuaternions[i].copy(parentQuaternion).multiply(qLocal).normalize()
      out.worldPositions[i].copy(pLocal).applyQuaternion(parentQuaternion).add(parentPosition)
    }
  }
}

const parsedCache = new Map<string, Promise<BvhMotion>>()

export function loadBVH(url: string) {
  const existing = parsedCache.get(url)
  if (existing) return existing

  const promise = fetch(url)
    .then((response) => {
      if (!response.ok) throw new Error(`BVH request failed (${response.status}) for ${url}`)
      return response.text()
    })
    .then(parseBVH)

  parsedCache.set(url, promise)
  return promise
}
