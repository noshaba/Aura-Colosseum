/**
 * Character registry: which bodies can perform Aura motions, and which one is selected.
 *
 * Every Aura motion is recorded on the Unitree G1 34-joint skeleton. A character is a
 * body that can be driven by those joints. Adding a model is ONE entry in `CHARACTERS`
 * (plus its asset under public/models/custom/, see the README there).
 *
 * Kinds and their contract:
 *
 * - 'g1-rigid': rigid STL parts placed per G1 joint (g1Rig.ts / g1Actor.ts), loaded from
 *   `modelBase` (g1.xml + meshes/). The viewer applies `toonStylize(root)` to it.
 *   `effectors` matches mesh names (the STL file names g1Actor sets on each part).
 *
 * - 'mixamo-retarget': one skinned GLB at `url` (relative to BASE_URL) with a Mixamo
 *   skeleton (`mixamorig:*` bones; GLTFLoader strips the colon, so `mixamorigHips`).
 *   Required bones: Hips, Spine, Spine1, Spine2, Neck, Head, {Left,Right}{UpLeg,Leg,Foot,
 *   ToeBase,Shoulder,Arm,ForeArm,Hand}. Rotations are retargeted from the G1 joints
 *   through a neutral pose computed from the bind pose (fairyRig.ts), so any bind pose
 *   (T, A or running lean) works. The material is stylised by the rig loader
 *   (createFairyMaterial today), not by `toonStylize(root)`.
 *   `effectors` matches bone names (after the loader's name sanitising).
 *   Optional `features` switch rig extras on without branching on the id:
 *     fingers: `{Left,Right}Hand{Thumb,Index,Middle,Ring,Pinky}{1,2,3}` bones exist and get
 *              the procedural curl (G1 has no fingers).
 *     wings:   regex of wing bones (root + optional `_tip` child under Spine2) that beat
 *              procedurally and whose skin weights mark the wing panes for the toon pass.
 *
 * Runtime selection: getCharacterId / setCharacterId / subscribeCharacter, persisted in
 * localStorage ('aura:character:v1') and synced across tabs. This module is data plus a
 * tiny store: it must not import fairyRig.ts, heroArenaScene.ts or any viewer.
 */

export type CharacterKind = 'g1-rigid' | 'mixamo-retarget'
/** Line icon shown on the switch and in the picker (components/CharacterIcon.tsx). */
export type CharacterIconId = 'robot' | 'fairy'

type CharacterBase = {
  /** Stable id, persisted in localStorage. Never rename a shipped id. */
  id: string
  /** Name in the picker. */
  label: string
  /** Short uppercase tag for compact overlays (e.g. "AURA -> FAIRY"). */
  tag: string
  /** Switch / picker icon; add a new id to CharacterIcon.tsx for a new kind of character. */
  icon: CharacterIconId
  /** Optional picker thumbnail (path relative to BASE_URL, square, ~64 px). */
  thumbnail?: string
  /** Optional attribution for the model (shown as a tooltip in the picker). */
  credit?: string
  /** Flare / trail effectors (hands, feet, head) for effectorsByName(root, effectors). */
  effectors: RegExp
}

export type G1RigidCharacter = CharacterBase & {
  kind: 'g1-rigid'
  /** Folder with g1.xml + meshes/, relative to BASE_URL, trailing slash. */
  modelBase: string
}

export type MixamoCharacter = CharacterBase & {
  kind: 'mixamo-retarget'
  /** Skinned GLB, relative to BASE_URL. */
  url: string
  features: {
    /** Finger bones present: procedural curl. */
    fingers: boolean
    /** Wing bone pattern (procedural beat + wing pane weights), or null. */
    wings: RegExp | null
  }
}

export type CharacterDef = G1RigidCharacter | MixamoCharacter

/** Order = picker order. Add new models here. */
export const CHARACTERS: readonly CharacterDef[] = [
  {
    id: 'fairy',
    label: 'Blossom Fairy',
    tag: 'FAIRY',
    icon: 'fairy',
    kind: 'mixamo-retarget',
    url: 'models/custom/blossom-fairy.rigged.web.glb',
    effectors: /^(mixamorig)?(LeftHand|RightHand|LeftToeBase|RightToeBase|Head)$/,
    features: { fingers: true, wings: /wing_[LR]/i },
  },
  {
    id: 'g1',
    label: 'Unitree G1',
    tag: 'G1',
    icon: 'robot',
    kind: 'g1-rigid',
    modelBase: 'models/g1-native/',
    effectors: /rubber_hand|ankle_roll_link|head_link/i,
  },
]

export const DEFAULT_CHARACTER_ID = 'fairy'
const STORAGE_KEY = 'aura:character:v1'

const byId = new Map(CHARACTERS.map(c => [c.id, c]))
if (import.meta.env.DEV && byId.size !== CHARACTERS.length) console.error('characters.ts: duplicate character id')

export function getCharacter(id: string): CharacterDef {
  return byId.get(id) ?? byId.get(DEFAULT_CHARACTER_ID)!
}

/** Unknown or missing ids resolve to the default. */
const normalize = (id: string | null | undefined) => (id && byId.has(id) ? id : DEFAULT_CHARACTER_ID)

function readStored(): string {
  try { return normalize(window.localStorage.getItem(STORAGE_KEY)) } catch { return DEFAULT_CHARACTER_ID }
}

let current: string | null = null
const listeners = new Set<() => void>()
let storageBound = false

function emit() { listeners.forEach(cb => cb()) }

function onStorage(e: StorageEvent) {
  if (e.key !== null && e.key !== STORAGE_KEY) return
  const next = normalize(e.key === null ? null : e.newValue)
  if (next === current) return
  current = next
  emit()
}

/** Currently selected character id (always a registered id). */
export function getCharacterId(): string {
  if (current === null) current = typeof window === 'undefined' ? DEFAULT_CHARACTER_ID : readStored()
  return current
}

export function getCurrentCharacter(): CharacterDef {
  return getCharacter(getCharacterId())
}

/** Select a character; unknown ids fall back to the default. Persists and notifies subscribers. */
export function setCharacterId(id: string) {
  const next = normalize(id)
  if (next === getCharacterId()) return
  current = next
  try { window.localStorage.setItem(STORAGE_KEY, next) } catch { /* private mode / blocked storage: keep in memory */ }
  emit()
}

/** Called whenever the selection changes (this tab or another). Returns the unsubscribe. */
export function subscribeCharacter(cb: () => void): () => void {
  listeners.add(cb)
  if (!storageBound && typeof window !== 'undefined') {
    window.addEventListener('storage', onStorage)
    storageBound = true
  }
  return () => {
    listeners.delete(cb)
    if (listeners.size === 0 && storageBound) {
      window.removeEventListener('storage', onStorage)
      storageBound = false
    }
  }
}
