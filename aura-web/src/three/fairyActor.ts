/**
 * Hero-arena actor for the Blossom Fairy: same interface as G1Actor (group, setClip,
 * setTime, setLook, dispose), driven by the same PreparedClip (in place, facing +z,
 * floor-normalised). Toon shading over her baked colours + ink inverted hull.
 */
import * as THREE from 'three'
import type { ActorPalette, PreparedClip } from './g1Actor'
import { FairyRig, createPoseSample, dressFairy, type FairyAsset } from './fairyRig'
import type { MixamoCharacter } from './characters'

const J = 34

/** What the arena needs from a robot, whichever body draws it. */
export interface ArenaActor {
  readonly group: THREE.Group
  readonly duration: number
  setClip(clip: PreparedClip | null): void
  setTime(t: number): void
  setLook(highlight: number, dim: number): void
  dispose(): void
}

export class FairyActor implements ArenaActor {
  static RIM = new THREE.Color(0xfff2dc)
  readonly group = new THREE.Group()
  readonly rig: FairyRig
  private fit = new THREE.Group()
  private bodyMat: THREE.MeshToonMaterial
  private outlineMat: THREE.MeshBasicMaterial
  private dimTint = new THREE.Color()
  private clip: PreparedClip | null = null
  private sample = createPoseSample()
  private tmpA = new THREE.Quaternion()
  private tmpB = new THREE.Quaternion()

  /** displayHeight: bind-pose height the fairy is fitted to, in arena metres. */
  /** def: the registry entry (features: fingers / wings); defaults to the selected Mixamo character. */
  constructor(asset: FairyAsset, palette: ActorPalette, dimColor: number, opts: { displayHeight: number }, def?: MixamoCharacter) {
    this.rig = new FairyRig(asset, def)
    this.dimTint.set(dimColor)
    // Her baked colours stay; the arena tones act on top (colour multiplier + emissive rim).
    const { body, ink } = dressFairy(this.rig, asset, { ink: palette.outline })
    this.bodyMat = body
    this.outlineMat = ink
    this.fit.scale.setScalar(opts.displayHeight / this.rig.height)
    this.fit.add(this.rig.root)
    this.group.add(this.fit)
    this.group.name = 'BlossomFairyActor'
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
    const s = this.sample
    for (let j = 0; j < J; j++) {
      const o0 = i0 * J + j, o1 = i1 * J + j
      for (let k = 0; k < 3; k++) s.pos[j * 3 + k] = c.pos[o0 * 3 + k] + (c.pos[o1 * 3 + k] - c.pos[o0 * 3 + k]) * a
      this.tmpA.set(c.quat[o0 * 4], c.quat[o0 * 4 + 1], c.quat[o0 * 4 + 2], c.quat[o0 * 4 + 3])
      this.tmpB.set(c.quat[o1 * 4], c.quat[o1 * 4 + 1], c.quat[o1 * 4 + 2], c.quat[o1 * 4 + 3])
      this.tmpA.slerp(this.tmpB, a)
      s.quat[j * 4] = this.tmpA.x; s.quat[j * 4 + 1] = this.tmpA.y; s.quat[j * 4 + 2] = this.tmpA.z; s.quat[j * 4 + 3] = this.tmpA.w
    }
    this.rig.applyPose(s)
  }

  /** highlight 0..1 (hover rim), dim 0..1 (lost the vote). */
  setLook(highlight: number, dim: number) {
    // White keeps the baked map as is; a vote loss tints it toward the arena's muted tone.
    this.bodyMat.color.setRGB(1, 1, 1).lerp(this.dimTint, dim * 0.55)
    this.bodyMat.emissive.setHex(0x14100c).lerp(FairyActor.RIM, highlight)
    this.bodyMat.emissiveIntensity = 0.025 + highlight * 0.12 // lighter than the G1's 0.22: keeps her baked colours readable
    this.outlineMat.opacity = 0.72 + highlight * 0.28 - dim * 0.4
  }

  dispose() {
    this.bodyMat.dispose(); this.outlineMat.dispose()
    this.rig.dispose()
    this.group.removeFromParent()
  }
}
