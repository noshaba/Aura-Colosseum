export type MotionSide = 'A' | 'B'
export type MotionQuality = 'reference' | 'degraded'
export const STUDY_CLIP_SECONDS = 6

export type AistReferenceMotion = {
  id: string
  label: string
  file: string
  url: string
  startOffsetSeconds: number
  seed: number
}

const urls = {
  'gBR_sBM_cAll_d04_mBR0_ch01.bvh': new URL('./assets/motions/gBR_sBM_cAll_d04_mBR0_ch01.bvh', import.meta.url).href,
  'gPO_sBM_cAll_d10_mPO0_ch01.bvh': new URL('./assets/motions/gPO_sBM_cAll_d10_mPO0_ch01.bvh', import.meta.url).href,
  'gLO_sBM_cAll_d13_mLO0_ch04.bvh': new URL('./assets/motions/gLO_sBM_cAll_d13_mLO0_ch04.bvh', import.meta.url).href,
  'gMH_sBM_cAll_d22_mMH0_ch01.bvh': new URL('./assets/motions/gMH_sBM_cAll_d22_mMH0_ch01.bvh', import.meta.url).href,
  'gWA_sBM_cAll_d25_mWA0_ch02.bvh': new URL('./assets/motions/gWA_sBM_cAll_d25_mWA0_ch02.bvh', import.meta.url).href,
  'gKR_sBM_cAll_d28_mKR0_ch01.bvh': new URL('./assets/motions/gKR_sBM_cAll_d28_mKR0_ch01.bvh', import.meta.url).href,
} as const

export const AIST_REFERENCE_MOTIONS: AistReferenceMotion[] = [
  { id: 'break', label: 'Break dance', file: 'gBR_sBM_cAll_d04_mBR0_ch01.bvh', url: urls['gBR_sBM_cAll_d04_mBR0_ch01.bvh'], startOffsetSeconds: 1.15, seed: 1103 },
  { id: 'pop', label: 'Pop', file: 'gPO_sBM_cAll_d10_mPO0_ch01.bvh', url: urls['gPO_sBM_cAll_d10_mPO0_ch01.bvh'], startOffsetSeconds: 2.10, seed: 2179 },
  { id: 'lock', label: 'Lock', file: 'gLO_sBM_cAll_d13_mLO0_ch04.bvh', url: urls['gLO_sBM_cAll_d13_mLO0_ch04.bvh'], startOffsetSeconds: 1.65, seed: 3251 },
  { id: 'middlehiphop', label: 'Middle hip-hop', file: 'gMH_sBM_cAll_d22_mMH0_ch01.bvh', url: urls['gMH_sBM_cAll_d22_mMH0_ch01.bvh'], startOffsetSeconds: 2.55, seed: 4319 },
  { id: 'waack', label: 'Waack', file: 'gWA_sBM_cAll_d25_mWA0_ch02.bvh', url: urls['gWA_sBM_cAll_d25_mWA0_ch02.bvh'], startOffsetSeconds: 2.35, seed: 7523 },
  { id: 'krump', label: 'Krump', file: 'gKR_sBM_cAll_d28_mKR0_ch01.bvh', url: urls['gKR_sBM_cAll_d28_mKR0_ch01.bvh'], startOffsetSeconds: 1.80, seed: 8623 },
]
