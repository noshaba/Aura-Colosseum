// Minimal typings for the parts of @mkkellogg/gaussian-splats-3d the comparison page uses
// (the package ships no .d.ts). Dev-only reference renderer.
declare module '@mkkellogg/gaussian-splats-3d' {
  import type { Group } from 'three'

  export const SceneFormat: { Splat: number; KSplat: number; Ply: number; Spz: number }

  export type DropInViewerOptions = {
    sharedMemoryForWorkers?: boolean
    gpuAcceleratedSort?: boolean
    integerBasedSort?: boolean
    sphericalHarmonicsDegree?: number
    antialiased?: boolean
    dynamicScene?: boolean
  }

  export type SplatSceneOptions = {
    format?: number
    showLoadingUI?: boolean
    splatAlphaRemovalThreshold?: number
    position?: [number, number, number]
    /** Quaternion as [x, y, z, w]. */
    rotation?: [number, number, number, number]
    scale?: [number, number, number]
    progressiveLoad?: boolean
  }

  export class DropInViewer extends Group {
    constructor(options?: DropInViewerOptions)
    addSplatScene(path: string, options?: SplatSceneOptions): Promise<void>
    dispose(): Promise<void>
  }
}
