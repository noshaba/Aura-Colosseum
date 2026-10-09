/**
 * Thick, screen-space lines for motion previews (three/examples/jsm/lines).
 *
 * `createStrokedSegments` draws N segments twice from one shared geometry: a
 * wide navy under-stroke, then a narrower eggshell core, so the lines read on
 * both the bright sage ground and the sunset sky. Positions are written in
 * place each frame (no per-frame setPositions / reallocation).
 */
import * as THREE from 'three'
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'
import { AURA_PALETTE } from './auraWorld'

export type StrokedSegments = {
  readonly group: THREE.Group
  /** segments * 6 floats: (ax, ay, az, bx, by, bz) per segment. Write, then call commit(). */
  readonly positions: Float32Array
  commit(): void
  /** CSS-pixel size of the canvas; widths are in CSS px. */
  setResolution(width: number, height: number): void
  dispose(): void
}

export function createStrokedSegments(segments: number, opts: {
  core?: THREE.ColorRepresentation
  under?: THREE.ColorRepresentation
  coreWidth?: number
  underWidth?: number
  layer?: number
} = {}): StrokedSegments {
  const geometry = new LineSegmentsGeometry()
  geometry.setPositions(new Float32Array(Math.max(1, segments) * 6))
  const buffer = (geometry.getAttribute('instanceStart') as THREE.InterleavedBufferAttribute).data
  const positions = buffer.array as Float32Array

  const make = (color: THREE.ColorRepresentation, linewidth: number) => new LineMaterial({
    color: new THREE.Color(color).getHex(),
    linewidth,
    worldUnits: false,
    depthTest: false, // under-strokes first, cores on top: a clean inked look at the joints
    depthWrite: false,
  })
  const underMat = make(opts.under ?? AURA_PALETTE.navy, opts.underWidth ?? 7)
  const coreMat = make(opts.core ?? AURA_PALETTE.eggshell, opts.coreWidth ?? 3.5)
  const under = new LineSegments2(geometry, underMat)
  const core = new LineSegments2(geometry, coreMat)
  under.renderOrder = 10
  core.renderOrder = 11
  const group = new THREE.Group()
  group.add(under, core)
  for (const o of [group, under, core]) {
    o.frustumCulled = false
    if (opts.layer !== undefined) o.layers.set(opts.layer)
  }

  return {
    group,
    positions,
    commit() { buffer.needsUpdate = true },
    setResolution(width: number, height: number) {
      underMat.resolution.set(width, height)
      coreMat.resolution.set(width, height)
    },
    dispose() {
      group.removeFromParent()
      geometry.dispose(); underMat.dispose(); coreMat.dispose()
    },
  }
}

/** A static thick polyline (e.g. a root path on the ground). Lives in the main scene so it is depth-tested and inked. */
export function createThickPath(points: THREE.Vector3[], color: THREE.ColorRepresentation, width = 3.5) {
  const geometry = new LineSegmentsGeometry()
  // Expand the polyline into segments (LineSegmentsGeometry takes pairs).
  const segs = new Float32Array(Math.max(0, points.length - 1) * 6)
  for (let i = 0; i < points.length - 1; i++) segs.set([points[i].x, points[i].y, points[i].z, points[i + 1].x, points[i + 1].y, points[i + 1].z], i * 6)
  geometry.setPositions(segs)
  const material = new LineMaterial({ color: new THREE.Color(color).getHex(), linewidth: width, worldUnits: false })
  const line = new LineSegments2(geometry, material)
  line.frustumCulled = false
  return {
    line,
    setResolution(w: number, h: number) { material.resolution.set(w, h) },
    dispose() { line.removeFromParent(); geometry.dispose(); material.dispose() },
  }
}
