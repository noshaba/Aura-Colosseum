import { useEffect, useRef, type ReactNode } from 'react'
import { AuraModelScene, type AuraModelDims, type AuraModelLayerId, type AuraModelMotion, type AuraModelPop, type AuraModelSceneMode } from '../three/auraModelScene'

export type AuraModelVizLabel = { id: AuraModelLayerId; number: string; title: string; learnable: boolean; shape?: string; blurb?: string }

type Props = {
  mode: AuraModelSceneMode
  dims: AuraModelDims
  labels: AuraModelVizLabel[]
  active?: AuraModelLayerId
  playing?: boolean
  motion: AuraModelMotion | null
  /** Normalised reward 0..1 (rank percentile), null = neutral / untrained. */
  reward: number | null
  onSelect?: (id: AuraModelLayerId) => void
  className?: string
  /** Text alternative for the (aria-hidden) canvas. */
  description: string
  /** Orb pop: fires whenever `key` changes. */
  pop?: { key: number; kind: AuraModelPop } | null
  /** Codex celebration on a stage: fires whenever `key` changes. */
  celebrate?: { key: number; id: AuraModelLayerId } | null
  /** Hero: stage the camera zooms to (null = overview). */
  view?: AuraModelLayerId | null
  /** Fractions of the height kept clear at the top / bottom (full-screen hero text and controls). */
  safeArea?: { top: number; bottom: number }
  /** DOM UI over the canvas (HUD, prompts). `--orb-x` / `--orb-y` on the host anchor it to the reward orb. */
  children?: ReactNode
}

/**
 * Thin React shell around AuraModelScene: one renderer per mount, setters for
 * live props, full disposal on unmount (StrictMode-safe: mount -> dispose -> mount).
 */
export function AuraModelViz({ mode, dims, labels, active, playing = false, motion, reward, onSelect, className, description, pop, celebrate, view = null, safeArea, children }: Props) {
  const hostRef = useRef<HTMLDivElement>(null)
  const labelsRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<AuraModelScene | null>(null)
  const selectRef = useRef(onSelect)
  selectRef.current = onSelect
  const init = useRef({ dims, motion, reward, active, playing, view, safeArea })
  init.current = { dims, motion, reward, active, playing, view, safeArea }

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    let scene: AuraModelScene
    try {
      scene = new AuraModelScene(host, { mode, dims: init.current.dims, labels: labelsRef.current, onSelect: id => selectRef.current?.(id) })
    } catch {
      return // no WebGL: the text description and layer buttons still work
    }
    const s = init.current
    scene.setMotion(s.motion)
    scene.setReward(s.reward)
    if (s.active) scene.setActive(s.active)
    scene.setPlaying(s.playing)
    if (s.safeArea) scene.setSafeArea(s.safeArea.top, s.safeArea.bottom)
    scene.setView(s.view ?? null)
    sceneRef.current = scene
    return () => { scene.dispose(); if (sceneRef.current === scene) sceneRef.current = null }
  }, [mode])

  useEffect(() => { sceneRef.current?.setDims(dims) }, [dims])
  useEffect(() => { sceneRef.current?.setMotion(motion) }, [motion])
  useEffect(() => { sceneRef.current?.setReward(reward) }, [reward])
  useEffect(() => { if (active) sceneRef.current?.setActive(active) }, [active])
  useEffect(() => { sceneRef.current?.setPlaying(playing) }, [playing])
  useEffect(() => { sceneRef.current?.setView(view) }, [view])
  const safeTop = safeArea?.top ?? 0, safeBottom = safeArea?.bottom ?? 0
  useEffect(() => { sceneRef.current?.setSafeArea(safeTop, safeBottom) }, [safeTop, safeBottom])
  const popKey = pop?.key ?? 0, popKind = pop?.kind ?? 'neutral'
  useEffect(() => { if (popKey) sceneRef.current?.pop(popKind) }, [popKey, popKind])
  const celebrateKey = celebrate?.key ?? 0, celebrateId = celebrate?.id
  useEffect(() => { if (celebrateKey && celebrateId) sceneRef.current?.celebrate(celebrateId) }, [celebrateKey, celebrateId])
  const labelKey = labels.map(l => `${l.id}:${l.title}`).join('|')
  useEffect(() => { sceneRef.current?.refreshLabels() }, [labelKey])

  return (
    <div className={`aura-model-viz aura-model-viz--${mode} ${className ?? ''}`} ref={hostRef}>
      <p className="aura-model-viz__sr">{description}</p>
      <div className="aura-model-viz__labels" ref={labelsRef} aria-hidden="true">
        {labels.map(l => (
          <span key={l.id} data-layer={l.id} className={l.learnable ? 'is-learned' : undefined}>
            <b>{l.number}</b>{l.title}
          </span>
        ))}
      </div>
      {children}
    </div>
  )
}
