import { useEffect, useState } from 'react'
import { AIST_REFERENCE_MOTIONS, STUDY_CLIP_SECONDS } from '../aistReferenceMotions'
import { DEFAULT_VIEW_POSE, XBotScene } from './XBotScene'
import type { ViewPose } from './XBotScene'

/**
 * Presentational loading state shown while Aura generates a candidate batch.
 * It reuses the bundled AIST++ retarget demo so the stage remains alive while
 * the local generation request is running. The dance itself is not a result.
 */
export function GeneratingDanceStage({ candidateCount = 2 }: { candidateCount?: number }) {
  const motion = AIST_REFERENCE_MOTIONS[3] ?? AIST_REFERENCE_MOTIONS[0]
  const [progress, setProgress] = useState(0)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [viewPose, setViewPose] = useState<ViewPose>({ ...DEFAULT_VIEW_POSE })

  useEffect(() => {
    let raf = 0
    const started = performance.now()
    let previous = started
    const tick = (now: number) => {
      const dt = Math.min((now - previous) / 1000, 0.1)
      previous = now
      setProgress(value => (value + (dt / STUDY_CLIP_SECONDS) * 100) % 100)
      setElapsedSeconds(Math.floor((now - started) / 1000))
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  return (
    <div className="aura-generating-dance" aria-label="Aura generation in progress">
      <XBotScene
        embodiment="g1"
        side="A"
        quality="reference"
        motionFile={motion.file}
        motionUrl={motion.url}
        degradationSeed={motion.seed}
        startOffsetSeconds={motion.startOffsetSeconds}
        paused={false}
        playhead={progress}
        autoRotate
        showTrails
        showLandmarks={false}
        resetViewSignal={0}
        viewPose={viewPose}
        onViewPoseChange={setViewPose}
      />
      <div className="aura-generating-overlay">
        <div className="aura-generating-orbit" aria-hidden="true"><span /></div>
        <div>
          <strong>AURA IS GENERATING</strong>
          <span>Creating {candidateCount} G1 candidates · {elapsedSeconds}s</span>
        </div>
      </div>
      <div className="aura-generating-note">Loading animation · AIST++ reference motion with live end-effector trails</div>
    </div>
  )
}
