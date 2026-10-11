import { GeneratedMotionLibrary } from './GeneratedMotionLibrary'
import { JudgeDemoLab } from './JudgeDemoLab'
import { MotionCta } from './MotionCta'
import { HeroArena } from './HeroArena'

export function MotionStudio() {
  const judgeQuery = new URLSearchParams(window.location.search).get('judge')
  const judgeMode = judgeQuery === '1' || (judgeQuery !== '0' && import.meta.env.VITE_AURA_JUDGE_MODE === 'true')
  if (judgeMode) return (
    <main className="fx-page">
      <HeroArena />
      <section className="fx-hero">
        <h1 className="fx-h1">Human-guided robot data.<br />Try it without a GPU.</h1>
        <p className="fx-hero__lede">This static mode lets judges inspect motions, collect comparisons, and train Aura's small preference model in-browser. Live mode adds NVIDIA Kimodo generation, constraint screening, downstream experiments, and Solana curation payouts.</p>
      </section>
      <JudgeDemoLab />
      <MotionCta judge />
    </main>
  )
  return (
    <main className="fx-page">
      <HeroArena live />
      <div className="fx-section">
        <GeneratedMotionLibrary />
      </div>
      <MotionCta />
    </main>
  )
}
