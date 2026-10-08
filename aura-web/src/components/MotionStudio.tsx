import { StudyHeader } from './StudyHeader'
import { GeneratedMotionLibrary } from './GeneratedMotionLibrary'
import { JudgeDemoLab } from './JudgeDemoLab'
import { AuraGeneratorStage } from './AuraGeneratorStage'

export function MotionStudio() {
  const judgeQuery = new URLSearchParams(window.location.search).get('judge')
  const judgeMode = judgeQuery === '1' || (judgeQuery !== '0' && import.meta.env.VITE_AURA_JUDGE_MODE === 'true')
  if (judgeMode) return (
    <main className="studio-shell">
      <div className="studio-heading"><StudyHeader />
        <div className="studio-heading-grid"><div><div className="eyebrow">AURA / JUDGE MODE</div><h1>Human-guided robot data.<br /><em>Try it without a GPU.</em></h1></div><p>This static mode lets judges inspect motions, collect comparisons, and train Aura's small preference model in-browser. Live mode adds Text2Motion Aura generation, constraint screening, downstream experiments, and Solana curation payouts.</p></div>
      </div>
      <JudgeDemoLab />
      <section className="judge-architecture"><div><strong>LIVE PIPELINE</strong><span>Text2Motion Aura generation</span><b>→</b><span>Native constraints</span><b>→</b><span>Human preferences</span><b>→</b><span>Learned ranking</span><b>→</b><span>Downstream benchmark</span><b>→</b><span>SOL curation rewards</span></div><p>Judge Mode uses NVIDIA Text2Motion Aura's bundled G1 examples solely to make the interface accessible without model weights or CUDA. Submission claims should use your separately collected same-prompt experiment, not these samples.</p></section>
    </main>
  )
  return (
    <main className="studio-shell">
      <div className="studio-heading"><StudyHeader />
        <div className="studio-heading-grid">
          <div><div className="eyebrow">AURA / GENERATIVE MOVEMENT RESEARCH</div><h1>Make movement.<br /><em>Measure possibility.</em></h1></div>
          <p>Generate Unitree G1 motion from text, play each result immediately, compare candidates, screen task constraints, and learn from human preference—all inside one Aura workspace.</p>
        </div>
      </div>
      <AuraGeneratorStage />
      <GeneratedMotionLibrary />
    </main>
  )
}
