import { MotionStrip } from './MotionStrip'

type Props = { judge?: boolean }

export function MotionCta({ judge = false }: Props) {
  return (
    <section className="fx-cta">
      <div className="fx-cta__header">
        <h2 className="fx-h1">Your choice becomes data</h2>
        <div className="fx-cta__info">
          <span className="fx-cta__pill">Compare, learn, rank, reward</span>
          <a className="fx-btn fx-btn--primary" href={judge ? '?judge=0' : '?judge=1'}>{judge ? 'Open live workspace' : 'Try judge mode'}</a>
        </div>
      </div>
      <MotionStrip />
    </section>
  )
}
