import { useInView } from '../hooks/useInView'

/** A walk cycle drawn as five flat geometric figures: Aura's own "motion as a
 *  sequence of poses" illustration in the brand palette. Limbs swing only while
 *  the strip is in view. */
const POSES = [
  { c: 'blue', arm: -28, leg: 22 },
  { c: 'yellow', arm: -10, leg: 8 },
  { c: 'orange', arm: 14, leg: -12 },
  { c: 'green', arm: 30, leg: -24 },
  { c: 'brown', arm: 6, leg: 4 },
]

export function MotionStrip() {
  const [ref, inView] = useInView<HTMLDivElement>(0.2)
  return (
    <div ref={ref} className={`fx-strip${inView ? ' is-playing' : ''}`} aria-hidden="true">
      <svg className="fx-strip__path" viewBox="0 0 1000 120" preserveAspectRatio="none"><path d="M10 100 C 200 20, 380 20, 500 70 S 820 120, 990 30" /></svg>
      {POSES.map((p, i) => (
        <div key={i} className={`fx-figure fx-figure--${p.c}`} style={{ ['--arm' as string]: `${p.arm}deg`, ['--leg' as string]: `${p.leg}deg`, ['--i' as string]: i }}>
          <span className="fx-figure__head" />
          <span className="fx-figure__arm is-back" />
          <span className="fx-figure__torso" />
          <span className="fx-figure__arm is-front" />
          <span className="fx-figure__hip" />
          <span className="fx-figure__leg is-back" />
          <span className="fx-figure__leg is-front" />
        </div>
      ))}
    </div>
  )
}
