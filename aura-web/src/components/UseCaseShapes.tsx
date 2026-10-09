import { useState } from 'react'

type Case = { title: string; body: string }
type Props = { cases: [Case, Case, Case, Case] }

/**
 * Four oversized flat shapes (rectangle, circle, half-round, quarter-round).
 * Desktop hover: the hovered shape holds still, earlier siblings slide -150px,
 * later ones +150px (0.6s expo-out), its description fades in (0.3s) and its
 * accent shape slides into place (0.5s).
 */
export function UseCaseShapes({ cases }: Props) {
  const [active, setActive] = useState<number | null>(null)
  return (
    <div className="fx-cases" onMouseLeave={() => setActive(null)}>
      {cases.map((c, i) => {
        const shift = active === null || active === i ? 0 : i < active ? -150 : 150
        return (
          <div
            key={c.title}
            className={`fx-case fx-case--${i + 1}${active === i ? ' is-active' : ''}`}
            style={{ ['--fx-shift' as string]: `${shift}px` }}
            onMouseEnter={() => setActive(i)}
            onFocus={() => setActive(i)}
            tabIndex={0}
          >
            <span className="fx-case__accent" aria-hidden="true" />
            <div className="fx-case__shape">
              <div className="fx-case__text">
                <h3>{c.title}</h3>
                <p>{c.body}</p>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
