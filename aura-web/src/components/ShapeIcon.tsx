import { useInView } from '../hooks/useInView'

export type ShapeIconKind = 'pair' | 'arch' | 'flag' | 'bowl' | 'grid' | 'arc'

type Props = { kind: ShapeIconKind; size?: 'sm' | 'md' | 'lg' }

/** Small geometric brand-palette icons with a looping micro-animation that only
 *  runs while the icon is in view (and never under prefers-reduced-motion). */
export function ShapeIcon({ kind, size = 'md' }: Props) {
  const [ref, inView] = useInView<HTMLSpanElement>(0.3)
  return (
    <span ref={ref} className={`fx-icon fx-icon--${kind} fx-icon--${size}${inView ? ' is-playing' : ''}`} aria-hidden="true">
      {kind === 'grid'
        ? Array.from({ length: 9 }, (_, i) => <i key={i} />)
        : <><i /><i /></>}
    </span>
  )
}
