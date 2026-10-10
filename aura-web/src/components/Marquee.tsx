import { useEffect, useRef } from 'react'
import { ShapeIcon } from './ShapeIcon'
import type { ShapeIconKind } from './ShapeIcon'

type Item = { label: string; icon: ShapeIconKind }
type Props = { label?: string; items: Item[] }

const SPEED_PX_PER_S = 60 // reference auto-scroll: 1px per frame at 60fps

/** Infinite, linear auto-scrolling row of pills (duplicated track, translate -50%). */
export function Marquee({ label, items }: Props) {
  const track = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const el = track.current
    if (!el) return
    const update = () => el.style.setProperty('--fx-marquee-duration', `${Math.max(10, el.scrollWidth / 2 / SPEED_PX_PER_S)}s`)
    update()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [items])
  const loop = [...items, ...items]
  return (
    <div className="fx-marquee">
      {label && <div className="fx-marquee__label">{label}</div>}
      <div className="fx-marquee__viewport">
        <div className="fx-marquee__fade is-left" aria-hidden="true" />
        <div className="fx-marquee__track" ref={track}>
          {loop.map((item, i) => (
            <div className="fx-marquee__pill" key={i} aria-hidden={i >= items.length}>
              <ShapeIcon kind={item.icon} size="sm" />
              <span>{item.label}</span>
            </div>
          ))}
        </div>
        <div className="fx-marquee__fade is-right" aria-hidden="true" />
      </div>
    </div>
  )
}

export const PIPELINE_ITEMS: Item[] = [
  { label: 'NVIDIA Kimodo generation', icon: 'pair' },
  { label: 'Native constraints', icon: 'arch' },
  { label: 'Human preferences', icon: 'bowl' },
  { label: 'Learned ranking', icon: 'flag' },
  { label: 'Downstream benchmark', icon: 'grid' },
  { label: 'SOL curation rewards', icon: 'arc' },
  { label: 'Human evaluation layer', icon: 'pair' },
  { label: 'Solana devnet', icon: 'arch' },
]
