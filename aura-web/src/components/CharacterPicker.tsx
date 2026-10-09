import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'
import { useCharacter } from '../hooks/useCharacter'
import { CHARACTERS, type CharacterDef } from '../three/characters'
import './character-picker.css'
import { CharacterIcon } from './CharacterIcon'

/** Up to this many models the picker is a segmented control; beyond it, a listbox menu. */
const SEGMENTED_MAX = 3

type Props = {
  /** False while the picker sits in a hidden panel: takes it out of the tab order. */
  tabbable?: boolean
  className?: string
}

const base = () => import.meta.env.BASE_URL || '/'

function Thumb({ c }: { c: CharacterDef }) {
  if (!c.thumbnail) return <CharacterIcon id={c.icon} className="char-picker__icon" />
  return <img className="char-picker__thumb" src={`${base()}${c.thumbnail}`} alt="" aria-hidden="true" />
}

/** Picks the body that performs every motion (registry: three/characters.ts). */
export function CharacterPicker({ tabbable = true, className }: Props) {
  const { character, setCharacter } = useCharacter()
  const cls = ['char-picker', className].filter(Boolean).join(' ')
  return CHARACTERS.length <= SEGMENTED_MAX
    ? <Segmented className={cls} current={character} onPick={setCharacter} tabbable={tabbable} />
    : <Menu className={cls} current={character} onPick={setCharacter} tabbable={tabbable} />
}

type ViewProps = { className: string; current: CharacterDef; onPick: (id: string) => void; tabbable: boolean }

/** radiogroup: one tab stop, arrows move and select (selection follows focus). */
function Segmented({ className, current, onPick, tabbable }: ViewProps) {
  const labelId = useId()
  const refs = useRef<(HTMLButtonElement | null)[]>([])
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = CHARACTERS.findIndex(c => c.id === current.id)
    const n = CHARACTERS.length
    let next = -1
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (i + 1) % n
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (i - 1 + n) % n
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = n - 1
    if (next < 0) return
    e.preventDefault()
    onPick(CHARACTERS[next].id)
    refs.current[next]?.focus()
  }
  return (
    <div className={`${className} char-picker--segmented`}>
      <span className="char-picker__caption" id={labelId}>Character</span>
      <div className="char-picker__track" role="radiogroup" aria-labelledby={labelId} onKeyDown={onKey}>
        {CHARACTERS.map((c, i) => {
          const on = c.id === current.id
          return (
            <button
              key={c.id}
              ref={el => { refs.current[i] = el }}
              type="button"
              role="radio"
              aria-checked={on}
              tabIndex={tabbable && on ? 0 : -1}
              title={c.credit}
              className={on ? 'char-picker__opt is-on' : 'char-picker__opt'}
              onClick={() => onPick(c.id)}
            >
              <Thumb c={c} />
              <span>{c.label}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

/** Disclosure button + listbox, for longer rosters. */
function Menu({ className, current, onPick, tabbable }: ViewProps) {
  const id = useId()
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const trigger = useRef<HTMLButtonElement | null>(null)
  const list = useRef<HTMLUListElement | null>(null)

  useEffect(() => { if (open) list.current?.focus() }, [open])
  useEffect(() => { if (!tabbable) setOpen(false) }, [tabbable])

  const show = () => { setActive(Math.max(0, CHARACTERS.findIndex(c => c.id === current.id))); setOpen(true) }
  const close = (refocus = true) => { setOpen(false); if (refocus) trigger.current?.focus() }
  const pick = (i: number) => { onPick(CHARACTERS[i].id); close() }

  const onTriggerKey = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); show() }
  }
  const onListKey = (e: KeyboardEvent<HTMLUListElement>) => {
    const n = CHARACTERS.length
    if (e.key === 'ArrowDown') setActive(a => Math.min(n - 1, a + 1))
    else if (e.key === 'ArrowUp') setActive(a => Math.max(0, a - 1))
    else if (e.key === 'Home') setActive(0)
    else if (e.key === 'End') setActive(n - 1)
    else if (e.key === 'Enter' || e.key === ' ') pick(active)
    else if (e.key === 'Escape') { e.stopPropagation(); close() } // keep the nav menu open
    else if (e.key === 'Tab') { setOpen(false); return }
    else return
    e.preventDefault()
  }

  return (
    <div className={`${className} char-picker--menu${open ? ' is-open' : ''}`}>
      <button
        ref={trigger}
        type="button"
        className="char-picker__trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={`${id}-list`}
        tabIndex={tabbable ? 0 : -1}
        onClick={() => (open ? close() : show())}
        onKeyDown={onTriggerKey}
      >
        <span className="char-picker__caption">Character</span>
        <Thumb c={current} />
        <span className="char-picker__value">{current.label}</span>
        <span className="char-picker__chev" aria-hidden="true" />
      </button>
      {open && (
        <ul
          ref={list}
          id={`${id}-list`}
          className="char-picker__list"
          role="listbox"
          aria-label="Character"
          tabIndex={-1}
          aria-activedescendant={`${id}-opt-${active}`}
          onKeyDown={onListKey}
          onBlur={e => { if (!e.currentTarget.parentElement?.contains(e.relatedTarget as Node)) setOpen(false) }}
        >
          {CHARACTERS.map((c, i) => (
            <li
              key={c.id}
              id={`${id}-opt-${i}`}
              role="option"
              aria-selected={c.id === current.id}
              title={c.credit}
              className={['char-picker__item', i === active && 'is-active', c.id === current.id && 'is-on'].filter(Boolean).join(' ')}
              onPointerEnter={() => setActive(i)}
              onClick={() => pick(i)}
            >
              <Thumb c={c} />
              <span>{c.label}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
