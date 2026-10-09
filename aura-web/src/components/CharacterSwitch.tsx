import { useEffect, useRef, useState } from 'react'
import { useCharacter } from '../hooks/useCharacter'
import { CHARACTERS } from '../three/characters'
import { CharacterIcon } from './CharacterIcon'
import './character-picker.css'

/**
 * Always-visible nav button that cycles the character (three/characters.ts) to the next
 * one in registry order, so it works for any roster size. Same store as the dropdown
 * picker (CharacterPicker), so both stay in sync; every viewer rebuilds its body live.
 * Enter / Space come from the native <button>.
 */
export function CharacterSwitch() {
  const { character, setCharacter } = useCharacter()
  const i = Math.max(0, CHARACTERS.findIndex(c => c.id === character.id))
  const next = CHARACTERS[(i + 1) % CHARACTERS.length]
  // Live region text after a change.
  const [said, setSaid] = useState('')
  const first = useRef(true)

  useEffect(() => {
    if (first.current) { first.current = false; return } // no announcement on mount
    setSaid(`Character: ${character.label}`)
  }, [character.id, character.label])

  if (CHARACTERS.length < 2) return null
  return (
    <div className="char-switch">
      <button
        type="button"
        className="char-switch__btn"
        aria-label={`Switch character, current: ${character.label}`}
        title={`Character: ${character.label}. Click for ${next.label}.`}
        onClick={() => setCharacter(next.id)}
      >
        {/* keyed so the icon pops in on each change */}
        <CharacterIcon key={character.id} id={character.icon} className="char-switch__icon" />
        <span className="char-switch__label" aria-hidden="true">{character.label}</span>
        <span className="char-switch__tag" aria-hidden="true">{character.tag}</span>
      </button>
      <span className="char-switch__live" role="status" aria-live="polite">{said}</span>
    </div>
  )
}
