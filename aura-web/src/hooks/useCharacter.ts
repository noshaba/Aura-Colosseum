import { useSyncExternalStore } from 'react'
import { getCharacter, getCharacterId, setCharacterId, subscribeCharacter, type CharacterDef } from '../three/characters'

/** The selected character (live across components and tabs) and its setter. */
export function useCharacter(): { character: CharacterDef; setCharacter: (id: string) => void } {
  const id = useSyncExternalStore(subscribeCharacter, getCharacterId, getCharacterId)
  return { character: getCharacter(id), setCharacter: setCharacterId }
}
