import type { CharacterIconId } from '../three/characters'

/**
 * Line icons for the character switch and picker. Single 1.6 stroke in currentColor so
 * they take the button's colour; decorative only (the button carries the name).
 */
const PATHS: Record<CharacterIconId, JSX.Element> = {
  robot: (
    <>
      <path d="M10 6.5V3.8" />
      <circle cx="10" cy="2.9" r="1" />
      <rect x="4" y="6.5" width="12" height="9" rx="2.4" />
      <path d="M2.4 10v2.6M17.6 10v2.6M8 13.2h4" />
      <circle cx="7.6" cy="10.4" r=".9" fill="currentColor" stroke="none" />
      <circle cx="12.4" cy="10.4" r=".9" fill="currentColor" stroke="none" />
    </>
  ),
  fairy: (
    <>
      <circle cx="10" cy="4.6" r="1.9" />
      <path d="M10 7.2 7.6 14.4h4.8L10 7.2Z" />
      <path d="M9 14.4v3M11 14.4v3" />
      <path d="M9.2 9.2C6.6 5.6 2.6 5.9 3.3 8.7c.4 1.4 2 2 3.3 2.1-1.6.8-2.4 2.4-1.4 3.4 1.3 1.3 3.2-.6 4.1-2.6" />
      <path d="M10.8 9.2c2.6-3.6 6.6-3.3 5.9-.5-.4 1.4-2 2-3.3 2.1 1.6.8 2.4 2.4 1.4 3.4-1.3 1.3-3.2-.6-4.1-2.6" />
    </>
  ),
}

export function CharacterIcon({ id, className, size = 18 }: { id: CharacterIconId; className?: string; size?: number }) {
  return (
    <svg className={className} viewBox="0 0 20 20" width={size} height={size} aria-hidden="true" focusable="false"
      fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      {PATHS[id]}
    </svg>
  )
}
