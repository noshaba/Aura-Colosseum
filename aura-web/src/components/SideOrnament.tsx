type Props = {
  side: 'left' | 'right'
}

export function SideOrnament({ side }: Props) {
  return (
    <div className={`side-ornament-image side-ornament-image-${side}`} aria-hidden="true">
      <img
        src={side === 'left' ? '/generated-ornaments/side-left.png' : '/generated-ornaments/side-right.png'}
        alt=""
      />
    </div>
  )
}
