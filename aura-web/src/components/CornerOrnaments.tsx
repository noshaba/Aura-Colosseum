export function CornerOrnaments() {
  const src = '/generated-ornaments/corner-top-left.svg'

  return (
    <div className="corner-ornament-layer" aria-hidden="true">
      <div className="corner-ornament corner-ornament-tl"><img src={src} alt="" /></div>
      <div className="corner-ornament corner-ornament-tr"><img src={src} alt="" /></div>
      <div className="corner-ornament corner-ornament-bl"><img src={src} alt="" /></div>
      <div className="corner-ornament corner-ornament-br"><img src={src} alt="" /></div>
    </div>
  )
}
