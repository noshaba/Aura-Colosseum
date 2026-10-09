/**
 * Shared look of Aura's fairy flare (loader -> hero split sweep -> robot flares):
 * colour stops hot white -> sunset -> gold -> terra, premultiplied / partly
 * additive blending, and one point-sprite fragment shader that draws every
 * particle kind (dust, four-point glint, trail blob, head) so each flare is a
 * single Points draw.
 */
import * as THREE from 'three'

export const FLARE_HEX = { hot: '#fffae6', sunset: '#f2cc8f', gold: '#e0a85f', terra: '#e07a5f' } as const

/** The four stops as colours (linear working space), indexed 0 hot, 1 sunset, 2 gold, 3 terra: sparkle tints. Shared, read-only. */
export const FLARE_COLORS: readonly THREE.Color[] = [FLARE_HEX.hot, FLARE_HEX.sunset, FLARE_HEX.gold, FLARE_HEX.terra].map(h => new THREE.Color(h))

/**
 * Flag only the live prefix [0, count) of point-pool attributes for upload. Nothing is
 * uploaded when count is 0 (the draw range is 0 too, so stale data is never drawn).
 */
export function uploadLiveRange(attrs: readonly THREE.BufferAttribute[], count: number) {
  if (count <= 0) return
  for (const a of attrs) {
    a.clearUpdateRanges()
    a.addUpdateRange(0, count * a.itemSize)
    a.needsUpdate = true
  }
}

/** Fresh uniform objects for the four colour stops (linear working space). */
export const FLARE_COLS = () => ({
  cHot: { value: new THREE.Color(FLARE_HEX.hot) }, cSun: { value: new THREE.Color(FLARE_HEX.sunset) },
  cGold: { value: new THREE.Color(FLARE_HEX.gold) }, cEdge: { value: new THREE.Color(FLARE_HEX.terra) },
})

export const FLARE_STOPS = `uniform vec3 cHot;
uniform vec3 cSun;
uniform vec3 cGold;
uniform vec3 cEdge;
// The loader's radial gradient: r = 0 centre .. 1 edge -> (colour, alpha), smooth (no banding steps).
vec4 flareStops( float r, float s1, float s2 ) {
  vec3 c = r < s1 ? mix( cHot, cSun, r / s1 ) : r < s2 ? mix( cSun, cGold, ( r - s1 ) / ( s2 - s1 ) ) : mix( cGold, cEdge, ( r - s2 ) / ( 1.0 - s2 ) );
  float a = r < s2 ? mix( 1.0, 0.55, smoothstep( 0.0, s2, r ) ) : mix( 0.55, 0.0, smoothstep( s2, 1.0, r ) );
  return vec4( c, a );
}
`

// Premultiplied output. `over` < 1 makes the blend partly additive (a glow that
// adds light); 1 is a plain premultiplied "over", which keeps gold readable on the pale backdrop.
export const premultiply = (over: number) => `
  gl_FragColor.rgb *= gl_FragColor.a;
  gl_FragColor.a *= ${over.toFixed(2)};`

export const GLOW_BLEND = {
  transparent: true, depthTest: false, depthWrite: false,
  blending: THREE.CustomBlending,
  blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
  blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
} as const

/** Particle kinds understood by FLARE_POINT_FRAG (passed as the `vStar` varying). */
export const FLARE_KIND = { dust: 0, glint: 1, trail: 2, head: 3 } as const

/**
 * Point-sprite fragment for all flare particles. Needs FLARE_COLS() uniforms and
 * varyings vCol (dust/glint tint), vA (alpha), vStar (kind, see FLARE_KIND).
 */
export const flarePointFrag = (over: number) => `${FLARE_STOPS}
varying vec3 vCol;
varying float vA;
varying float vStar;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  vec2 q = abs( p );
  float d = length( p );
  vec4 c;
  if ( vStar > 2.5 ) {
    // head: soft halo + four-point twinkle (the loader's quadratic-curve star)
    c = flareStops( min( d, 1.0 ), 0.18, 0.5 );
    c.a *= 1.0 - smoothstep( 0.92, 1.0, d );
    vec2 sq = q / 0.38;
    float s = smoothstep( 1.0, 0.82, sqrt( sq.x ) + sqrt( sq.y ) );
    c = vec4( mix( c.rgb, cHot, s ), max( c.a, s * 0.95 ) );
  } else if ( vStar > 1.5 ) {
    // trail blob: the ribbon's cross-section as a round, soft stamp
    c = flareStops( min( d, 1.0 ), 0.28, 0.6 );
    c.a *= 1.0 - smoothstep( 0.85, 1.0, d );
  } else if ( vStar > 0.5 ) {
    // glint: four-point star with soft cross rays and a small halo
    float glint = smoothstep( 1.0, 0.72, sqrt( q.x ) + sqrt( q.y ) );
    float rays = exp( -q.x * 12.0 ) * exp( -q.y * 2.4 ) + exp( -q.y * 12.0 ) * exp( -q.x * 2.4 );
    float a = clamp( max( glint, rays * 0.65 ) + exp( -d * d * 7.0 ) * 0.4, 0.0, 1.0 ) * smoothstep( 1.0, 0.85, d );
    c = vec4( mix( vCol, cHot, smoothstep( 0.5, 0.0, d ) ), a );
  } else {
    // fine dust
    float a = exp( -d * d * 4.5 ) * smoothstep( 1.0, 0.7, d );
    c = vec4( mix( vCol, cHot, smoothstep( 0.5, 0.0, d ) * 0.5 ), a );
  }
  c.a *= vA;
  if ( c.a < 0.003 ) discard;
  gl_FragColor = c;
  #include <colorspace_fragment>
  ${premultiply(over)}
}`
