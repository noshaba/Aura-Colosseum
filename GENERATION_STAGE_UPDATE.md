# Generation stage update

This build unifies Aura's generated-motion viewer with the visual language of the retargeted AIST++ G1 reference.

- Generated G1 playback now uses the same ivory/white + antique-gold material palette as the retargeted dance viewer.
- While a Text2Motion Aura generation request is running, the main viewer switches to a looping AIST++ retarget dance with end-effector trails enabled.
- The Generate button changes to an indeterminate loading state.
- The loading dance is explicitly labelled as a reference/loading animation, not the generated result.
- When the generation request returns, the newly saved G1 motion replaces the loading animation and autoplays as before.
