# Starter rating update

The live Motion Studio has been simplified around the core interaction.

## Visible live flow

1. Aura generation controls and the main Motion-vs-Motion player.
2. Generated Motion Library.

The standalone AIST++ retargeted-dance reference section and all sections that followed it (Constraint Lab, Discovery Lab, Downstream Benchmark, Curation Market, workflow footer) are no longer rendered in the live Motion Studio.

## Immediate first rating

Before a user generates anything, the main comparison player is preloaded with six bundled AIST++ dance clips retargeted to the Unitree G1. The clips run as a small tournament so a visitor can make a Motion A vs Motion B choice immediately.

Starter AIST++ ratings are saved separately in browser localStorage (`aura:aist-starter-preferences:v1`) and are intentionally not mixed into Aura's same-prompt generated-motion preference training data. Once the user generates an Aura batch, the player automatically switches to the newly generated candidates and those votes are saved to Aura's real generated-motion preference dataset.
