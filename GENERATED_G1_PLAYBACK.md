# Generated G1 playback

Aura now plays Text2Motion Aura-generated G1 motions on the full Unitree G1 mesh in the Generated Motion Library.

## Flow

1. Generate a G1 motion in the embedded Text2Motion Aura editor.
2. Text2Motion Aura saves the native NPZ to `aura-motion-library/`.
3. Aura's library server emits a `g1-joints-v2` browser payload containing the generated world-space joint positions and global joint rotation matrices.
4. The Aura frontend polls the local library. When a new generation appears, it automatically selects it and starts playback on the full gold/black G1 mesh.
5. Older generated motions remain selectable. Choosing a second motion under **Compare two generated motions** opens a second robot player beside the selected motion.

Existing `g1-joints-v1` previews are upgraded automatically from their saved NPZ files the next time `/motions` is requested, so old generated motions do not need to be regenerated.

The AIST++ dance viewer remains a separate reference section and does not replace generated-motion playback.
