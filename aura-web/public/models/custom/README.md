# Custom robot meshes

Drop your own robot model here. Files in this folder are served by Vite at
`/models/custom/<file>` (e.g. `/models/custom/my-robot.glb`).

> Status: `blossom-fairy.rigged.web.glb` is the character for every motion viewer:
> `blossom-fairy.web.glb` plus 15 finger bones per hand (curled procedurally) and
> wing bones (`wing_L`, `wing_L_tip`, `wing_R`, `wing_R_tip` under Spine2) that flap
> on the skeleton. It was produced by a one-off rig pass over the web copy (the
> build script is not in the repo yet).
> `blossom-fairy.web.glb` is
> a 6 MB copy of the 26.8 MB source `aura-web/models-src/Blossom Fairy Running.glb`
> (kept outside `public/` so it isn't shipped): the unused
> metallic-roughness map is dropped and the colour/normal maps are WebP
> (`EXT_texture_webp`). Regenerate it from the source after editing the model. It is a Mixamo rig (`mixamorig:*` bones, not the G1 skeleton below),
> so it is retargeted from the G1 joints at runtime; its `Running` clip is not
> used. See
> `src/three/fairyRig.ts` (`ROBOT_MODEL = 'g1'` there brings the Unitree G1 STL
> robot back).

## Format: glTF binary (`.glb`)

One self-contained `.glb` per robot (geometry + materials + textures embedded).
Export from Blender with **File → Export → glTF 2.0**, format *glTF Binary*,
**+Y up**, units in **metres**, *Apply Modifiers* on.

Keep it light: under ~10 MB, under ~100k triangles total. Use Draco or
meshopt compression only if asked; plain GLB is the safest start.

## The one hard rule: it must fit the G1 skeleton

Every motion in Aura is recorded on the Unitree G1's 34-joint skeleton (world
positions + rotations per joint). Your mesh is moved by those joints, so it has
to be built on the same skeleton: same proportions and joint positions, standing
in the G1 rest pose (upright, arms down, facing +Z). A mesh with different
proportions will stretch apart or float.

Pick ONE of these two layouts:

### A. Rigid parts (easiest, matches how the G1 is drawn today)
Split the mesh into separate objects, one per body segment, and name each
object exactly after the joint that moves it (see the list below). Each part
follows its joint rigidly. Not every joint needs a part; e.g. a single
`pelvis_skel` part plus limbs is fine.

### B. Skinned mesh (one continuous body)
One mesh with an armature. Bone names must match the joint names below exactly,
with the same parent/child hierarchy, and the mesh skinned (weight-painted) to
them.

## Joint names (G1Skeleton34, parent in brackets)

```
pelvis_skel            (root)
left_hip_pitch_skel    (pelvis_skel)
left_hip_roll_skel     (left_hip_pitch_skel)
left_hip_yaw_skel      (left_hip_roll_skel)
left_knee_skel         (left_hip_yaw_skel)
left_ankle_pitch_skel  (left_knee_skel)
left_ankle_roll_skel   (left_ankle_pitch_skel)
left_toe_base          (left_ankle_roll_skel)
right_hip_pitch_skel   (pelvis_skel)
right_hip_roll_skel    (right_hip_pitch_skel)
right_hip_yaw_skel     (right_hip_roll_skel)
right_knee_skel        (right_hip_yaw_skel)
right_ankle_pitch_skel (right_knee_skel)
right_ankle_roll_skel  (right_ankle_pitch_skel)
right_toe_base         (right_ankle_roll_skel)
waist_yaw_skel         (pelvis_skel)
waist_roll_skel        (waist_yaw_skel)
waist_pitch_skel       (waist_roll_skel)
left_shoulder_pitch_skel  (waist_pitch_skel)
left_shoulder_roll_skel   (left_shoulder_pitch_skel)
left_shoulder_yaw_skel    (left_shoulder_roll_skel)
left_elbow_skel           (left_shoulder_yaw_skel)
left_wrist_roll_skel      (left_elbow_skel)
left_wrist_pitch_skel     (left_wrist_roll_skel)
left_wrist_yaw_skel       (left_wrist_pitch_skel)
left_hand_roll_skel       (left_wrist_yaw_skel)
right_shoulder_pitch_skel (waist_pitch_skel)
right_shoulder_roll_skel  (right_shoulder_pitch_skel)
right_shoulder_yaw_skel   (right_shoulder_roll_skel)
right_elbow_skel          (right_shoulder_yaw_skel)
right_wrist_roll_skel     (right_elbow_skel)
right_wrist_pitch_skel    (right_wrist_roll_skel)
right_wrist_yaw_skel      (right_wrist_pitch_skel)
right_hand_roll_skel      (right_wrist_yaw_skel)
```

Source of truth: `text2motion-aura/text2motion_aura/skeleton/definitions.py`
(`G1Skeleton34`). Reference geometry for proportions: `../g1-native/meshes/*.STL`
placed by `../g1-native/g1.xml` — importing those into Blender as a guide is
the quickest way to match the G1 rest pose.

## Licensing
Only drop meshes you have the rights to use and publish.
