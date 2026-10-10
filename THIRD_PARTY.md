# Third-party software and assets

This file is a practical attribution summary, not legal advice. The authoritative license files in each dependency control.

## NVIDIA Kimodo

`text2motion-aura/` is the local compatibility path for a vendored and modified copy of NVIDIA Kimodo, which is Aura's upstream generation engine. The repository includes its `LICENSE` and `ATTRIBUTIONS.MD`. The Kimodo repository code is distributed under the license included there. Kimodo model weights are separately listed by NVIDIA under the NVIDIA Open Model License; review the model's Hugging Face/license page before commercial deployment.

Aura modifications include automatic motion-library publishing, G1-native preview output, G1 default styling/branding integration, and UI theme changes.

## Unitree G1 assets

Kimodo's `ATTRIBUTIONS.MD` attributes Unitree MuJoCo assets under the BSD 3-Clause License. Keep the original attribution/license text with redistributed assets. Aura's gold material treatment is only a visual modification and is not an official Unitree appearance.


## AIST++ reference motion

`aura-web/src/assets/motions/` contains a small bundled set of AIST++ BVH dance annotations used only for the reference retargeting viewer. The original Aura source identified the AIST++ annotations as © Google LLC under CC BY 4.0 and cited Li, Yang, Ross & Kanazawa (ICCV 2021), with the original AIST Dance Video Database by Tsuchida et al. (ISMIR 2019). Preserve the relevant dataset attribution and license information when redistributing these files.

The viewer retargets those human motions onto the bundled Unitree G1 visualization. The AIST++ reference motions are not Kimodo generations and should not be presented as Aura downstream-benchmark evidence.

## Judge Mode samples

`aura-web/public/demo/` contains derived JSON skeletal previews of the G1 example motions already bundled inside Kimodo. They are included only for interface demonstration and are explicitly not Aura benchmark evidence.

## Aura branding / generated ornaments

Brand files and decorative assets under `aura-web/public/brand`, `generated-ornaments`, and `ornaments` come from the supplied Aura project bundle. Confirm ownership/permission before public commercial redistribution if any were sourced externally.

## Solana Web3.js

The Aura frontend imports `@solana/web3.js` as a pinned npm dependency and Vite bundles it into the application. This removes the previous runtime dependency on an external unpkg script. Retain the dependency's upstream license notices when redistributing builds/source.

## Naming

Aura presents the upstream generator as **NVIDIA Kimodo** in user-facing documentation and UI. The `text2motion-aura` / `text2motion_aura` names remain only as internal compatibility paths/namespaces from an earlier integration rename. They do not imply that Aura authored or owns the Kimodo generation model.
