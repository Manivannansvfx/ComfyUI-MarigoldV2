# ComfyUI-MarigoldV2

A ComfyUI custom node for **Marigold V2 Depth - Log Stage 2**.

This project provides a simple ComfyUI interface for generating high-quality
monocular depth maps using the official Marigold V2 model.

> **Current scope:** Depth - Log Stage 2 only.

## Features

- Marigold V2 Depth - Log Stage 2
- ComfyUI IMAGE input
- Depth IMAGE output
- Depth MASK output
- Native or fixed inference resolution
- Adjustable width and height
- Seed control
- Optional depth inversion
- Percentile-based depth normalization
- Batch image support
- Native Linux backend
- Windows + WSL2 backend
- No model weights are redistributed by this repository

## Basic Workflow

```text
Load Image
    |
    v
Marigold V2 Depth - Log Stage 2
    |
    +----> depth_image
    |
    +----> depth_mask
    ```

    
## Upstream Project

Marigold V2 was developed by **Huawei Bayer Lab**.

Official repository:

https://github.com/huawei-bayerlab/marigold-v2

This repository is an independent ComfyUI integration and is not the official
Marigold V2 repository.

## Supported Model

This initial release supports:

**Depth - Log Stage 2**

This is the default depth checkpoint used by the official Marigold V2
inference implementation.

Other Marigold V2 modalities and checkpoints are intentionally not included
in this initial ComfyUI integration.

For the complete Marigold V2 project and model family, please visit the
official upstream repository.

## Requirements

The Marigold V2 runtime follows the requirements of the upstream project.

Typical requirements include:

- Linux
- Python 3.10
- NVIDIA CUDA GPU
- PyTorch
- Official Marigold V2 runtime
- Required Qwen model components
- Marigold V2 Depth - Log Stage 2 checkpoint

This custom node itself only adds lightweight Python dependencies to ComfyUI.

## Linux

On Linux, use:

`backend = native`

`repo_path` must point to a working installation of the official Marigold V2 repository.

Example:

`/home/user/marigold-v2`

If `python_executable` is left empty, the node uses the current Python executable.

A separate Marigold V2 Python environment can also be specified.

Example:

`/home/user/miniconda3/envs/marigold-v2/bin/python`

## Windows + WSL2

Windows users can run ComfyUI on Windows while running the official Marigold V2 environment inside WSL2.

Use:

`backend = wsl`

Example configuration:

- `repo_path`: `/home/user/marigold-v2`
- `python_executable`: `/home/user/miniconda3/envs/marigold-v2/bin/python`
- `wsl_distro`: `Ubuntu`

The exact paths depend on your WSL installation and Linux username.

## Resolution

The node provides two resolution modes:

### Native

Uses the source image resolution, adjusted by the upstream Marigold V2 pipeline where required.

### Fixed

Runs inference at the selected width and height.

Common examples:

- `512 x 512`
- `768 x 768`
- `1024 x 1024`

Higher resolutions require substantially more GPU VRAM.

## Depth Output

Marigold V2 produces affine-invariant depth.

The node converts the raw floating-point prediction into a normalized ComfyUI-compatible depth image and mask.

`low_percentile` and `high_percentile` control normalization.

`invert_depth` can reverse the normalized depth representation for workflows that expect near objects to appear brighter.

## Model Downloads

Model weights are **not included in this GitHub repository**.

Required Marigold V2 and Qwen assets must be obtained from their official upstream sources.

This keeps the ComfyUI integration lightweight and avoids redistributing third-party model files.

## Comfy Cloud

This project is designed with native Linux execution in mind so that it can be evaluated for managed environments such as Comfy Cloud.

Cloud support depends on the platform providing the required Marigold V2 runtime, model assets, Python dependencies, and compatible GPU environment.

The Windows WSL2 backend is not required for native Linux or Cloud execution.

## License

The original ComfyUI integration code in this repository is released under the **Apache License 2.0**.

Marigold V2 remains subject to its upstream license.

Qwen and other third-party components remain subject to their respective licenses and terms.

See `NOTICE` for additional attribution information.



## Official Marigold V2 Resources

For more information about the original Marigold V2 project:

- Official Repository: https://github.com/huawei-bayerlab/marigold-v2
- Paper: https://arxiv.org/abs/2609.08084
- Project Page: https://marigoldmonodepth.github.io/

## Acknowledgements

Thanks to the Marigold V2 authors and Huawei Bayer Lab for releasing the Marigold V2 project and model family.

## Disclaimer

This is an independent community integration.

It is not affiliated with, endorsed by, or an official release of Huawei Bayer Lab or the Marigold V2 authors.
