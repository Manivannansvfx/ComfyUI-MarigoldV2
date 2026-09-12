from __future__ import annotations

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# -------------------------------------------------------------------------
# ComfyUI-MarigoldV2
# Marigold V2 Depth - Log Stage 2
#
# Upstream:
# https://github.com/huawei-bayerlab/marigold-v2
#
# This node is an independent ComfyUI integration.
# It does not redistribute Marigold V2 model weights.
# -------------------------------------------------------------------------


class MarigoldV2Depth:
    """
    ComfyUI wrapper for the official Marigold V2 inference script.

    Supported model:
        Depth - Log Stage 2

    Backends:
        native  -> Linux / Comfy Cloud / native Linux ComfyUI
        wsl     -> Windows ComfyUI calling Marigold V2 inside WSL2
    """

    CATEGORY = "Marigold V2"
    FUNCTION = "generate_depth"

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("depth_image", "depth_mask")

    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

                "backend": (
                    ["native", "wsl"],
                    {
                        "default": "native",
                    },
                ),

                "repo_path": (
                    "STRING",
                    {
                        "default": "/home/user/marigold-v2",
                        "multiline": False,
                    },
                ),

                "python_executable": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),

                "resolution_mode": (
                    ["native", "fixed"],
                    {
                        "default": "fixed",
                    },
                ),

                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 16,
                    },
                ),

                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 16,
                    },
                ),

                "seed": (
                    "INT",
                    {
                        "default": 2025,
                        "min": 0,
                        "max": 0x7FFFFFFF,
                    },
                ),

                "invert_depth": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),

                "low_percentile": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 49.0,
                        "step": 0.1,
                    },
                ),

                "high_percentile": (
                    "FLOAT",
                    {
                        "default": 99.5,
                        "min": 51.0,
                        "max": 100.0,
                        "step": 0.1,
                    },
                ),
            },

            "optional": {
                "wsl_distro": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
            },
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        """
        Convert ComfyUI IMAGE tensor [H,W,C] in range 0..1 to PIL RGB.
        """

        image_tensor = image_tensor.detach().cpu().float()

        image_tensor = torch.clamp(
            image_tensor,
            0.0,
            1.0,
        )

        array = (
            image_tensor.numpy() * 255.0
        ).round().astype(np.uint8)

        if array.ndim != 3:
            raise RuntimeError(
                f"Expected IMAGE tensor with 3 dimensions [H,W,C], "
                f"received shape {array.shape}"
            )

        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)

        elif array.shape[-1] == 4:
            array = array[..., :3]

        elif array.shape[-1] != 3:
            raise RuntimeError(
                f"Unsupported channel count: {array.shape[-1]}"
            )

        return Image.fromarray(array, mode="RGB")

    # ------------------------------------------------------------------

    @staticmethod
    def _load_depth_npy(path: Path) -> np.ndarray:
        """
        Load Marigold V2 .npy depth prediction and convert to HxW.
        """

        depth = np.load(str(path))

        depth = np.asarray(
            depth,
            dtype=np.float32,
        )

        depth = np.squeeze(depth)

        if depth.ndim == 3:

            if depth.shape[0] == 1:
                depth = depth[0]

            elif depth.shape[-1] == 1:
                depth = depth[..., 0]

        if depth.ndim != 2:
            raise RuntimeError(
                f"Unexpected Marigold depth shape "
                f"{depth.shape} in file:\n{path}"
            )

        return depth

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_depth(
        depth: np.ndarray,
        low_percentile: float,
        high_percentile: float,
        invert: bool,
    ) -> np.ndarray:
        """
        Percentile-normalize affine-invariant Marigold depth into 0..1.
        """

        depth = depth.astype(
            np.float32,
            copy=False,
        )

        finite_mask = np.isfinite(depth)

        if not np.any(finite_mask):
            raise RuntimeError(
                "Marigold returned a depth map containing no finite values."
            )

        valid_values = depth[finite_mask]

        low = float(
            np.percentile(
                valid_values,
                low_percentile,
            )
        )

        high = float(
            np.percentile(
                valid_values,
                high_percentile,
            )
        )

        if high <= low:
            high = low + 1e-6

        normalized = (
            depth - low
        ) / (
            high - low
        )

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        normalized[~finite_mask] = 0.0

        if invert:
            normalized = 1.0 - normalized

        return normalized.astype(
            np.float32
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _windows_path_to_wsl(
        windows_path: Path,
        distro: str,
    ) -> str:

        command = ["wsl.exe"]

        if distro.strip():
            command += [
                "-d",
                distro.strip(),
            ]

        command += [
            "--",
            "wslpath",
            "-a",
            str(windows_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        converted = result.stdout.strip()

        if not converted:
            raise RuntimeError(
                f"Failed converting Windows path to WSL path:\n"
                f"{windows_path}"
            )

        return converted

    # ------------------------------------------------------------------

    @staticmethod
    def _find_depth_file(
        output_dir: Path,
        input_stem: str,
    ) -> Path:

        prediction_dir = (
            output_dir / "predictions_npy"
        )

        search_root = (
            prediction_dir
            if prediction_dir.exists()
            else output_dir
        )

        candidates = list(
            search_root.rglob("*.npy")
        )

        exact_matches = [
            path
            for path in candidates
            if path.stem == input_stem
        ]

        if exact_matches:
            return exact_matches[0]

        contains_matches = [
            path
            for path in candidates
            if input_stem in path.stem
        ]

        if contains_matches:
            return contains_matches[0]

        raise RuntimeError(
            "Marigold inference finished but no depth "
            f".npy file was found for:\n{input_stem}\n\n"
            f"Output directory:\n{output_dir}"
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _validate_percentiles(
        low_percentile: float,
        high_percentile: float,
    ):

        if low_percentile >= high_percentile:
            raise ValueError(
                "low_percentile must be smaller than "
                "high_percentile."
            )

    # ------------------------------------------------------------------

    def _build_native_command(
        self,
        repo_path: str,
        python_executable: str,
        input_dir: Path,
        output_dir: Path,
        resolution_mode: str,
        width: int,
        height: int,
        seed: int,
    ):

        repo = Path(
            os.path.expanduser(repo_path)
        )

        infer_script = (
            repo / "scripts" / "infer.py"
        )

        if not infer_script.is_file():
            raise FileNotFoundError(
                "Could not find official Marigold V2 "
                f"inference script:\n{infer_script}\n\n"
                "repo_path must point to the official "
                "huawei-bayerlab/marigold-v2 repository."
            )

        python_bin = (
            python_executable.strip()
            if python_executable.strip()
            else sys.executable
        )

        command = [
            python_bin,
            str(infer_script),

            "--modality",
            "depth",

            "--image_dir",
            str(input_dir),

            "--output_dir",
            str(output_dir),

            "--seed",
            str(seed),
        ]

        if resolution_mode == "fixed":
            command += [
                "--width",
                str(width),

                "--height",
                str(height),
            ]

        return command, repo

    # ------------------------------------------------------------------

    def _build_wsl_command(
        self,
        repo_path: str,
        python_executable: str,
        input_dir: Path,
        output_dir: Path,
        resolution_mode: str,
        width: int,
        height: int,
        seed: int,
        wsl_distro: str,
    ):

        if os.name != "nt":
            raise RuntimeError(
                "WSL backend is intended for Windows ComfyUI."
            )

        repo_path = repo_path.strip()

        if not repo_path:
            raise RuntimeError(
                "repo_path cannot be empty."
            )

        python_executable = (
            python_executable.strip()
            or "python"
        )

        infer_script = (
            repo_path.rstrip("/")
            + "/scripts/infer.py"
        )

        wsl_input = self._windows_path_to_wsl(
            input_dir,
            wsl_distro,
        )

        wsl_output = self._windows_path_to_wsl(
            output_dir,
            wsl_distro,
        )

        command = [
            "wsl.exe",
        ]

        if wsl_distro.strip():
            command += [
                "-d",
                wsl_distro.strip(),
            ]

        command += [
            "--",
            python_executable,

            infer_script,

            "--modality",
            "depth",

            "--image_dir",
            wsl_input,

            "--output_dir",
            wsl_output,

            "--seed",
            str(seed),
        ]

        if resolution_mode == "fixed":
            command += [
                "--width",
                str(width),

                "--height",
                str(height),
            ]

        return command

    # ------------------------------------------------------------------

    def generate_depth(
        self,
        image,
        backend,
        repo_path,
        python_executable,
        resolution_mode,
        width,
        height,
        seed,
        invert_depth,
        low_percentile,
        high_percentile,
        wsl_distro="",
    ):

        self._validate_percentiles(
            low_percentile,
            high_percentile,
        )

        if not isinstance(
            image,
            torch.Tensor,
        ):
            raise TypeError(
                "Input image must be a ComfyUI IMAGE tensor."
            )

        if image.ndim != 4:
            raise RuntimeError(
                "Expected ComfyUI IMAGE shape "
                f"[B,H,W,C], received {tuple(image.shape)}"
            )

        batch_size = image.shape[0]

        if batch_size < 1:
            raise RuntimeError(
                "Input IMAGE batch is empty."
            )

        temp_root = Path(
            tempfile.mkdtemp(
                prefix="comfyui_marigold_v2_"
            )
        )

        input_dir = (
            temp_root / "input"
        )

        output_dir = (
            temp_root / "output"
        )

        input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        input_stems = []

        try:

            # ----------------------------------------------------------
            # Save ComfyUI images for official Marigold inference
            # ----------------------------------------------------------

            for index in range(batch_size):

                stem = (
                    f"marigold_input_{index:06d}"
                )

                input_stems.append(stem)

                image_path = (
                    input_dir / f"{stem}.png"
                )

                pil_image = self._tensor_to_pil(
                    image[index]
                )

                pil_image.save(
                    image_path,
                    format="PNG",
                )

            # ----------------------------------------------------------
            # Build command
            # ----------------------------------------------------------

            if backend == "native":

                command, repo = (
                    self._build_native_command(
                        repo_path=repo_path,
                        python_executable=python_executable,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        resolution_mode=resolution_mode,
                        width=width,
                        height=height,
                        seed=seed,
                    )
                )

                working_directory = str(repo)

            elif backend == "wsl":

                command = (
                    self._build_wsl_command(
                        repo_path=repo_path,
                        python_executable=python_executable,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        resolution_mode=resolution_mode,
                        width=width,
                        height=height,
                        seed=seed,
                        wsl_distro=wsl_distro,
                    )
                )

                working_directory = None

            else:

                raise RuntimeError(
                    f"Unsupported backend: {backend}"
                )

            # ----------------------------------------------------------
            # Run official Marigold V2 inference
            # ----------------------------------------------------------

            print()
            print(
                "[ComfyUI-MarigoldV2] "
                "Starting Marigold V2 Depth - Log Stage 2"
            )

            print(
                f"[ComfyUI-MarigoldV2] "
                f"Backend: {backend}"
            )

            print(
                f"[ComfyUI-MarigoldV2] "
                f"Images: {batch_size}"
            )

            if resolution_mode == "native":

                print(
                    "[ComfyUI-MarigoldV2] "
                    "Resolution: native"
                )

            else:

                print(
                    "[ComfyUI-MarigoldV2] "
                    f"Resolution: {width}x{height}"
                )

            process = subprocess.run(
                command,
                cwd=working_directory,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            if process.stdout:
                print(
                    process.stdout
                )

            if process.returncode != 0:

                raise RuntimeError(
                    "Marigold V2 inference failed.\n\n"
                    f"Exit code: {process.returncode}\n\n"
                    f"{process.stdout}"
                )

            # ----------------------------------------------------------
            # Read Marigold predictions
            # ----------------------------------------------------------

            depth_images = []
            depth_masks = []

            for stem in input_stems:

                depth_path = (
                    self._find_depth_file(
                        output_dir,
                        stem,
                    )
                )

                raw_depth = (
                    self._load_depth_npy(
                        depth_path
                    )
                )

                normalized_depth = (
                    self._normalize_depth(
                        raw_depth,
                        low_percentile,
                        high_percentile,
                        invert_depth,
                    )
                )

                depth_tensor = (
                    torch.from_numpy(
                        normalized_depth
                    ).float()
                )

                mask_tensor = (
                    depth_tensor.clone()
                )

                image_tensor = (
                    depth_tensor
                    .unsqueeze(-1)
                    .repeat(
                        1,
                        1,
                        3,
                    )
                )

                depth_images.append(
                    image_tensor
                )

                depth_masks.append(
                    mask_tensor
                )

            # ----------------------------------------------------------
            # Ensure batch output dimensions match
            # ----------------------------------------------------------

            first_shape = (
                depth_images[0].shape
            )

            for item in depth_images:

                if item.shape != first_shape:
                    raise RuntimeError(
                        "Marigold returned different output "
                        "resolutions within the same batch. "
                        "Use fixed resolution for batched images "
                        "with different source dimensions."
                    )

            depth_image_batch = torch.stack(
                depth_images,
                dim=0,
            )

            depth_mask_batch = torch.stack(
                depth_masks,
                dim=0,
            )

            print(
                "[ComfyUI-MarigoldV2] "
                "Depth generation complete."
            )

            return (
                depth_image_batch,
                depth_mask_batch,
            )

        finally:

            # ----------------------------------------------------------
            # Remove temporary input/output files
            # ----------------------------------------------------------

            try:
                shutil.rmtree(
                    temp_root,
                    ignore_errors=True,
                )

            except Exception:
                pass


# -------------------------------------------------------------------------
# ComfyUI registration
# -------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "MarigoldV2Depth": MarigoldV2Depth,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "MarigoldV2Depth":
        "Marigold V2 Depth - Log Stage 2",
}
