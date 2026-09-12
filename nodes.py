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
        native -> Native Linux / managed Linux environment
        wsl    -> Windows ComfyUI calling Marigold V2 inside WSL2
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
                        "default": "",
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
    # Native Linux / Cloud Marigold repository detection
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_native_repo_path(repo_path: str) -> Path:
        """
        Find the official Marigold V2 repository.

        Search priority:

        1. MARIGOLD_V2_PATH environment variable
        2. repo_path entered manually in the node
        3. runtime/marigold-v2 inside this custom node
        4. vendor/marigold-v2 inside this custom node
        5. ~/marigold-v2
        """

        candidates = []

        # --------------------------------------------------------------
        # 1. Environment variable
        # --------------------------------------------------------------

        env_path = os.environ.get(
            "MARIGOLD_V2_PATH",
            "",
        ).strip()

        if env_path:
            candidates.append(
                (
                    "MARIGOLD_V2_PATH",
                    Path(
                        os.path.expandvars(
                            os.path.expanduser(
                                env_path
                            )
                        )
                    ),
                )
            )

        # --------------------------------------------------------------
        # 2. Manual repo_path
        # --------------------------------------------------------------

        if repo_path.strip():
            candidates.append(
                (
                    "repo_path",
                    Path(
                        os.path.expandvars(
                            os.path.expanduser(
                                repo_path.strip()
                            )
                        )
                    ),
                )
            )

        # --------------------------------------------------------------
        # 3 / 4. Bundled locations
        # --------------------------------------------------------------

        node_dir = Path(__file__).resolve().parent

        candidates.append(
            (
                "runtime",
                node_dir
                / "runtime"
                / "marigold-v2",
            )
        )

        candidates.append(
            (
                "vendor",
                node_dir
                / "vendor"
                / "marigold-v2",
            )
        )

        # --------------------------------------------------------------
        # 5. Common Linux user location
        # --------------------------------------------------------------

        candidates.append(
            (
                "home",
                Path.home()
                / "marigold-v2",
            )
        )

        # --------------------------------------------------------------
        # Validate candidates
        # --------------------------------------------------------------

        for source, candidate in candidates:

            candidate = candidate.resolve()

            infer_script = (
                candidate
                / "scripts"
                / "infer.py"
            )

            if infer_script.is_file():

                print(
                    "[ComfyUI-MarigoldV2] "
                    f"Using Marigold V2 from {source}: "
                    f"{candidate}"
                )

                return candidate

        searched = "\n".join(
            f"- {source}: {candidate}"
            for source, candidate in candidates
        )

        raise FileNotFoundError(
            "Could not locate the official Marigold V2 repository.\n\n"
            "Searched:\n"
            f"{searched}\n\n"
            "To configure Marigold V2:\n"
            "1. Set MARIGOLD_V2_PATH, or\n"
            "2. Enter the repository path in repo_path.\n\n"
            "Official repository:\n"
            "https://github.com/huawei-bayerlab/marigold-v2"
        )

    # ------------------------------------------------------------------
    # Image conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_pil(
        image_tensor: torch.Tensor,
    ) -> Image.Image:

        image_tensor = (
            image_tensor
            .detach()
            .cpu()
            .float()
        )

        image_tensor = torch.clamp(
            image_tensor,
            0.0,
            1.0,
        )

        array = (
            image_tensor.numpy()
            * 255.0
        ).round().astype(
            np.uint8
        )

        if array.ndim != 3:
            raise RuntimeError(
                "Expected IMAGE tensor with shape "
                "[H,W,C], received "
                f"{array.shape}"
            )

        if array.shape[-1] == 1:

            array = np.repeat(
                array,
                3,
                axis=-1,
            )

        elif array.shape[-1] == 4:

            array = array[..., :3]

        elif array.shape[-1] != 3:

            raise RuntimeError(
                "Unsupported channel count: "
                f"{array.shape[-1]}"
            )

        return Image.fromarray(
            array,
            mode="RGB",
        )

    # ------------------------------------------------------------------
    # Depth loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_depth_npy(
        path: Path,
    ) -> np.ndarray:

        depth = np.load(
            str(path)
        )

        depth = np.asarray(
            depth,
            dtype=np.float32,
        )

        depth = np.squeeze(
            depth
        )

        if depth.ndim == 3:

            if depth.shape[0] == 1:
                depth = depth[0]

            elif depth.shape[-1] == 1:
                depth = depth[..., 0]

        if depth.ndim != 2:
            raise RuntimeError(
                "Unexpected Marigold depth shape "
                f"{depth.shape} in file:\n"
                f"{path}"
            )

        return depth

    # ------------------------------------------------------------------
    # Depth normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_depth(
        depth: np.ndarray,
        low_percentile: float,
        high_percentile: float,
        invert: bool,
    ) -> np.ndarray:

        depth = depth.astype(
            np.float32,
            copy=False,
        )

        finite_mask = np.isfinite(
            depth
        )

        if not np.any(
            finite_mask
        ):
            raise RuntimeError(
                "Marigold returned a depth map "
                "containing no finite values."
            )

        valid_values = depth[
            finite_mask
        ]

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

        normalized[
            ~finite_mask
        ] = 0.0

        if invert:
            normalized = (
                1.0 - normalized
            )

        return normalized.astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Windows -> WSL path conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _windows_path_to_wsl(
        windows_path: Path,
        distro: str,
    ) -> str:

        command = [
            "wsl.exe",
        ]

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

        converted = (
            result.stdout.strip()
        )

        if not converted:
            raise RuntimeError(
                "Failed converting Windows path "
                "to WSL path:\n"
                f"{windows_path}"
            )

        return converted

    # ------------------------------------------------------------------
    # Find Marigold output
    # ------------------------------------------------------------------

    @staticmethod
    def _find_depth_file(
        output_dir: Path,
        input_stem: str,
    ) -> Path:

        candidates = list(
            output_dir.rglob(
                "*.npy"
            )
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
            "Marigold inference finished but "
            "no depth .npy file was found for:\n"
            f"{input_stem}\n\n"
            "Output directory:\n"
            f"{output_dir}"
        )

    # ------------------------------------------------------------------
    # Validate normalization settings
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_percentiles(
        low_percentile: float,
        high_percentile: float,
    ):

        if (
            low_percentile
            >= high_percentile
        ):
            raise ValueError(
                "low_percentile must be smaller "
                "than high_percentile."
            )

    # ------------------------------------------------------------------
    # Native Linux command
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

        repo = (
            self._resolve_native_repo_path(
                repo_path
            )
        )

        infer_script = (
            repo
            / "scripts"
            / "infer.py"
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

        if (
            resolution_mode
            == "fixed"
        ):

            command += [
                "--width",
                str(width),

                "--height",
                str(height),
            ]

        return (
            command,
            repo,
        )

    # ------------------------------------------------------------------
    # Windows + WSL command
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
                "WSL backend is intended "
                "for Windows ComfyUI."
            )

        repo_path = (
            repo_path.strip()
        )

        if not repo_path:
            raise RuntimeError(
                "For the WSL backend, repo_path "
                "must point to the Marigold V2 "
                "repository inside WSL."
            )

        python_executable = (
            python_executable.strip()
            or "python"
        )

        infer_script = (
            repo_path.rstrip("/")
            + "/scripts/infer.py"
        )

        wsl_input = (
            self._windows_path_to_wsl(
                input_dir,
                wsl_distro,
            )
        )

        wsl_output = (
            self._windows_path_to_wsl(
                output_dir,
                wsl_distro,
            )
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

        if (
            resolution_mode
            == "fixed"
        ):

            command += [
                "--width",
                str(width),

                "--height",
                str(height),
            ]

        return command

    # ------------------------------------------------------------------
    # Main execution
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
                "Input image must be "
                "a ComfyUI IMAGE tensor."
            )

        if image.ndim != 4:
            raise RuntimeError(
                "Expected ComfyUI IMAGE shape "
                "[B,H,W,C], received "
                f"{tuple(image.shape)}"
            )

        batch_size = (
            image.shape[0]
        )

        if batch_size < 1:
            raise RuntimeError(
                "Input IMAGE batch is empty."
            )

        temp_root = Path(
            tempfile.mkdtemp(
                prefix=(
                    "comfyui_marigold_v2_"
                )
            )
        )

        input_dir = (
            temp_root
            / "input"
        )

        output_dir = (
            temp_root
            / "output"
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
            # Save ComfyUI input images
            # ----------------------------------------------------------

            for index in range(
                batch_size
            ):

                stem = (
                    f"marigold_input_{index:06d}"
                )

                input_stems.append(
                    stem
                )

                image_path = (
                    input_dir
                    / f"{stem}.png"
                )

                pil_image = (
                    self._tensor_to_pil(
                        image[index]
                    )
                )

                pil_image.save(
                    image_path,
                    format="PNG",
                )

            # ----------------------------------------------------------
            # Build backend command
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

                working_directory = (
                    str(repo)
                )

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
                    "Unsupported backend: "
                    f"{backend}"
                )

            # ----------------------------------------------------------
            # Run official Marigold V2 inference
            # ----------------------------------------------------------

            print()

            print(
                "[ComfyUI-MarigoldV2] "
                "Starting Marigold V2 "
                "Depth - Log Stage 2"
            )

            print(
                "[ComfyUI-MarigoldV2] "
                f"Backend: {backend}"
            )

            print(
                "[ComfyUI-MarigoldV2] "
                f"Images: {batch_size}"
            )

            if (
                resolution_mode
                == "native"
            ):

                print(
                    "[ComfyUI-MarigoldV2] "
                    "Resolution: native"
                )

            else:

                print(
                    "[ComfyUI-MarigoldV2] "
                    f"Resolution: "
                    f"{width}x{height}"
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
                    f"Exit code: "
                    f"{process.returncode}\n\n"
                    f"{process.stdout}"
                )

            # ----------------------------------------------------------
            # Read depth predictions
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
            # Validate batch dimensions
            # ----------------------------------------------------------

            first_shape = (
                depth_images[0].shape
            )

            for item in depth_images:

                if (
                    item.shape
                    != first_shape
                ):
                    raise RuntimeError(
                        "Marigold returned different "
                        "output resolutions within "
                        "the same batch. "
                        "Use fixed resolution for "
                        "batched images with different "
                        "source dimensions."
                    )

            depth_image_batch = (
                torch.stack(
                    depth_images,
                    dim=0,
                )
            )

            depth_mask_batch = (
                torch.stack(
                    depth_masks,
                    dim=0,
                )
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
            # Clean temporary files
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
    "MarigoldV2Depth":
        MarigoldV2Depth,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "MarigoldV2Depth":
        "Marigold V2 Depth - Log Stage 2",
}
