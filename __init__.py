from __future__ import annotations

import os
import io
import mimetypes
import tempfile
import shutil
import subprocess
from urllib.parse import urlparse

import torch
import numpy as np
from PIL import Image

from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import folder_paths


# =====================
# Shared helpers
# =====================

def _get_tmpdir() -> str:
    """Return a container-local temp directory for safe concurrent use.

    Honors COMFYUI_S3_TMPDIR if set; otherwise uses OS-specific defaults
    and ensures a subdirectory exists.
    """
    env = os.environ.get("COMFYUI_S3_TMPDIR")
    if env:
        try:
            os.makedirs(env, exist_ok=True)
            return env
        except Exception:
            pass
    # Fallbacks
    if os.name == "nt":
        base = os.environ.get("TEMP") or os.environ.get("TMP") or "C:\\Windows\\Temp"
        path = os.path.join(base, "comfyui_s3")
    else:
        path = "/tmp/comfyui_s3"
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        path = tempfile.gettempdir()
    return path


def _find_ffmpeg() -> str:
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("ffmpeg not found. Install ffmpeg and ensure it is on PATH or set FFMPEG_PATH.")


def _download_url_to_temp(url: str) -> tuple[str, bytes, str | None]:
    """Download URL to a temp file and return (path, data_bytes, content_type)."""
    if not url:
        raise ValueError("URL is required.")

    tmpdir = _get_tmpdir()
    parsed = urlparse(url)
    base = os.path.basename(parsed.path) or "downloaded"
    fd, path = tempfile.mkstemp(prefix="comfyui_url_", suffix=os.path.splitext(base)[1], dir=tmpdir)
    os.close(fd)

    data = b""
    content_type = None

    # Try requests first
    try:
        import requests  # type: ignore

        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type")
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
        with open(path, "rb") as f:
            data = f.read()
    except ModuleNotFoundError:
        # Fallback to urllib
        import urllib.request

        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=60) as resp:
            headers = resp.info()
            content_type = headers.get("Content-Type")
            with open(path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        raise

    return path, data, content_type


def _normalize_content_type(ct: str | None, url: str) -> str:
    """Best-effort content-type normalization from header + URL extension."""
    if ct:
        ct = ct.split(";", 1)[0].strip().lower()
    if not ct or ct == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(url)
        if guessed:
            ct = guessed.split(";", 1)[0].strip().lower()
    return ct or ""


def _dummy_image() -> torch.Tensor:
    # [1,64,64,3] zeros
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _dummy_audio() -> dict:
    # Minimal [B,C,T] = [1,1,1] silent audio
    waveform = torch.zeros((1, 1, 1), dtype=torch.float32)
    return {"waveform": waveform, "sample_rate": 44100}


# =====================
# LoadFromUrl
# =====================

class LoadFromUrl(ComfyNodeABC):
    CATEGORY = "io/url"
    DESCRIPTION = (
        "Load media from an HTTP(S) URL (e.g. S3 signed URL). "
        "Outputs either an IMAGE batch (for images/videos) or AUDIO object depending on content."
    )

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "url": (IO.STRING, {"tooltip": "HTTP(S) URL to media (image / video / audio)."}),
            },
            "optional": {
                "force_fps": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 60,
                        "step": 1,
                        "tooltip": (
                            "For video: 0 = use source FPS; >0 = resample using this FPS when slicing into frames."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (IO.IMAGE, IO.AUDIO, IO.STRING)
    RETURN_NAMES = ("IMAGE", "AUDIO", "media_type")
    FUNCTION = "load"

    def load(self, url: str, force_fps: int = 0):
        path = None
        try:
            path, data, ct = _download_url_to_temp(url)
            ctype = _normalize_content_type(ct, url)

            # Image
            if ctype.startswith("image/"):
                image_batch = self._load_image_from_bytes(data)
                return (image_batch, _dummy_audio(), "image")

            # Audio
            if ctype.startswith("audio/"):
                audio_obj = self._load_audio_from_path(path)
                return (_dummy_image(), audio_obj, "audio")

            # Video
            if ctype.startswith("video/"):
                image_batch = self._load_video_frames(path, force_fps)
                return (image_batch, _dummy_audio(), "video")

            # Fallback: try image, then audio
            try:
                image_batch = self._load_image_from_bytes(data)
                return (image_batch, _dummy_audio(), "image")
            except Exception:
                pass

            try:
                audio_obj = self._load_audio_from_path(path)
                return (_dummy_image(), audio_obj, "audio")
            except Exception:
                pass

            raise RuntimeError(f"Could not determine media type for URL: {url}")

        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    # ---- internal helpers ----

    def _load_image_from_bytes(self, data: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]  # [1, H, W, 3]

    def _load_audio_from_path(self, path: str) -> dict:
        try:
            import torchaudio  # type: ignore
        except Exception as e:
            raise RuntimeError("torchaudio is required to load audio in LoadFromUrl. Install it with: pip install torchaudio") from e

        waveform, sample_rate = torchaudio.load(path)  # [C, T]
        return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}  # [B, C, T]

    def _load_video_frames(self, path: str, force_fps: int) -> torch.Tensor:
        ff = _find_ffmpeg()
        tmpdir = _get_tmpdir()
        frame_dir = tempfile.mkdtemp(prefix="comfyui_url_frames_", dir=tmpdir)

        cmd = [ff, "-hide_banner", "-loglevel", "error", "-i", path]
        if force_fps > 0:
            cmd += ["-vf", f"fps={force_fps}"]
        out_pattern = os.path.join(frame_dir, "frame_%05d.png")
        cmd += [out_pattern]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed to decode video: {e}") from e

        frame_files = sorted(
            f for f in os.listdir(frame_dir) if f.lower().endswith(".png")
        )
        if not frame_files:
            shutil.rmtree(frame_dir, ignore_errors=True)
            raise RuntimeError("No frames were extracted from video by ffmpeg.")

        frames = []
        for fname in frame_files:
            fpath = os.path.join(frame_dir, fname)
            img = Image.open(fpath).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            frames.append(arr)

        shutil.rmtree(frame_dir, ignore_errors=True)

        return torch.from_numpy(np.stack(frames, axis=0))  # [N, H, W, 3]


# =====================
# SaveImageToDisk (no UI)
# =====================

class SaveImageToDisk(ComfyNodeABC):
    CATEGORY = "image"
    DESCRIPTION = "Save IMAGE batch to ComfyUI output directory as PNG and return relative and absolute file paths."

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.compress_level = 1  # low compression (high quality / fast)

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "images": (IO.IMAGE, {"tooltip": "The image batch to save (single image = batch size 1)."}),
                "filename_prefix": (
                    IO.STRING,
                    {
                        "default": "ComfyUI",
                        "tooltip": (
                            "Prefix for the filename. Can include formatting such as "
                            "%date:yyyy-MM-dd% or %Empty Latent Image.width% like the built-in SaveImage node."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("relative_path", "absolute_path")
    FUNCTION = "save"

    def save(self, images, filename_prefix: str = "ComfyUI"):
        if images is None or images.size(0) == 0:
            raise ValueError("No images provided to SaveImageToDisk.")

        # Match built-in SaveImage behavior for save path generation
        h = images[0].shape[0]
        w = images[0].shape[1]
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, w, h
        )

        first_rel = ""
        first_abs = ""

        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            abs_path = os.path.join(full_output_folder, file)

            # PNG is lossless; "95% quality" interpreted as low compression, fast save, high fidelity
            img.save(abs_path, format="PNG", compress_level=self.compress_level, optimize=True)

            if batch_number == 0:
                if subfolder:
                    first_rel = os.path.join(subfolder, file)
                else:
                    first_rel = file
                first_abs = abs_path

            counter += 1

        return (first_rel, first_abs)


# =====================
# S3UploadPresigned
# =====================

class S3UploadPresigned(ComfyNodeABC):
    CATEGORY = "io/s3"
    DESCRIPTION = "Upload a local file to a provided presigned URL (no AWS credentials required)."

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "file_path": (IO.STRING, {"tooltip": "Absolute or relative path to the file on disk."}),
                "presigned_url": (IO.STRING, {"tooltip": "Presigned HTTP URL (typically PUT) to upload to."}),
            },
            "optional": {
                "content_type": (IO.STRING, {"default": "", "tooltip": "Optional Content-Type header; auto-detected when empty."}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("file_path", "presigned_url")
    FUNCTION = "upload"

    def upload(self, file_path: str, presigned_url: str, content_type: str = ""):
        if not file_path:
            raise ValueError("file_path is required.")
        if not presigned_url:
            raise ValueError("presigned_url is required.")

        path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Local file not found: {path}")

        headers = {}
        ct = (content_type or "").strip()
        if not ct:
            guessed, _ = mimetypes.guess_type(path)
            if guessed:
                ct = guessed
        if ct:
            headers["Content-Type"] = ct

        try:
            import requests  # type: ignore

            with open(path, "rb") as fh:
                resp = requests.put(presigned_url, data=fh, headers=headers, timeout=120)
            if resp.status_code >= 400:
                raise RuntimeError(f"Presigned upload failed with status {resp.status_code}: {resp.text[:200]}")
        except ModuleNotFoundError:
            with open(path, "rb") as fh:
                data = fh.read()
            try:
                import urllib.request

                req = urllib.request.Request(presigned_url, data=data, headers=headers, method="PUT")
                with urllib.request.urlopen(req, timeout=120):
                    pass
            except Exception as e:
                raise RuntimeError(f"Presigned upload via urllib failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Presigned upload failed: {e}") from e

        return (path, presigned_url)


# =====================
# ComfyUI registration
# =====================

NODE_CLASS_MAPPINGS = {
    "LoadFromUrl": LoadFromUrl,
    "SaveImageToDisk": SaveImageToDisk,
    "S3UploadPresigned": S3UploadPresigned,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFromUrl": "URL: Load Media",
    "SaveImageToDisk": "Image: Save To Disk (No UI)",
    "S3UploadPresigned": "S3: Upload Presigned",
}
