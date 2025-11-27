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

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import json

# Constants matching Weaver's frame-based system
TOTAL_FRAMES = 77

def pil2tensor(image):
    """Convert PIL Image to tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(tensor):
    """Convert tensor to PIL Image"""
    return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def parse_color(color):
    """Parse color string to RGB tuple"""
    if isinstance(color, str):
        if color.startswith('#'):
            # Handle hex colors (from Weaver)
            color = color.lstrip('#')
            return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        elif ',' in color:
            return tuple(int(c.strip()) for c in color.split(','))
        else:
            from PIL import ImageColor
            try:
                return ImageColor.getrgb(color)
            except:
                return (255, 255, 255)
    return color

def apply_easing(t, easing_type='linear'):
    """
    Apply easing function to parameter t (0.0 to 1.0)
    Matches Weaver's EASING_FUNCTIONS
    """
    if easing_type == 'ease-in':
        # Quadratic ease in
        return t * t
    elif easing_type == 'ease-out':
        # Quadratic ease out
        return t * (2 - t)
    elif easing_type == 'ease-in-out':
        # Quadratic ease in-out
        if t < 0.5:
            return 2 * t * t
        else:
            return -1 + (4 - 2 * t) * t
    elif easing_type == 'ease-in-cubic':
        # Cubic ease in (strong)
        return t * t * t
    elif easing_type == 'ease-out-cubic':
        # Cubic ease out (strong)
        t -= 1
        return t * t * t + 1
    elif easing_type == 'ease-in-out-cubic':
        # Cubic ease in-out (strong)
        if t < 0.5:
            return 4 * t * t * t
        else:
            return (t - 1) * (2 * t - 2) * (2 * t - 2) + 1
    else:
        # Linear (default)
        return t

class WeaverAnimator:
    """
    Creates animated shapes that follow user-drawn paths from Weaver.
    
    Consumes Weaver's JSON format with:
    - Frame-based timing (startFrame/endFrame, 0-76 for 77 frames)
    - Easing functions (linear, ease-in, ease-out, etc.)
    - Static anchor points (isSinglePoint: true)
    - Canvas dimensions for coordinate scaling
    
    Outputs WAN ATI-compatible coordinate strings with proper 77-point resampling.
    """

    RETURN_TYPES = ("IMAGE", "MASK", "STRING",)
    RETURN_NAMES = ("image", "mask", "coordinates",)
    FUNCTION = "animate_paths"
    CATEGORY = "io"
    DESCRIPTION = """
Creates animated shapes that follow user-drawn paths from Weaver.
Open the path editor to draw trajectories on a reference image, then shapes will follow these paths over time.
Outputs WAN ATI-compatible coordinate strings with proper 77-point resampling for stable video generation.
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "frame_count": ("INT", {"default": TOTAL_FRAMES, "min": 1, "max": 500, "step": 1}),
                "shape": ([
                    'circle',
                    'square',
                    'triangle',
                    'hexagon',
                    'star',
                ], {"default": 'circle'}),
                "shape_size": ("INT", {"default": 20, "min": 2, "max": 500, "step": 1}),
                "shape_color": ("STRING", {"default": 'white'}),
                "bg_color": ("STRING", {"default": 'black'}),
            },
            "optional": {
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "trail_length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation_speed": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "border_width": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
                "border_color": ("STRING", {"default": 'white'}),
                "paths_data": ("STRING", {"default": '{"paths": [], "canvas": {"width": 512, "height": 512}, "total_frames": 77}', "multiline": True}),
                "visibility_mode": (["pop", "static"], {"default": "pop"}),
            }
        }

    def draw_shape(self, draw, shape, center_x, center_y, size, rotation, fill_color, border_width=0, border_color='white'):
        """Draw a shape at the specified location"""
        half_size = size / 2

        if shape == 'circle':
            bbox = [center_x - half_size, center_y - half_size,
                   center_x + half_size, center_y + half_size]
            if border_width > 0:
                draw.ellipse(bbox, fill=fill_color, outline=border_color, width=border_width)
            else:
                draw.ellipse(bbox, fill=fill_color)

        elif shape == 'square':
            bbox = [center_x - half_size, center_y - half_size,
                   center_x + half_size, center_y + half_size]
            if border_width > 0:
                draw.rectangle(bbox, fill=fill_color, outline=border_color, width=border_width)
            else:
                draw.rectangle(bbox, fill=fill_color)

        elif shape == 'triangle':
            points = [
                (center_x, center_y - half_size),
                (center_x - half_size, center_y + half_size),
                (center_x + half_size, center_y + half_size),
            ]
            if rotation != 0:
                points = self.rotate_points(points, center_x, center_y, rotation)
            if border_width > 0:
                draw.polygon(points, fill=fill_color, outline=border_color, width=border_width)
            else:
                draw.polygon(points, fill=fill_color)

        elif shape == 'hexagon':
            points = []
            for i in range(6):
                angle = math.radians(60 * i + rotation)
                x = center_x + half_size * math.cos(angle)
                y = center_y + half_size * math.sin(angle)
                points.append((x, y))
            if border_width > 0:
                draw.polygon(points, fill=fill_color, outline=border_color, width=border_width)
            else:
                draw.polygon(points, fill=fill_color)

        elif shape == 'star':
            points = []
            for i in range(10):
                angle = math.radians(36 * i + rotation)
                r = half_size if i % 2 == 0 else half_size * 0.4
                x = center_x + r * math.cos(angle - math.pi / 2)
                y = center_y + r * math.sin(angle - math.pi / 2)
                points.append((x, y))
            if border_width > 0:
                draw.polygon(points, fill=fill_color, outline=border_color, width=border_width)
            else:
                draw.polygon(points, fill=fill_color)

    def rotate_points(self, points, cx, cy, angle):
        """Rotate points around a center"""
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        rotated = []
        for x, y in points:
            x -= cx
            y -= cy
            new_x = x * cos_a - y * sin_a + cx
            new_y = x * sin_a + y * cos_a + cy
            rotated.append((new_x, new_y))
        return rotated

    def resample_path_uniform(self, points, num_samples=TOTAL_FRAMES):
        """
        Resample path to exactly num_samples points with even arc-length spacing.
        Default is 77 points to match Weaver's TOTAL_FRAMES for WAN ATI compatibility.

        Args:
            points: List of {x, y} dicts representing the path
            num_samples: Number of points to resample to (default 77 for Weaver/WAN ATI)

        Returns:
            List of {x, y} dicts with exactly num_samples points evenly distributed along the arc
        """
        if len(points) == 0:
            return []

        # Support static single points (anchors)
        if len(points) == 1:
            # Single point - repeat for all samples (creates static anchor)
            return [{'x': points[0]['x'], 'y': points[0]['y']} for _ in range(num_samples)]

        # Calculate cumulative arc lengths along the path
        cumulative_lengths = [0.0]
        for i in range(len(points) - 1):
            dx = points[i + 1]['x'] - points[i]['x']
            dy = points[i + 1]['y'] - points[i]['y']
            length = math.sqrt(dx * dx + dy * dy)
            cumulative_lengths.append(cumulative_lengths[-1] + length)

        total_length = cumulative_lengths[-1]

        # Handle zero-length path (all points are the same)
        if total_length == 0:
            return [{'x': points[0]['x'], 'y': points[0]['y']} for _ in range(num_samples)]

        # Resample at even intervals along the arc
        resampled = []
        for i in range(num_samples):
            # Calculate target distance along path
            if num_samples == 1:
                target_length = 0
            else:
                target_length = (i / (num_samples - 1)) * total_length

            # Find segment containing target length
            for j in range(len(cumulative_lengths) - 1):
                if cumulative_lengths[j] <= target_length <= cumulative_lengths[j + 1]:
                    # Interpolate within this segment
                    seg_length = cumulative_lengths[j + 1] - cumulative_lengths[j]
                    if seg_length > 0:
                        t = (target_length - cumulative_lengths[j]) / seg_length
                    else:
                        t = 0

                    x = points[j]['x'] + t * (points[j + 1]['x'] - points[j]['x'])
                    y = points[j]['y'] + t * (points[j + 1]['y'] - points[j]['y'])
                    resampled.append({'x': x, 'y': y})
                    break
            else:
                # Fallback to last point (shouldn't happen with correct logic)
                resampled.append({'x': points[-1]['x'], 'y': points[-1]['y']})

        return resampled

    def interpolate_path(self, points, t):
        """
        Interpolate position along a path at time t (0.0 to 1.0)
        Returns (x, y) coordinates

        NOTE: This is used for visualization/animation only.
        For WAN ATI output, use resample_path_uniform() instead.
        """
        if len(points) == 0:
            return (0, 0)

        # Support static single points
        if len(points) == 1:
            return (points[0]['x'], points[0]['y'])

        # Calculate total path length
        total_length = 0
        segment_lengths = []
        for i in range(len(points) - 1):
            dx = points[i + 1]['x'] - points[i]['x']
            dy = points[i + 1]['y'] - points[i]['y']
            length = math.sqrt(dx * dx + dy * dy)
            segment_lengths.append(length)
            total_length += length

        if total_length == 0:
            return (points[0]['x'], points[0]['y'])

        # Find target distance along path
        target_distance = t * total_length

        # Find which segment contains target distance
        current_distance = 0
        for i, seg_length in enumerate(segment_lengths):
            if current_distance + seg_length >= target_distance:
                # Interpolate within this segment
                segment_t = (target_distance - current_distance) / seg_length if seg_length > 0 else 0
                x = points[i]['x'] + (points[i + 1]['x'] - points[i]['x']) * segment_t
                y = points[i]['y'] + (points[i + 1]['y'] - points[i]['y']) * segment_t
                return (x, y)
            current_distance += seg_length

        # Return last point if we've gone past the end
        return (points[-1]['x'], points[-1]['y'])

    def frame_to_normalized(self, frame, total_frames):
        """Convert frame number to normalized time (0.0 to 1.0)"""
        if total_frames <= 1:
            return 0.0
        return frame / (total_frames - 1)

    def animate_paths(self, frame_width, frame_height, frame_count, shape, shape_size,
                     shape_color, bg_color, blur_radius=0.0, trail_length=0.0,
                     rotation_speed=0.0, border_width=0, border_color='white',
                     paths_data='{"paths": [], "canvas": {"width": 512, "height": 512}, "total_frames": 77}',
                     visibility_mode='pop'):

        # Parse colors
        shape_color = parse_color(shape_color)
        bg_color = parse_color(bg_color)
        border_color = parse_color(border_color)

        # Parse paths data (Weaver JSON format)
        try:
            paths_obj = json.loads(paths_data)
            paths = paths_obj.get('paths', [])
            
            # Support both 'canvas' and 'canvas_size' keys for compatibility
            canvas_size = paths_obj.get('canvas', paths_obj.get('canvas_size', {'width': frame_width, 'height': frame_height}))
            
            # Get total frames from JSON or use default
            json_total_frames = paths_obj.get('total_frames', TOTAL_FRAMES)
            
        except json.JSONDecodeError:
            print("WeaverAnimator: Invalid JSON in paths_data, using empty paths")
            paths = []
            canvas_size = {'width': frame_width, 'height': frame_height}
            json_total_frames = TOTAL_FRAMES

        # Calculate scaling factors to transform from canvas coordinates to frame coordinates
        canvas_width = canvas_size.get('width', frame_width)
        canvas_height = canvas_size.get('height', frame_height)
        scale_x = frame_width / canvas_width if canvas_width > 0 else 1.0
        scale_y = frame_height / canvas_height if canvas_height > 0 else 1.0

        # Scale all path coordinates and convert frame-based timing
        scaled_paths = []
        for path in paths:
            scaled_path = path.copy()
            scaled_points = []
            for point in path.get('points', []):
                scaled_points.append({
                    'x': point['x'] * scale_x,
                    'y': point['y'] * scale_y
                })
            scaled_path['points'] = scaled_points

            # Preserve isSinglePoint flag if it exists
            if 'isSinglePoint' in path:
                scaled_path['isSinglePoint'] = path['isSinglePoint']

            # Convert Weaver's frame-based timing to normalized time (0.0 to 1.0)
            # Weaver uses startFrame/endFrame (0 to total_frames-1)
            start_frame = path.get('startFrame', 0)
            end_frame = path.get('endFrame', json_total_frames - 1)
            
            # Convert to normalized time for internal processing
            scaled_path['startTime'] = self.frame_to_normalized(start_frame, json_total_frames)
            scaled_path['endTime'] = self.frame_to_normalized(end_frame, json_total_frames)
            
            # Keep original frame values for coordinate output
            scaled_path['startFrame'] = start_frame
            scaled_path['endFrame'] = end_frame
            
            # Use Weaver's easing field (falls back to linear)
            scaled_path['easing'] = path.get('easing', 'linear')

            scaled_paths.append(scaled_path)

        images_list = []
        masks_list = []
        previous_output = None

        for frame in range(frame_count):
            # Create blank image with bg_color
            image = Image.new("RGB", (frame_width, frame_height), bg_color)
            draw = ImageDraw.Draw(image)

            # Calculate global time (0.0 to 1.0)
            global_t = frame / max(frame_count - 1, 1)

            # Draw each path's shape
            for path_idx, path in enumerate(scaled_paths):
                points = path.get('points', [])
                if len(points) == 0:
                    continue

                # Get timeline parameters (already converted to normalized time)
                start_time = path.get('startTime', 0.0)
                end_time = path.get('endTime', 1.0)
                easing = path.get('easing', 'linear')

                # Determine if shape should be visible and animated
                is_in_timeline = start_time <= global_t <= end_time

                # Skip rendering based on visibility mode
                if visibility_mode == 'pop' and not is_in_timeline:
                    # Pop mode: don't render outside timeline
                    continue

                # Calculate local time and position
                if is_in_timeline and end_time > start_time:
                    # Animate within timeline
                    local_t = (global_t - start_time) / (end_time - start_time)
                    eased_t = apply_easing(local_t, easing)
                    x, y = self.interpolate_path(points, eased_t)
                    # Calculate rotation based on local time
                    current_rotation = rotation_speed * eased_t * 360.0
                elif visibility_mode == 'static':
                    # Static mode: show at start or end position when outside timeline
                    if global_t < start_time:
                        # Before timeline: show at start position
                        x, y = self.interpolate_path(points, 0.0)
                        current_rotation = 0.0
                    else:
                        # After timeline: show at end position
                        x, y = self.interpolate_path(points, 1.0)
                        current_rotation = rotation_speed * 360.0
                else:
                    # Fallback: show at start position
                    x, y = self.interpolate_path(points, 0.0)
                    current_rotation = 0.0

                # Draw the shape
                self.draw_shape(draw, shape, x, y, shape_size, current_rotation,
                              shape_color, border_width, border_color)

            # Apply blur
            if blur_radius > 0:
                image = image.filter(ImageFilter.GaussianBlur(blur_radius))

            # Convert to tensor
            image_tensor = pil2tensor(image)

            # Apply trailing effect
            if trail_length > 0 and previous_output is not None:
                image_tensor = image_tensor + trail_length * previous_output
                image_tensor = image_tensor / image_tensor.max()

            previous_output = image_tensor.clone()

            # Clamp values
            image_tensor = torch.clamp(image_tensor, 0.0, 1.0)

            # Extract mask from red channel
            mask = image_tensor[:, :, :, 0]

            images_list.append(image_tensor)
            masks_list.append(mask)

        # Concatenate all frames
        out_images = torch.cat(images_list, dim=0)
        out_masks = torch.cat(masks_list, dim=0)

        # Generate WAN ATI-compatible coordinate string
        # Resample each path to exactly 77 points with easing applied
        coord_tracks = []
        for path in scaled_paths:
            points = path.get('points', [])
            
            if len(points) == 0:
                continue

            # Check if this is a single-point path (static anchor)
            is_single_point = path.get('isSinglePoint', False) or len(points) == 1
            
            # Get frame range for this path
            start_frame = path.get('startFrame', 0)
            end_frame = path.get('endFrame', TOTAL_FRAMES - 1)
            easing = path.get('easing', 'linear')

            # Resample to exactly 77 points with easing applied
            resampled_points = self.resample_path_with_easing(
                points, 
                num_samples=TOTAL_FRAMES,
                easing=easing,
                start_frame=start_frame,
                end_frame=end_frame
            )

            # Format: [{"x": x, "y": y}, {"x": x, "y": y}, ...]
            track_coords = [
                {"x": int(round(p["x"])), "y": int(round(p["y"]))}
                for p in resampled_points
            ]

            coord_tracks.append(track_coords)

        # Output as list of tracks (each track is a list of 77 {x, y} points)
        coord_string = json.dumps(coord_tracks)

        print(f"WeaverAnimator: Generated {len(coord_tracks)} tracks with {TOTAL_FRAMES} points each for Animate")

        return (out_images, out_masks, coord_string)

    def resample_path_with_easing(self, points, num_samples=TOTAL_FRAMES, easing='linear', 
                                   start_frame=0, end_frame=None):
        """
        Resample path to exactly num_samples points with easing applied.
        Points outside the active frame range hold their position.
        
        Args:
            points: List of {x, y} dicts representing the path
            num_samples: Number of points to resample to (default 77)
            easing: Easing function name
            start_frame: Frame where animation starts (0-based)
            end_frame: Frame where animation ends (0-based, default num_samples-1)
            
        Returns:
            List of {x, y} dicts with exactly num_samples points
        """
        if end_frame is None:
            end_frame = num_samples - 1
            
        if len(points) == 0:
            return []

        # Single point - repeat for all samples (static anchor)
        if len(points) == 1:
            return [{'x': points[0]['x'], 'y': points[0]['y']} for _ in range(num_samples)]

        # First, get uniformly sampled path (without easing)
        uniform_samples = self.resample_path_uniform(points, num_samples=1000)  # High-res intermediate
        
        # Now sample with easing applied, respecting frame range
        resampled = []
        for frame in range(num_samples):
            if frame < start_frame:
                # Before animation: hold at start
                t = 0.0
            elif frame > end_frame:
                # After animation: hold at end
                t = 1.0
            else:
                # During animation: apply easing
                if end_frame > start_frame:
                    local_t = (frame - start_frame) / (end_frame - start_frame)
                    t = apply_easing(local_t, easing)
                else:
                    t = 1.0
            
            # Interpolate along the uniform samples
            x, y = self.interpolate_path(uniform_samples, t)
            resampled.append({'x': x, 'y': y})
        
        return resampled

# =====================
# ComfyUI registration
# =====================

NODE_CLASS_MAPPINGS = {
    "LoadFromUrl": LoadFromUrl,
    "SaveImageToDisk": SaveImageToDisk,
    "S3UploadPresigned": S3UploadPresigned,
    "WeaverAnimator": WeaverAnimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFromUrl": "URL: Load Media",
    "SaveImageToDisk": "Image: Save To Disk (No UI)",
    "S3UploadPresigned": "S3: Upload Presigned",
    "WeaverAnimator": "Weaver Path Animator",
}
