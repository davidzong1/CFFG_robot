"""Sample frames from the most recent training videos for the LLM."""

from __future__ import annotations

import io
from pathlib import Path

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    iio = None  # type: ignore

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None
    ImageFont = None


class VideoSampler:
    """Pull a few evenly-spaced frames out of the most recent train videos.

    Frames are returned as a list of ``(label, png_bytes)`` pairs ready to
    be attached to a multimodal LLM request.
    """

    def __init__(
        self,
        video_dir: Path,
        clips_per_cycle: int = 2,
        frames_per_clip: int = 6,
    ):
        self.video_dir = Path(video_dir)
        self.clips_per_cycle = clips_per_cycle
        self.frames_per_clip = frames_per_clip

    def _recent_videos(self) -> list[Path]:
        if not self.video_dir.exists():
            return []
        files = sorted(
            self.video_dir.glob("*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return files[: self.clips_per_cycle]

    def frames(self, overlay: dict[str, str] | None = None) -> list[tuple[str, bytes]]:
        if iio is None or Image is None:
            return []
        out: list[tuple[str, bytes]] = []
        for video_path in self._recent_videos():
            try:
                frames = list(iio.imiter(str(video_path), plugin="FFMPEG"))
            except Exception:
                continue
            if not frames:
                continue
            n = len(frames)
            step = max(n // self.frames_per_clip, 1)
            picks = list(range(0, n, step))[: self.frames_per_clip]
            for i, idx in enumerate(picks):
                img = Image.fromarray(frames[idx])
                if overlay:
                    img = _overlay(img, overlay | {"frame": f"{idx}/{n}"})
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                out.append((f"{video_path.stem}:frame{i}", buf.getvalue()))
        return out


def _overlay(img, info: dict[str, str]):
    """Burn a small key/value table into the top-left of the frame."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    lines = [f"{k}: {v}" for k, v in info.items()]
    text = "\n".join(lines)
    # Background rectangle for readability
    if font is not None:
        bbox = draw.multiline_textbbox((4, 4), text, font=font)
    else:
        bbox = (4, 4, 200, 60)
    draw.rectangle(bbox, fill=(0, 0, 0, 160))
    draw.multiline_text((4, 4), text, fill=(255, 255, 255), font=font)
    return img
