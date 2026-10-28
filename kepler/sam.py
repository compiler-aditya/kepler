"""SAM 3.1 calls through Meta Model API (Responses API, OpenAI-compatible).

Docs: https://dev.meta.ai/docs/sam/segmenting
- One concept per request ("steel ball"), or box/point prompts for known objects.
- Video: MP4/MOV, H.264/H.265/MPEG-4, up to 15,000 frames per request.
- Billed per frame ($0.20 per 1,000), minimum 50 frames per request.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from .samparse import Record, parse_output

BASE_URL = "https://api.meta.ai/v1"
MODEL = "sam-3.1"


def _client():
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install openai  (the SAM calls use the OpenAI SDK)") from e
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise SystemExit("Set MODEL_API_KEY (from dev.meta.ai > API keys) in your environment or .env")
    return OpenAI(base_url=BASE_URL, api_key=key)


def upload(path: str | Path, cache: str | Path | None = None) -> str:
    """Upload a clip once and return its file_id. Caches ids in a JSON file next to the clip."""
    path = Path(path)
    cache = Path(cache) if cache else path.with_suffix(path.suffix + ".fileid.json")
    stat = path.stat()
    key = f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("key") == key:
            return saved["file_id"]
    with open(path, "rb") as f:
        file = _client().files.create(purpose="user_data", file=f)
    cache.write_text(json.dumps({"key": key, "file_id": file.id}))
    return file.id


def _run(prompt_text: str, media_part: dict, metadata: dict, raw_log: Path | None) -> list[Record]:
    client = _client()
    chunks: list[str] = []
    with client.responses.stream(
        model=MODEL,
        input=[{"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": prompt_text}, media_part]}],
        metadata=metadata,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
    # Each frame arrives as its own delta; join with newlines so frames stay separate.
    text = "\n".join(chunks)
    if raw_log:
        raw_log.parent.mkdir(parents=True, exist_ok=True)
        raw_log.write_text(text)
    return parse_output(text)


def track_video(video: str | Path, concept: str, *, raw_log: Path | None = None,
                file_id: str | None = None) -> list[Record]:
    """Track every instance of `concept` through the clip."""
    fid = file_id or upload(video)
    return _run(
        concept,
        {"type": "input_video", "file_id": fid},
        {"mask_encoding": "one_bit", "include_confidence": "true"},
        raw_log,
    )


def track_video_box(video: str | Path, box: tuple[int, int, int, int], size: tuple[int, int],
                    frame: int = 0, *, raw_log: Path | None = None,
                    file_id: str | None = None) -> list[Record]:
    """Track one object you marked with a box on `frame` (x1, y1, x2, y2 inclusive, display pixels)."""
    x1, y1, x2, y2 = box
    w, h = size
    prompt = f"<{frame}f>1<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h}|>"
    fid = file_id or upload(video)
    return _run(
        prompt,
        {"type": "input_video", "file_id": fid},
        {"include_confidence": "true", "propagation_direction": "forward"},
        raw_log,
    )


def segment_image(image_bgr, concept: str, *, raw_log: Path | None = None) -> list[Record]:
    """Segment `concept` in one still (used for the A4 sheet, which doesn't move)."""
    import cv2

    ok, png = cv2.imencode(".png", image_bgr)
    if not ok:
        raise SystemExit("could not encode frame as PNG")
    url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
    return _run(
        concept,
        {"type": "input_image", "image_url": url},
        {"mask_encoding": "one_bit", "include_confidence": "true"},
        raw_log,
    )
