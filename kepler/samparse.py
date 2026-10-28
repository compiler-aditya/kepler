"""Parse SAM 3.1 segmentation output (the special-token text in the output_text lane).

Grammar (from https://dev.meta.ai/docs/sam/reading-segmentation):

    line   ::= "<" N "f>" record ("," record)*
    record ::= OBJECT_ID box mask
    box    ::= "<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..[;c=..]|>"
    mask   ::= "<|mask;x=0;y=0[;c=..];data=H,W,<enc>payload|>"

x2 and y2 are inclusive. Frame markers may skip and may not be ascending, so
records are keyed by the explicit frame number, never by line position.

Masks are kept as raw tokens. Decoding them needs Meta's parser
(`pip install meta-sam-parser`); positions here come from box centres, which
for a ball are the centre of its disc to within half a pixel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FRAME_RE = re.compile(r"<(\d+)f>")
RECORD_HEAD_RE = re.compile(r"(\d+)<\|box;([^|]*)\|><\|mask;")
# A mask payload is base85 and may itself contain "|", "!" or "~", so its end
# is the "|>" that is followed by the next record or by the end of the line.
MASK_END_RE = re.compile(r"\|>(?=,\d+<\|box;|\s*$)")


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int  # inclusive
    y2: int  # inclusive
    w: int  # source frame width
    h: int  # source frame height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        return self.y2 - self.y1 + 1


@dataclass(frozen=True)
class Record:
    frame: int
    object_id: int
    box: Box
    confidence: float | None
    mask_rows: int
    mask_cols: int
    mask_encoding: str  # "lossless" or "one_bit"
    mask_payload: str = field(repr=False)


def _fields(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_line(line: str) -> list[Record]:
    """Parse one frame line into records. Returns [] for blank lines."""
    line = line.strip()
    if not line:
        return []
    m = FRAME_RE.match(line)
    if not m:
        raise ValueError(f"line does not start with a frame marker: {line[:40]!r}")
    frame = int(m.group(1))
    body = line[m.end():]
    records: list[Record] = []
    pos = 0
    while True:
        rm = RECORD_HEAD_RE.match(body, pos)
        if not rm:
            break
        end = MASK_END_RE.search(body, rm.end())
        if not end:
            raise ValueError(f"unterminated mask token on frame {frame}")
        mask_text = body[rm.end():end.start()]
        pos = end.end() + (1 if body[end.end():end.end() + 1] == "," else 0)
        obj = int(rm.group(1))
        bf = _fields(rm.group(2))
        head, data = mask_text.split("data=", 1)
        mf = _fields(head)
        rows, cols, rest = data.split(",", 2)
        enc_char, payload = rest[:1], rest[1:]
        conf = bf.get("c") or mf.get("c")
        records.append(
            Record(
                frame=frame,
                object_id=obj,
                box=Box(
                    int(bf["x1"]), int(bf["y1"]), int(bf["x2"]), int(bf["y2"]),
                    int(bf["w"]), int(bf["h"]),
                ),
                confidence=float(conf) if conf is not None else None,
                mask_rows=int(rows),
                mask_cols=int(cols),
                mask_encoding={"~": "lossless", "!": "one_bit"}.get(enc_char, enc_char),
                mask_payload=payload,
            )
        )
    return records


def parse_output(text: str) -> list[Record]:
    """Parse the whole accumulated output_text (one line per frame)."""
    records: list[Record] = []
    # Lines may arrive as separate deltas without newlines, so split on frame markers.
    starts = [m.start() for m in FRAME_RE.finditer(text)]
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        records.extend(parse_line(text[s:e]))
    return records


def by_frame(records: list[Record], object_id: int | None = None) -> dict[int, Record]:
    """Index records by frame for one object.

    With object_id=None, picks the object seen on the most frames (the ball in
    a typical clip) and, on frames where several objects appear, keeps that one.
    """
    if not records:
        return {}
    if object_id is None:
        counts: dict[int, int] = {}
        for r in records:
            counts[r.object_id] = counts.get(r.object_id, 0) + 1
        object_id = max(counts, key=counts.get)
    return {r.frame: r for r in records if r.object_id == object_id}
