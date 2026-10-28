"""Find the A4 sheet's corners and build the pixel -> millimetre homography.

The sheet must lie in the plane the ball's centre moves in. A sheet behind
the motion makes every distance wrong by (gap / camera distance).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

A4_LONG_MM = 297.0
A4_SHORT_MM = 210.0


@dataclass
class SheetFit:
    corners_px: np.ndarray  # 4x2, order: tl, tr, br, bl (as seen in the image)
    H: np.ndarray  # 3x3, image px -> sheet mm (origin at tl corner)
    size_mm: tuple[float, float]  # (along tl->tr, along tl->bl)
    edge_rms_mm: float  # how straight the sheet's edges are after mapping

    def to_json(self) -> dict:
        return {
            "corners_px": self.corners_px.round(2).tolist(),
            "homography_px_to_mm": self.H.tolist(),
            "size_mm": list(self.size_mm),
            "edge_rms_mm": round(self.edge_rms_mm, 3),
        }


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    cx, cy = pts.mean(0)
    # Image y points down, so ascending angle runs tl, tr, br, bl.
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))]
    start = int(np.argmin(pts.sum(1)))
    return np.roll(pts, -start, axis=0)


def _quad_from_contour(cnt: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float64)
    return None


def detect_corners(frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Find the brightest large quadrilateral (the sheet). roi = x1, y1, x2, y2 inclusive."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    x1, y1, x2, y2 = roi if roi else (0, 0, W - 1, H - 1)
    pad = 12
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W - 1, x2 + pad), min(H - 1, y2 + pad)
    crop = cv2.GaussianBlur(gray[y1:y2 + 1, x1:x2 + 1], (5, 5), 0)
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise SystemExit("no bright region found for the A4 sheet; pass --sheet with 4 corners")
    cnt = max(cnts, key=cv2.contourArea)
    quad = _quad_from_contour(cnt)
    if quad is None:
        raise SystemExit("the sheet's outline isn't a clean 4-sided shape; pass --sheet with 4 corners")
    quad = order_corners(quad + (x1, y1))
    edge_pts = _edge_points(gray, quad)
    refined = _corners_from_edges(edge_pts, quad)
    # Keep the refinement only if it stayed close to the rough quad.
    if refined is not None and np.all(np.linalg.norm(refined - quad, axis=1) < 6):
        return refined
    return quad


def _edge_points(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Sub-pixel points on the sheet's outline: strongest gradient across each edge."""
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    pts = []
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        d = b - a
        L = np.linalg.norm(d)
        u = d / L
        nrm = np.array([-u[1], u[0]])
        for s in np.linspace(0.12, 0.88, max(20, int(L / 3))):  # skip near the corners
            p = a + d * s
            offs = np.arange(-6, 6.01, 0.25)
            samp = p[None, :] + offs[:, None] * nrm[None, :]
            vals = cv2.remap(mag, samp[:, 0].astype(np.float32).reshape(1, -1),
                             samp[:, 1].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)[0]
            i = int(np.argmax(vals))
            if 0 < i < len(vals) - 1 and vals[i] > 0:
                # Parabolic peak interpolation for a sub-sample edge position.
                den = vals[i - 1] - 2 * vals[i] + vals[i + 1]
                frac = 0.5 * (vals[i - 1] - vals[i + 1]) / den if den != 0 else 0.0
                pts.append((k, *(p + (offs[i] + frac * 0.25) * nrm)))
    return np.array(pts)


def _corners_from_edges(pts: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    lines = []
    for k in range(4):
        e = pts[pts[:, 0] == k][:, 1:]
        if len(e) < 8:
            return None
        vx, vy, x0, y0 = cv2.fitLine(e.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], float), np.array([vx, vy], float)))
    corners = []
    for k in range(4):
        (p1, d1), (p2, d2) = lines[(k - 1) % 4], lines[k]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-9:
            return None
        s = np.linalg.solve(A, p2 - p1)
        corners.append(p1 + s[0] * d1)
    return np.array(corners)


def fit(frame_bgr: np.ndarray, corners_px: np.ndarray) -> SheetFit:
    c = order_corners(corners_px)
    top = np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])
    side = np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])
    size = (A4_LONG_MM, A4_SHORT_MM) if top >= side else (A4_SHORT_MM, A4_LONG_MM)
    dst = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], dtype=np.float64)
    H = cv2.getPerspectiveTransform(c.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
    return SheetFit(c, H, size, _edge_rms(frame_bgr, c, H, size))


def _edge_rms(frame_bgr, c, H, size) -> float:
    """Map the sheet's detected outline through H; measure distance to a perfect rectangle (mm)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, c.astype(np.int32), 255)
    band = cv2.dilate(mask, np.ones((15, 15), np.uint8)) & ~cv2.erode(mask, np.ones((15, 15), np.uint8))
    _, bw = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(bw, 50, 150) & band
    ys, xs = np.nonzero(edges)
    if len(xs) < 20:
        return float("nan")
    pts = cv2.perspectiveTransform(np.stack([xs, ys], 1).reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    W, Hh = size
    dist = np.minimum.reduce([np.abs(pts[:, 0]), np.abs(pts[:, 0] - W), np.abs(pts[:, 1]), np.abs(pts[:, 1] - Hh)])
    dist = dist[dist < 10]  # ignore stray edges far from the outline
    return float(np.sqrt(np.mean(dist ** 2))) if len(dist) else float("nan")


def px_to_mm(H: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.stack([x, y], 1).reshape(-1, 1, 2).astype(np.float64)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out[:, 0], out[:, 1]


def mm_to_px(H: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return px_to_mm(np.linalg.inv(H), x_mm, y_mm)


def mm_per_px(H: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Local scale of the homography at each pixel: sqrt of the area a 1x1 pixel maps to."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x0, y0 = px_to_mm(H, x, y)
    x1, y1 = px_to_mm(H, x + 1, y)
    x2, y2 = px_to_mm(H, x, y + 1)
    return np.sqrt(np.abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)))
