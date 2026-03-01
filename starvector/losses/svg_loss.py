#!/usr/bin/env python3
"""
svg_loss.py

Computes a composite loss between two SVGs:
- Raster similarity (pixel L2)
- Edge similarity (Sobel L1)
- Structural penalties (too many segments, redundant points, unstable Bezier handles)
- Style penalties (stroke-width mismatch, color set mismatch)
- Optional palette constraint

Usage:
  python svg_loss.py pred.svg target.svg --size 256

Dependencies:
  pip install torch pillow cairosvg numpy
Optional (better geometry parsing):
  pip install svgpathtools
"""

from __future__ import annotations

import argparse
import io
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import cairosvg

try:
    from svgpathtools import parse_path  # type: ignore
except Exception:
    parse_path = None


# -----------------------------
# Utilities: SVG rendering
# -----------------------------

def render_svg_to_tensor(svg_str: str, size: int = 256) -> torch.Tensor:
    """
    Render SVG string to a float tensor in [0,1], shape [1, 3, H, W].
    Uses CairoSVG (non-differentiable).
    """
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), output_width=size, output_height=size)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    # Composite onto white background to avoid alpha weirdness
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("RGB")

    arr = np.asarray(img).astype(np.float32) / 255.0  # H,W,3
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return t


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,H,W] float in [0,1]
    returns: [B,1,H,W] edge magnitude
    """
    # grayscale
    gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3])

    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag


# -----------------------------
# Utilities: Style + color parsing
# -----------------------------

HEX_COLOR_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
RGB_COLOR_RE = re.compile(r"^rgb\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*\)$")


def parse_color(s: str) -> Optional[Tuple[int, int, int]]:
    s = s.strip()
    if s == "" or s.lower() == "none":
        return None

    m = HEX_COLOR_RE.match(s)
    if m:
        h = m.group(1)
        if len(h) == 3:
            r = int(h[0] * 2, 16)
            g = int(h[1] * 2, 16)
            b = int(h[2] * 2, 16)
        else:
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
        return (r, g, b)

    m = RGB_COLOR_RE.match(s)
    if m:
        r, g, b = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return (r, g, b)

    # Unknown formats (named colors, gradients, etc.)
    return None


def parse_style_attr(style_str: str) -> Dict[str, str]:
    """
    Converts "fill:#fff;stroke:#000;stroke-width:2" to dict.
    """
    out: Dict[str, str] = {}
    for part in style_str.split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip().lower()] = v.strip()
    return out


@dataclass
class SvgStyleSummary:
    fills: List[Tuple[int, int, int]]
    strokes: List[Tuple[int, int, int]]
    stroke_widths: List[float]

    @property
    def unique_colors(self) -> List[Tuple[int, int, int]]:
        return sorted(list({*self.fills, *self.strokes}))


def extract_style_summary(svg_str: str) -> SvgStyleSummary:
    """
    Pulls fill/stroke/stroke-width from common SVG elements.
    This is intentionally "best-effort", because SVG styling can be complex.
    """
    try:
        root = ET.fromstring(svg_str)
    except Exception:
        return SvgStyleSummary(fills=[], strokes=[], stroke_widths=[])

    fills: List[Tuple[int, int, int]] = []
    strokes: List[Tuple[int, int, int]] = []
    stroke_widths: List[float] = []

    def get_attr(el: ET.Element, key: str, style_map: Dict[str, str]) -> Optional[str]:
        if key in el.attrib:
            return el.attrib[key]
        return style_map.get(key)

    for el in root.iter():
        style_map = parse_style_attr(el.attrib.get("style", ""))
        fill_s = get_attr(el, "fill", style_map)
        stroke_s = get_attr(el, "stroke", style_map)
        sw_s = get_attr(el, "stroke-width", style_map)

        if fill_s is not None:
            c = parse_color(fill_s)
            if c is not None:
                fills.append(c)

        if stroke_s is not None:
            c = parse_color(stroke_s)
            if c is not None:
                strokes.append(c)

        if sw_s is not None:
            try:
                # strip "px" if present
                sw = float(sw_s.replace("px", "").strip())
                if math.isfinite(sw):
                    stroke_widths.append(sw)
            except Exception:
                pass

    return SvgStyleSummary(fills=fills, strokes=strokes, stroke_widths=stroke_widths)


# -----------------------------
# Utilities: Geometry / structure
# -----------------------------

@dataclass
class SvgGeomSummary:
    num_paths: int
    num_segments: int
    approx_anchor_points: int
    redundant_point_count: int
    bezier_instability_score: float


CMD_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")


def _approx_from_path_d(d: str) -> Tuple[int, int]:
    """
    Very rough fallback if svgpathtools is unavailable:
    counts commands and estimates anchor points from them.
    """
    cmds = CMD_RE.findall(d)
    num_cmds = len(cmds)
    # Anchor points roughly scale with number of drawing commands.
    approx_anchors = sum(1 for c in cmds if c.lower() not in ["z"])
    return num_cmds, approx_anchors


def extract_geom_summary(svg_str: str, eps: float = 1e-3) -> SvgGeomSummary:
    """
    Attempts to count:
    - number of paths
    - number of segments
    - redundant points (consecutive points closer than eps)
    - bezier instability (control handle too large relative to chord)

    If svgpathtools is available, this is much better.
    Otherwise, returns approximate counts.
    """
    try:
        root = ET.fromstring(svg_str)
    except Exception:
        return SvgGeomSummary(0, 0, 0, 0, 0.0)

    path_elems = [el for el in root.iter() if el.tag.lower().endswith("path") and "d" in el.attrib]
    num_paths = len(path_elems)

    total_segments = 0
    total_anchors = 0
    redundant_points = 0
    instability = 0.0

    if parse_path is None:
        for el in path_elems:
            segs, anchors = _approx_from_path_d(el.attrib.get("d", ""))
            total_segments += segs
            total_anchors += anchors
        return SvgGeomSummary(num_paths, total_segments, total_anchors, 0, 0.0)

    # svgpathtools-backed parsing
    for el in path_elems:
        d = el.attrib.get("d", "")
        try:
            p = parse_path(d)
        except Exception:
            segs, anchors = _approx_from_path_d(d)
            total_segments += segs
            total_anchors += anchors
            continue

        seg_list = list(p)
        total_segments += len(seg_list)

        # Approx anchors: number of segment endpoints + 1
        total_anchors += (len(seg_list) + 1) if len(seg_list) > 0 else 0

        # Redundant points: consecutive endpoints extremely close
        prev = None
        for seg in seg_list:
            end = seg.end  # complex
            if prev is not None:
                if abs(end - prev) < eps:
                    redundant_points += 1
            prev = end

        # Bezier instability: huge control handles compared to chord length
        for seg in seg_list:
            name = seg.__class__.__name__.lower()
            start = seg.start
            end = seg.end
            chord = abs(end - start) + 1e-9

            if "cubic" in name:
                c1 = seg.control1
                c2 = seg.control2
                h1 = abs(c1 - start) / chord
                h2 = abs(c2 - end) / chord
                # Penalize handles that are too large
                instability += max(0.0, h1 - 2.0) ** 2 + max(0.0, h2 - 2.0) ** 2

            elif "quadratic" in name:
                c = seg.control
                h = abs(c - start) / chord
                instability += max(0.0, h - 2.0) ** 2

    return SvgGeomSummary(
        num_paths=num_paths,
        num_segments=total_segments,
        approx_anchor_points=total_anchors,
        redundant_point_count=redundant_points,
        bezier_instability_score=float(instability),
    )


# -----------------------------
# Loss components
# -----------------------------

@dataclass
class LossWeights:
    w_raster: float = 1.0
    w_edge: float = 0.3
    w_simplicity: float = 0.05
    w_redundant: float = 0.2
    w_bezier_stability: float = 0.1
    w_style: float = 0.2
    w_palette: float = 0.2


def raster_l2_loss(pred_svg: str, tgt_svg: str, size: int) -> torch.Tensor:
    pred = render_svg_to_tensor(pred_svg, size=size)
    tgt = render_svg_to_tensor(tgt_svg, size=size)
    return F.mse_loss(pred, tgt)


def edge_l1_loss(pred_svg: str, tgt_svg: str, size: int) -> torch.Tensor:
    pred = render_svg_to_tensor(pred_svg, size=size)
    tgt = render_svg_to_tensor(tgt_svg, size=size)
    ep = sobel_edges(pred)
    et = sobel_edges(tgt)
    return F.l1_loss(ep, et)


def simplicity_penalty(pred_geom: SvgGeomSummary, tgt_geom: SvgGeomSummary) -> float:
    """
    Penalize extra anchors/segments beyond target.
    """
    extra_anchors = max(0, pred_geom.approx_anchor_points - tgt_geom.approx_anchor_points)
    extra_segments = max(0, pred_geom.num_segments - tgt_geom.num_segments)
    return float(extra_anchors + 0.5 * extra_segments)


def style_penalty(pred_style: SvgStyleSummary, tgt_style: SvgStyleSummary) -> float:
    """
    Penalize mismatched stroke widths and color sets (best-effort).
    """
    # stroke width mismatch
    def safe_mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if len(xs) > 0 else 0.0

    sw_pred = safe_mean(pred_style.stroke_widths)
    sw_tgt = safe_mean(tgt_style.stroke_widths)
    sw_loss = abs(sw_pred - sw_tgt)

    # color set mismatch (symmetric difference size)
    pred_colors = set(pred_style.unique_colors)
    tgt_colors = set(tgt_style.unique_colors)
    color_diff = len(pred_colors.symmetric_difference(tgt_colors))

    return float(sw_loss + 0.25 * color_diff)


def palette_penalty(pred_style: SvgStyleSummary, palette: Sequence[Tuple[int, int, int]], thresh: float = 10.0) -> float:
    """
    Penalize colors that are not close to palette colors.
    thresh is max allowed euclidean RGB distance before penalizing.
    """
    if len(palette) == 0:
        return 0.0

    pal = np.array(palette, dtype=np.float32)  # [P,3]
    penalty = 0.0
    for c in pred_style.unique_colors:
        col = np.array(c, dtype=np.float32)[None, :]  # [1,3]
        d = np.sqrt(((pal - col) ** 2).sum(axis=1)).min()
        if d > thresh:
            penalty += float((d - thresh) / 50.0)  # scale down
    return penalty


def compute_svg_loss(
    pred_svg: str,
    tgt_svg: str,
    size: int = 256,
    weights: Optional[LossWeights] = None,
    palette: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns:
      total_loss (torch scalar)
      components (python floats for logging)
    """
    if weights is None:
        weights = LossWeights()
    if palette is None:
        palette = []

    # Raster components (torch)
    l_raster = raster_l2_loss(pred_svg, tgt_svg, size=size)
    l_edge = edge_l1_loss(pred_svg, tgt_svg, size=size)

    # Structural + style components (floats)
    pred_geom = extract_geom_summary(pred_svg)
    tgt_geom = extract_geom_summary(tgt_svg)
    pred_style = extract_style_summary(pred_svg)
    tgt_style = extract_style_summary(tgt_svg)

    l_simp = simplicity_penalty(pred_geom, tgt_geom)
    l_red = float(pred_geom.redundant_point_count)
    l_bez = float(pred_geom.bezier_instability_score)
    l_style = style_penalty(pred_style, tgt_style)
    l_pal = palette_penalty(pred_style, palette=palette)

    total = (
        weights.w_raster * l_raster
        + weights.w_edge * l_edge
        + weights.w_simplicity * torch.tensor(l_simp, dtype=l_raster.dtype)
        + weights.w_redundant * torch.tensor(l_red, dtype=l_raster.dtype)
        + weights.w_bezier_stability * torch.tensor(l_bez, dtype=l_raster.dtype)
        + weights.w_style * torch.tensor(l_style, dtype=l_raster.dtype)
        + weights.w_palette * torch.tensor(l_pal, dtype=l_raster.dtype)
    )

    components = {
        "raster_mse": float(l_raster.detach().cpu().item()),
        "edge_l1": float(l_edge.detach().cpu().item()),
        "simplicity": float(l_simp),
        "redundant_pts": float(l_red),
        "bezier_instability": float(l_bez),
        "style": float(l_style),
        "palette": float(l_pal),
        "pred_num_paths": float(pred_geom.num_paths),
        "pred_num_segments": float(pred_geom.num_segments),
        "pred_anchor_points": float(pred_geom.approx_anchor_points),
    }
    return total, components


# -----------------------------
# CLI
# -----------------------------

def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_svg_path", type=str)
    ap.add_argument("tgt_svg_path", type=str)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--palette", type=str, default="", help="Comma-separated hex colors, ex: #000000,#ffffff,#ff0055")
    args = ap.parse_args()

    pred = _read_file(args.pred_svg_path)
    tgt = _read_file(args.tgt_svg_path)

    palette: List[Tuple[int, int, int]] = []
    if args.palette.strip():
        for item in args.palette.split(","):
            c = parse_color(item.strip())
            if c is not None:
                palette.append(c)

    loss, comps = compute_svg_loss(pred, tgt, size=args.size, palette=palette)
    print(f"TOTAL LOSS: {float(loss.detach().cpu().item()):.6f}")
    for k in sorted(comps.keys()):
        print(f"{k:>18s}: {comps[k]:.6f}")


if __name__ == "__main__":
    main()