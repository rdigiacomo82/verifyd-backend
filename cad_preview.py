# ============================================================
#  VeriFYD — cad_preview.py
#
#  Free/open-source CAD preview helper.
#
#  Scope:
#    - DXF visual previews using ezdxf.
#    - Primary path uses ezdxf + matplotlib backend.
#    - Fallback path manually draws common DXF entities in black-on-white.
#    - DWG requires caller-side conversion to DXF first.
# ============================================================

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("verifyd.cad_preview")


@dataclass
class CadPreviewResult:
    success: bool
    source_path: str
    preview_path: str = ""
    file_type: str = ""
    entity_count: int = 0
    layer_count: int = 0
    layout_name: str = "Model"
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _preview_has_visible_ink(path: str, *, min_ratio: float = 0.0008) -> bool:
    # Return True when a rendered PNG appears to contain actual drawing ink.
    try:
        if not path or not os.path.exists(path) or os.path.getsize(path) < 10_000:
            return False
        from PIL import Image
        img = Image.open(path).convert("L")
        w, h = img.size
        if w <= 0 or h <= 0:
            return False
        dark = sum(1 for p in img.getdata() if p < 245)
        return (dark / float(w * h)) >= min_ratio
    except Exception:
        try:
            return os.path.getsize(path) >= 18_000
        except Exception:
            return False


def _entity_points_for_bounds(entity: Any) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    try:
        typ = entity.dxftype().upper()
        if typ == "LINE":
            pts.append((float(entity.dxf.start.x), float(entity.dxf.start.y)))
            pts.append((float(entity.dxf.end.x), float(entity.dxf.end.y)))
        elif typ == "LWPOLYLINE":
            for p in entity.get_points("xy"):
                pts.append((float(p[0]), float(p[1])))
        elif typ == "POLYLINE":
            for v in entity.vertices:
                loc = v.dxf.location
                pts.append((float(loc.x), float(loc.y)))
        elif typ in ("CIRCLE", "ARC"):
            c = entity.dxf.center
            r = abs(float(entity.dxf.radius))
            pts.extend([(float(c.x) - r, float(c.y) - r), (float(c.x) + r, float(c.y) + r)])
        elif typ in ("TEXT", "MTEXT", "POINT"):
            p = getattr(entity.dxf, "insert", None) or getattr(entity.dxf, "location", None)
            if p is not None:
                pts.append((float(p.x), float(p.y)))
    except Exception:
        pass
    return pts


def _collect_bounds(entities: List[Any]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for e in entities:
        for x, y in _entity_points_for_bounds(e):
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return (0.0, 0.0, 10.0, 7.5)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if abs(max_x - min_x) < 0.1:
        max_x = min_x + 10.0
    if abs(max_y - min_y) < 0.1:
        max_y = min_y + 7.5

    pad_x = max((max_x - min_x) * 0.08, 2.0)
    pad_y = max((max_y - min_y) * 0.08, 2.0)
    return (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)


def _plain_entity_text(entity: Any) -> str:
    try:
        if hasattr(entity, "plain_text"):
            return str(entity.plain_text() or "").strip()
    except Exception:
        pass
    try:
        return str(entity.dxf.text or "").strip()
    except Exception:
        return ""


def _render_dxf_preview_simple(src_path: str, out_path: str, *, dpi: int = 180) -> CadPreviewResult:
    # Deterministic black-on-white fallback renderer for common 2D DXF entities.
    try:
        import ezdxf
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Arc

        doc = ezdxf.readfile(src_path)
        msp = doc.modelspace()
        entities = list(msp)
        entity_count = len(entities)
        try:
            layer_count = len(list(doc.layers))
        except Exception:
            layer_count = 0

        if entity_count <= 0:
            return CadPreviewResult(False, source_path=src_path, file_type="dxf", entity_count=0, layer_count=layer_count, message="DXF loaded but modelspace contains no drawable entities.")

        min_x, min_y, max_x, max_y = _collect_bounds(entities)
        width = max_x - min_x
        height = max_y - min_y
        aspect = width / max(height, 0.001)

        fig_w = 13.0
        fig_h = max(6.0, min(10.0, fig_w / max(aspect, 0.15)))
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        drawn = 0

        for e in entities:
            try:
                typ = e.dxftype().upper()

                if typ == "LINE":
                    ax.plot([float(e.dxf.start.x), float(e.dxf.end.x)], [float(e.dxf.start.y), float(e.dxf.end.y)], color="black", linewidth=1.25)
                    drawn += 1

                elif typ == "LWPOLYLINE":
                    pts = [(float(p[0]), float(p[1])) for p in e.get_points("xy")]
                    if len(pts) >= 2:
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys, color="black", linewidth=1.25)
                        try:
                            if bool(e.closed):
                                ax.plot([pts[-1][0], pts[0][0]], [pts[-1][1], pts[0][1]], color="black", linewidth=1.25)
                        except Exception:
                            pass
                        drawn += 1

                elif typ == "POLYLINE":
                    pts = []
                    for v in e.vertices:
                        loc = v.dxf.location
                        pts.append((float(loc.x), float(loc.y)))
                    if len(pts) >= 2:
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys, color="black", linewidth=1.25)
                        drawn += 1

                elif typ == "CIRCLE":
                    c = e.dxf.center
                    r = abs(float(e.dxf.radius))
                    ax.add_patch(Circle((float(c.x), float(c.y)), r, fill=False, edgecolor="black", linewidth=1.25))
                    drawn += 1

                elif typ == "ARC":
                    c = e.dxf.center
                    r = abs(float(e.dxf.radius))
                    ax.add_patch(Arc((float(c.x), float(c.y)), 2 * r, 2 * r, angle=0, theta1=float(e.dxf.start_angle), theta2=float(e.dxf.end_angle), edgecolor="black", linewidth=1.25))
                    drawn += 1

                elif typ == "POINT":
                    p = e.dxf.location
                    ax.plot([float(p.x)], [float(p.y)], marker="o", color="black", markersize=3)
                    drawn += 1

                elif typ in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
                    p = getattr(e.dxf, "insert", None)
                    text = _plain_entity_text(e)
                    if p is not None and text:
                        height_val = abs(float(getattr(e.dxf, "height", 4.0) or 4.0))
                        fontsize = max(6.5, min(13.0, height_val * 1.15))
                        ax.text(float(p.x), float(p.y), text, color="black", fontsize=fontsize, ha="left", va="center")
                        drawn += 1

            except Exception:
                continue

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)

        if drawn <= 0 or not _preview_has_visible_ink(out_path, min_ratio=0.0005):
            return CadPreviewResult(False, source_path=src_path, file_type="dxf", entity_count=entity_count, layer_count=layer_count, message=f"Fallback renderer produced no usable visible preview; drawn={drawn}.")

        log.info("cad_preview: simple fallback rendered DXF preview src=%s preview=%s size=%d entities=%d drawn=%d layers=%d", src_path, out_path, os.path.getsize(out_path), entity_count, drawn, layer_count)
        return CadPreviewResult(True, source_path=src_path, preview_path=out_path, file_type="dxf", entity_count=entity_count, layer_count=layer_count, layout_name="Model", message=f"DXF preview rendered successfully with simple fallback renderer; drawn={drawn}.")

    except Exception as exc:
        log.warning("cad_preview: simple DXF render failed src=%s error=%s", src_path, exc)
        try:
            if out_path and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return CadPreviewResult(False, source_path=src_path, file_type="dxf", message=f"Simple DXF preview rendering failed: {str(exc)[:220]}")


def render_dxf_preview(src_path: str, *, out_path: Optional[str] = None, dpi: int = 180) -> CadPreviewResult:
    # Render a DXF file to a PNG preview, with blank-image detection and fallback.
    src_path = str(src_path or "")
    if not src_path or not os.path.exists(src_path):
        return CadPreviewResult(False, source_path=src_path, file_type="dxf", message="DXF source file not found.")

    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix="_verifyd_dxf_preview.png")
        os.close(fd)

    try:
        import ezdxf
        from ezdxf.addons.drawing import RenderContext, Frontend
        from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        doc = ezdxf.readfile(src_path)
        msp = doc.modelspace()

        try:
            entities = list(msp)
        except Exception:
            entities = []
        entity_count = len(entities)

        try:
            layer_count = len(list(doc.layers))
        except Exception:
            layer_count = 0

        if entity_count <= 0:
            return CadPreviewResult(False, source_path=src_path, file_type="dxf", entity_count=0, layer_count=layer_count, message="DXF loaded but modelspace contains no drawable entities.")

        fig = plt.figure(figsize=(13, 8.5), dpi=dpi)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        ax.set_facecolor("white")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        Frontend(ctx, backend).draw_layout(msp, finalize=True)

        try:
            ax.autoscale(enable=True, axis="both", tight=True)
            ax.margins(0.08)
        except Exception:
            pass

        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        plt.close(fig)

        if _preview_has_visible_ink(out_path):
            log.info("cad_preview: rendered DXF preview src=%s preview=%s size=%d entities=%d layers=%d", src_path, out_path, os.path.getsize(out_path), entity_count, layer_count)
            return CadPreviewResult(True, source_path=src_path, preview_path=out_path, file_type="dxf", entity_count=entity_count, layer_count=layer_count, layout_name="Model", message="DXF preview rendered successfully.")

        log.warning("cad_preview: ezdxf backend produced mostly blank preview; falling back to simple renderer src=%s preview_size=%s entities=%d", src_path, os.path.getsize(out_path) if os.path.exists(out_path) else 0, entity_count)
        try:
            if out_path and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return _render_dxf_preview_simple(src_path, out_path, dpi=dpi)

    except Exception as exc:
        log.warning("cad_preview: DXF render failed src=%s error=%s; trying simple fallback", src_path, exc)
        try:
            if out_path and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return _render_dxf_preview_simple(src_path, out_path, dpi=dpi)
