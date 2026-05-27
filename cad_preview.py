# ============================================================
#  VeriFYD — cad_preview.py
#
#  Free/open-source CAD preview helper.
#
#  Scope:
#    - DXF visual previews using ezdxf + matplotlib.
#    - DWG is intentionally not rendered here because DWG usually needs
#      a DWG-to-DXF converter first. The caller should keep the current
#      evidence-preview fallback for DWG when conversion is unavailable.
# ============================================================

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

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


def render_dxf_preview(src_path: str, *, out_path: Optional[str] = None, dpi: int = 180) -> CadPreviewResult:
    """
    Render a DXF file to a PNG preview using ezdxf's matplotlib backend.

    This is intentionally best-effort. If ezdxf/matplotlib cannot render the
    file, the caller should fall back to the existing VeriFYD CAD evidence PDF.
    """
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
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        Frontend(ctx, backend).draw_layout(msp, finalize=True)

        try:
            ax.autoscale()
            ax.margins(0.03)
        except Exception:
            pass

        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            return CadPreviewResult(False, source_path=src_path, file_type="dxf", entity_count=entity_count, layer_count=layer_count, message="DXF render produced no usable preview image.")

        log.info("cad_preview: rendered DXF preview src=%s preview=%s size=%d entities=%d layers=%d", src_path, out_path, os.path.getsize(out_path), entity_count, layer_count)
        return CadPreviewResult(True, source_path=src_path, preview_path=out_path, file_type="dxf", entity_count=entity_count, layer_count=layer_count, layout_name="Model", message="DXF preview rendered successfully.")

    except Exception as exc:
        log.warning("cad_preview: DXF render failed src=%s error=%s", src_path, exc)
        try:
            if out_path and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return CadPreviewResult(False, source_path=src_path, file_type="dxf", message=f"DXF preview rendering failed: {str(exc)[:220]}")
