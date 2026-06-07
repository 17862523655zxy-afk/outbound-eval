"""PDF rendering for evaluation reports.

Uses :mod:`weasyprint` when available. On systems where the required
native libraries (libgobject, libpango, libcairo) are not installed, a
clear ``PdfRenderError`` is raised so callers can surface the exact
``brew install``/``apt`` command needed to fix it.
"""

from __future__ import annotations

from pathlib import Path


class PdfRenderError(RuntimeError):
    """Raised when HTML → PDF rendering is not possible on this system."""


def html_to_pdf(html_text: str, output_path: Path) -> Path:
    """Render an HTML string to a PDF file at ``output_path``.

    Returns the resolved output path. Raises :class:`PdfRenderError`
    on any failure (missing weasyprint, missing system libraries, etc.).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from weasyprint import HTML  # noqa: F401
    except ImportError as e:
        raise PdfRenderError(
            "PDF 渲染需要 weasyprint，但 Python 包未安装。\n"
            "  → pip install weasyprint\n"
            "  macOS 还需要: brew install cairo pango gdk-pixbuf libffi"
        ) from e
    except OSError as e:
        # weasyprint 包装了 cgo 系统库加载；缺 libgobject/pango/cairo 时抛 OSError
        raise PdfRenderError(
            "PDF 渲染失败：缺少系统库（weasyprint 已装但无法加载原生库）。\n"
            "  macOS: brew install cairo pango gdk-pixbuf libffi\n"
            "  Ubuntu/Debian: apt install libpango-1.0-0 libpangoft2-1.0-0\n"
            "  然后重试。降级方案：HTML 报告可直接在浏览器中打开并'打印 → 另存为 PDF'。\n"
            f"  原始错误: {str(e)[:200]}"
        ) from e

    try:
        HTML(string=html_text, base_url=str(output_path.parent)).write_pdf(
            target=str(output_path)
        )
    except Exception as e:
        # Most common cause: missing system libraries on macOS
        err = str(e)
        if "libgobject" in err or "libpango" in err or "libcairo" in err:
            raise PdfRenderError(
                "PDF 渲染失败：缺少系统库。\n"
                "  macOS: brew install cairo pango gdk-pixbuf libffi\n"
                "  Ubuntu/Debian: apt install libpango-1.0-0 libpangoft2-1.0-0\n"
                "  然后重试。降级方案：HTML 报告可直接在浏览器中打开并'打印 → 另存为 PDF'。\n"
                f"  原始错误: {err[:200]}"
            ) from e
        raise PdfRenderError(f"PDF 渲染失败: {e}") from e

    return output_path
