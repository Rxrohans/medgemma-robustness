"""Jinja2 template -> HTML -> PNG, via Playwright.

Playwright rather than WeasyPrint: no MSYS2/GTK on Windows, no poppler, no PDF
intermediate, identical behaviour locally and on Kaggle, and Chromium's CSS
support is what makes eight *visually distinct* templates achievable.

The single most important detail in this module is that ``Renderer`` launches
Chromium **once** and reuses it for every report. Launching per report is
roughly 20x slower and dominates generation time completely.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from playwright.sync_api import sync_playwright

from .schema import Report

TEMPLATE_DIR = Path(__file__).parent / "templates"

# A4 at 96 CSS px/in. device_scale_factor multiplies this to reach print DPI:
# scale 2 -> 1588px wide (~192 DPI), scale 3 -> ~288 DPI.
A4_WIDTH_CSS_PX = 794
A4_HEIGHT_CSS_PX = 1123


def build_env() -> Environment:
    """StrictUndefined so a template typo fails loudly instead of rendering ''."""
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


class Renderer:
    """Context manager holding one Chromium instance across many renders.

        with Renderer() as r:
            for report in reports:
                r.render(report, "classic_bordered.html.j2", out_png)
    """

    def __init__(self, scale: int = 2) -> None:
        self.scale = scale
        self.env = build_env()
        self._pw = None
        self._browser = None
        self._page = None

    def __enter__(self) -> "Renderer":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        self._page = self._browser.new_page(
            viewport={"width": A4_WIDTH_CSS_PX, "height": A4_HEIGHT_CSS_PX},
            device_scale_factor=self.scale,
        )
        return self

    def __exit__(self, *exc) -> None:
        for closer in (self._page, self._browser):
            if closer is not None:
                closer.close()
        if self._pw is not None:
            self._pw.stop()

    def render_html(self, report: Report, template: str) -> str:
        return self.env.get_template(template).render(r=report)

    def render(self, report: Report, template: str, out_png: Path) -> Path:
        if self._page is None:
            raise RuntimeError("Renderer must be used as a context manager")
        html = self.render_html(report, template)
        # 'load' rather than 'networkidle': the templates reference no external
        # resources, and networkidle would add a fixed ~500ms wait per report.
        self._page.set_content(html, wait_until="load")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        self._page.screenshot(path=str(out_png), full_page=True)
        return out_png


def available_templates() -> list[str]:
    return sorted(p.name for p in TEMPLATE_DIR.glob("*.html.j2"))
