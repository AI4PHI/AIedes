#!/usr/bin/env python3
"""
render_figure.py
----------------
Renders concept_model_panel.html → concept_model_panel.pdf
using a headless Chromium browser via playwright.

Requirements:
    pip install playwright
    playwright install chromium

Usage:
    python render_figure.py [--html PATH] [--out PATH] [--width MM] [--height MM] [--scale FLOAT]

Defaults:
    --html   concept_model_panel.html   (same directory as this script)
    --out    concept_model_panel.pdf    (same directory as this script)
    --width  260                        (mm, fits comfortably on A4/letter)
    --height auto                       (fits content height automatically)
    --scale  1.8                        (device scale factor → sharpness)
"""

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Render HTML figure to PDF via headless Chromium.")
    p.add_argument("--html",   default=None,  help="Path to input HTML file")
    p.add_argument("--out",    default=None,  help="Path to output PDF file")
    p.add_argument("--width",  type=float, default=260,  help="Page width in mm (default: 260)")
    p.add_argument("--height", type=float, default=None, help="Page height in mm (auto if omitted)")
    p.add_argument("--scale",  type=float, default=1.8,  help="Device scale factor (default: 1.8)")
    p.add_argument("--no-wait-fonts", action="store_true",
                   help="Skip waiting for Google Fonts (use if offline)")
    return p.parse_args()


def mm_to_px(mm, dpi=96):
    return mm * dpi / 25.4


def main():
    args = parse_args()

    # Resolve paths relative to this script
    script_dir = Path(__file__).parent.resolve()
    html_path  = Path(args.html)  if args.html else script_dir / "concept_model_panel.html"
    out_path   = Path(args.out)   if args.out  else script_dir / "concept_model_panel.pdf"

    if not html_path.exists():
        print(f"[ERROR] HTML file not found: {html_path}", file=sys.stderr)
        sys.exit(1)

    # ── import playwright ──────────────────────────────────────────────────────
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        print(
            "[ERROR] playwright is not installed.\n"
            "  Run:  pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        sys.exit(1)

    file_url = html_path.as_uri()
    print(f"[INFO] Input  : {html_path}")
    print(f"[INFO] Output : {out_path}")
    print(f"[INFO] Width  : {args.width} mm")
    print(f"[INFO] Scale  : {args.scale}x")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--disable-web-security", "--no-sandbox"],
        )

        # Viewport width matches the requested paper width at 96 dpi
        vp_width  = int(mm_to_px(args.width) * args.scale)
        vp_height = 900  # initial guess; will be adjusted

        page = browser.new_page(
            viewport={"width": vp_width, "height": vp_height},
            device_scale_factor=args.scale,
        )

        page.goto(file_url, wait_until="domcontentloaded")

        # Wait for Google Fonts (networkidle or timeout)
        if not args.no_wait_fonts:
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except PWTimeout:
                print("[WARN] Network idle timeout — fonts may be loading from cache or offline. Continuing.")

        # Extra settle time for font rendering
        time.sleep(0.5)

        # Measure actual content height
        content_height_px = page.evaluate(
            "() => document.documentElement.scrollHeight"
        )
        print(f"[INFO] Content height: {content_height_px}px")

        # Compute page height in mm (with a tiny margin)
        if args.height:
            page_height_mm = args.height
        else:
            page_height_mm = (content_height_px / args.scale) * 25.4 / 96 + 4

        print(f"[INFO] Page height: {page_height_mm:.1f} mm")

        # Generate PDF
        pdf_bytes = page.pdf(
            width=f"{args.width}mm",
            height=f"{page_height_mm:.2f}mm",
            print_background=True,
            margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"},
        )

        browser.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pdf_bytes)
    size_kb = out_path.stat().st_size / 1024
    print(f"[OK]  Written {size_kb:.1f} KB → {out_path}")


if __name__ == "__main__":
    main()
