"""Export reports to PDF and PNG."""

from pathlib import Path

from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]


def export_to_pdf(html_content: str, output_path: str | Path) -> Path:
    """Export HTML report to PDF.

    Args:
        html_content: HTML report content
        output_path: Output file path

    Returns:
        Path to generated PDF
    """
    output_path = Path(output_path)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_content)
        page.wait_for_timeout(2000)
        page.pdf(path=str(output_path), format="A4", print_background=True)
        browser.close()

    return output_path


def export_to_png(html_content: str, output_path: str | Path, width: int = 1200) -> Path:
    """Export HTML report to PNG (for Twitter sharing).

    Args:
        html_content: HTML report content
        output_path: Output file path
        width: Screenshot width

    Returns:
        Path to generated PNG
    """
    output_path = Path(output_path)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": 800})
        page.set_content(html_content)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(output_path), full_page=True)
        browser.close()

    return output_path
