"""Creates a small synthetic annual report with a table (to test numbers and table extraction)."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def make(dest: Path) -> None:
    st = getSampleStyleSheet()
    table = Table([["Metric (INR crore)", "2021", "2022", "2023"],
                   ["Revenue", "1,820", "2,145", "2,510"],
                   ["Net profit", "212", "268", "331"],
                   ["EBITDA margin", "14.2%", "15.1%", "16.0%"]])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    intro = ("Acme Industries manufactures industrial sensors and is headquartered in Pune. "
             "The Chief Executive Officer is Meera Rao. The company had 4,200 employees at year end. ") * 2
    SimpleDocTemplate(str(dest), pagesize=A4).build([
        Paragraph("Acme Industries Annual Report 2023", st["Title"]), Paragraph(intro, st["Normal"]), PageBreak(),
        Paragraph("Consolidated financial summary", st["Heading2"]), table, Spacer(1, 12),
        Paragraph("Revenue grew 17 percent in 2023 driven by exports.", st["Normal"])])


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "pdfs" / "acme_annual_report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    make(out)
    print("created", out)
