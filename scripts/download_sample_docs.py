"""Downloads the 5 public arXiv papers used for testing and creates the synthetic Acme report."""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sample_report import make  # noqa: E402

PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"
PAPERS = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
    "rag_lewis_2020.pdf": "https://arxiv.org/pdf/2005.11401",
    "llama2_technical_report.pdf": "https://arxiv.org/pdf/2307.09288",
    "rag_survey_gao_2023.pdf": "https://arxiv.org/pdf/2312.10997",
    "lora_hu_2021.pdf": "https://arxiv.org/pdf/2106.09685",
}

if __name__ == "__main__":
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in PAPERS.items():
        dest = PDF_DIR / name
        if dest.exists() and dest.stat().st_size > 50_000:
            print("exists", name)
            continue
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (research-rag-assignment)"}, timeout=90)
        if r.content[:4] == b"%PDF":
            dest.write_bytes(r.content)
            print("downloaded", name)
        else:
            print("not a PDF, skipped", name)
    make(PDF_DIR / "acme_annual_report.pdf")
    print("created acme_annual_report.pdf")
