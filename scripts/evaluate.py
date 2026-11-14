"""Runs eval/questions.json against the real models and writes eval/results.md.

kind: answerable  -> must cite every expected document and contain a keyword
      near        -> topic is in the documents but the answer is not; must be refused
      off         -> unrelated question; must be refused
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.engine import Engine  # noqa: E402
from app.rag import NOT_FOUND_MSG, ConversationalRAG  # noqa: E402


def main():
    cfg = Settings()
    eng = Engine.build(cfg)
    rows = []
    for e in json.loads((ROOT / "eval" / "questions.json").read_text()):
        r = ConversationalRAG(eng.index, eng.llm, cfg).ask(e["q"])
        refused = r["answer"] == NOT_FOUND_MSG
        row = dict(kind=e["kind"], q=e["q"], refused=refused, top=r["top_score"], p_yes=r["p_yes"],
                   sec=r["seconds"], answer=r["answer"].replace("\n", " ")[:90])
        if e["kind"] == "answerable":
            files = [s["file"] for s in r["sources"]]
            row["src_hit"] = all(any(x in f for f in files) for x in e["src"])
            row["kw_hit"] = any(k in r["answer"].lower() for k in e["kw"])
        rows.append(row)
        print(f"{e['kind']:<10} refused={refused!s:<5} {e['q'][:70]}")

    ans = [r for r in rows if r["kind"] == "answerable"]
    no = [r for r in rows if r["kind"] != "answerable"]
    near = [r for r in rows if r["kind"] == "near"]
    pct = lambda xs: f"{100 * sum(xs) / max(len(xs), 1):.0f}%"
    summary = [
        f"Answerable ({len(ans)}): source hit {pct([r['src_hit'] for r in ans])}, "
        f"keyword hit {pct([r['kw_hit'] for r in ans])}, wrongly refused {pct([r['refused'] for r in ans])}",
        f"Unanswerable ({len(no)}): correct refusals {pct([r['refused'] for r in no])} "
        f"(near-miss {pct([r['refused'] for r in near])})",
        f"Mean latency: {sum(r['sec'] for r in rows) / len(rows):.1f}s",
    ]
    for line in summary:
        print(line)

    # memory check: does a follow-up get rewritten with the right topic?
    rag = ConversationalRAG(eng.index, eng.llm, cfg)
    rag.ask("What is LoRA?")
    follow = rag.ask("How does it reduce trainable parameters?")
    summary.append(f"Memory check: {'ok' if 'lora' in follow['standalone'].lower() else 'FAILED'} "
                   f"(rewritten as: {follow['standalone']})")
    print(summary[-1])

    md = ["# Evaluation results", "", *[f"- {s}" for s in summary], "",
          "| kind | question | refused | rerank | p_yes | answer |", "|---|---|---|---|---|---|"]
    for r in rows:
        top = "-" if r["top"] is None else f"{r['top']:.1f}"
        py = "-" if r["p_yes"] is None else f"{r['p_yes']:.2g}"
        md.append(f"| {r['kind']} | {r['q']} | {r['refused']} | {top} | {py} | {r['answer']} |")
    (ROOT / "eval" / "results.md").write_text("\n".join(md) + "\n")
    print("saved eval/results.md")


if __name__ == "__main__":
    main()
