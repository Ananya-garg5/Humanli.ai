"""Conversational RAG: condense follow-ups, ground answers, cite sources, refuse when unanswerable, keep memory."""
from __future__ import annotations

import re
import time

from .config import Settings
from .index import MULTI_RE, DocumentIndex

NOT_FOUND_MSG = "I could not find this information in the uploaded documents."

SYSTEM_PROMPT = f"""You are a precise document assistant. Answer ONLY using the numbered context passages.
Rules:
1. Cite the passages you used inline like [1] or [2][3].
2. If the context does not contain the answer, reply exactly: "{NOT_FOUND_MSG}"
3. Never use outside knowledge. Do not guess numbers.
4. Answer in 1-4 sentences. Do NOT quote long passages and do NOT mention page numbers.
5. For comparison questions, say what each document says. Name each document (see its source) and cite it.
6. Conversation history is only for understanding the question. Facts must come from the passages."""

ANSWER_TMPL = "{memory}Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer (with [n] citations):"

CONDENSE_TMPL = (
    "Rewrite the follow-up question as a standalone question. Only replace words like 'it', 'they', "
    "'that' using the conversation. Do NOT add topics or details that are not in the follow-up. "
    "If it is already standalone, return it unchanged. Output only the question.\n\n"
    "Example:\nUser: What is LoRA?\nAssistant: A low-rank adaptation method.\n"
    "Follow-up: How many parameters does it train?\nStandalone: How many parameters does LoRA train?\n\n"
    "{memory}Follow-up: {question}\nStandalone:")

VERIFY_TMPL = ("Passages:\n{context}\n\nQuestion: {question}\n\n"
               "Reply YES if the passages contain the answer to the question, otherwise reply NO.")

CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
REF_RE = re.compile(r"\b(it|its|they|them|their|this|that|these|those|former|latter|same|above|earlier|previous)\b", re.I)


def strip_cites(text: str) -> str:
    """Old [n] markers point at old passages, so drop them before saving to memory."""
    text = CITE_RE.sub("", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


class ConversationalRAG:
    """One instance per chat session. `llm` needs chat(messages, max_new_tokens) and yes_prob(messages)."""

    def __init__(self, index: DocumentIndex, llm, cfg: Settings):
        self.index, self.llm, self.cfg = index, llm, cfg
        self.history: list[tuple[str, str]] = []
        self.summary = ""

    def reset(self) -> None:
        self.history, self.summary = [], ""

    # ---------------- memory ----------------
    def _memory_text(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Summary of earlier conversation: {self.summary}\n")
        if self.history:
            turns = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in self.history)
            parts.append(f"Recent conversation:\n{turns}\n")
        return "\n".join(parts) + ("\n" if parts else "")

    def _condense(self, question: str) -> str:
        if not self.history and not self.summary:
            return question
        if not REF_RE.search(question) and len(question.split()) > 5:
            return question                      # nothing to resolve, skip the LLM call
        prompt = CONDENSE_TMPL.format(memory=self._memory_text(), question=question)
        out = self.llm.chat([{"role": "user", "content": prompt}], 80).split("\n")[0].strip()
        return out if 5 < len(out) < 400 else question

    def _remember(self, question: str, answer: str) -> None:
        self.history.append((question, strip_cites(answer)))
        while len(self.history) > self.cfg.keep_turns:
            old_q, old_a = self.history.pop(0)
            prompt = (f"Current summary: {self.summary or '(none)'}\n\nNew exchange:\nUser: {old_q}\n"
                      f"Assistant: {old_a}\n\nWrite an updated summary in at most 3 sentences, "
                      "keeping names, numbers and topics.")
            self.summary = self.llm.chat([{"role": "user", "content": prompt}], 120)

    # ---------------- answering ----------------
    def ask(self, question: str) -> dict:
        t0 = time.time()
        standalone = self._condense(question)
        hits = [(d, s) for d, s in self.index.retrieve(standalone) if s >= self.cfg.rerank_min_score]
        answer, sources, top, p_yes = NOT_FOUND_MSG, [], None, None

        if hits:                                 # no hits = nothing relevant, skip the LLM
            top = hits[0][1]
            context = "\n\n".join(
                f"[{i}] (source: {d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}"
                for i, (d, _) in enumerate(hits, 1))
            named = self.index.named_sources(standalone)
            multi = bool(MULTI_RE.search(standalone)) or len(named) > 1   # comparisons span documents
            p_yes = self.llm.yes_prob([{"role": "user", "content": VERIFY_TMPL.format(
                context=context, question=standalone)}])
            if multi or not (self.cfg.verify and p_yes < self.cfg.verify_min):   # check skipped for comparisons
                q_llm = standalone
                if len(named) > 1:               # comparison: make the model cover every document
                    q_llm += " Answer with one separate sentence for each of these documents: " + ", ".join(named) + "."
                user_msg = ANSWER_TMPL.format(memory=self._memory_text(), context=context, question=q_llm)
                answer = self.llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                                        {"role": "user", "content": user_msg}])
                cited = sorted({int(n) for m in CITE_RE.findall(answer) for n in re.split(r"\s*,\s*", m)
                                if 1 <= int(n) <= len(hits)})
                used = [hits[i - 1][0] for i in cited] if cited else [h[0] for h in hits[:3]]
                sources = sorted({(d.metadata["source"], d.metadata["page"]) for d in used})
                if NOT_FOUND_MSG.lower().rstrip(".") in answer.lower():
                    answer, sources = NOT_FOUND_MSG, []

        self._remember(question, answer)
        return dict(question=question, standalone=standalone, answer=answer,
                    sources=[{"file": f, "page": p} for f, p in sources],
                    grounded=bool(sources), top_score=top, p_yes=p_yes,
                    seconds=round(time.time() - t0, 1))
