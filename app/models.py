"""Model loading: embeddings, reranker and the open-source LLM (heavy imports stay inside functions)."""
from __future__ import annotations

import os

from .config import Settings


def get_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embeddings(cfg: Settings):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=cfg.embed_model,
                                 model_kwargs={"device": get_device()},
                                 encode_kwargs={"normalize_embeddings": True})


def load_reranker(cfg: Settings):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(cfg.rerank_model, device=get_device())


class LocalLLM:
    """Qwen2.5 instruct model with two methods: chat() and yes_prob()."""

    def __init__(self, cfg: Settings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = cfg.max_new_tokens
        cuda = torch.cuda.is_available()
        # GPU: float16. CPU: bfloat16 (half the RAM of float32). Override with LLM_DTYPE=float32 if you have plenty of RAM.
        dtype = getattr(torch, os.getenv("LLM_DTYPE", "float16" if cuda else "bfloat16"))
        self.tok = AutoTokenizer.from_pretrained(cfg.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": 0} if cuda else None)
        self.model.eval()
        self._ids: dict | None = None

    def _inputs(self, messages):
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tok(prompt, return_tensors="pt").to(self.model.device)

    def chat(self, messages, max_new_tokens: int | None = None) -> str:
        inputs = self._inputs(messages)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None, top_k=None,
                                      repetition_penalty=1.05)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def yes_prob(self, messages) -> float:
        """P(YES) from the first-token logits: a score that can be thresholded."""
        if self._ids is None:
            first = lambda w: self.tok.encode(w, add_special_tokens=False)[0]
            self._ids = {"yes": sorted({first(w) for w in ("YES", "Yes", "yes")}),
                         "no": sorted({first(w) for w in ("NO", "No", "no")})}
        inputs = self._inputs(messages)
        with self.torch.no_grad():
            # logits_to_keep=1: score only the last position (the full prompt x vocab matrix ran out of memory)
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        yes = self.torch.logsumexp(logits[self._ids["yes"]], 0)
        no = self.torch.logsumexp(logits[self._ids["no"]], 0)
        return self.torch.sigmoid(yes - no).item()