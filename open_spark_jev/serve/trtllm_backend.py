"""In-container backend on the TensorRT-LLM Python LLM API (PyTorch backend).

Use when you want the *full-vocabulary* first-token logits from the TRT-LLM engine rather
than the top-N logprobs the OpenAI chat endpoint exposes: exact label gathering for any menu
size, no ``floor_logprob`` approximation, and no HTTP hop per question. Runs inside
``nvcr.io/nvidia/tensorrt-llm/release:*`` (see deploy/spark/gateway_trtllm.sh), where the
gateway is started with ``--backend trtllm``.

Mechanics: one ``LLM`` instance with prefix caching on; for each question we submit the
rendered prompt with ``SamplingParams(max_tokens=1, return_generation_logits=True)`` and read
``outputs[0].generation_logits[0]`` (the logits that produced the first token) at the label
ids. All questions for a state are submitted in one ``generate`` call so the scheduler
batches them and the state prefix is reused from the KV block cache.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from ..model import Calibration
from ..prompting import LabelSpace, render_prompt
from ..schema import Answer, Question, State


class TRTLLMBackend:
    def __init__(self, model_dir: str, calibration: Optional[Calibration] = None, free_gpu_memory_fraction: float = 0.35,
                 max_batch_size: int = 64, max_num_tokens: int = 16384):
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig
        from transformers import AutoTokenizer

        self.name = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.labels = LabelSpace.build(self.tokenizer)
        self.calibration = calibration or Calibration()
        self.llm = LLM(
            model=model_dir,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=free_gpu_memory_fraction, enable_block_reuse=True),
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
        )
        self.sp = SamplingParams(max_tokens=1, temperature=0.0, return_generation_logits=True)

    def _label_logits(self, state: State, questions: Sequence[Question]) -> list[list[float]]:
        prompts = [render_prompt(state, q) for q in questions]
        outs = self.llm.generate(prompts, self.sp)
        result = []
        for q, o in zip(questions, outs):
            logits = o.outputs[0].generation_logits  # [num_generated=1, vocab] (torch tensor)
            row = logits[0] if logits.dim() == 2 else logits.reshape(-1, logits.shape[-1])[0]
            ids = self.labels.ids_for(len(q.labels))
            result.append([float(row[i]) for i in ids])
        return result

    def decide(self, state: State, questions: Sequence[Question], temperature: Optional[float] = None, return_logits: bool = False) -> list[Answer]:
        answers = []
        for q, z in zip(questions, self._label_logits(state, questions)):
            t = temperature if temperature is not None else self.calibration.t(q.type)
            m = max(z)
            e = [math.exp((x - m) / t) for x in z]
            s = sum(e)
            answers.append(Answer.from_probs(q, [x / s for x in e], raw_logits=z if return_logits else None))
        return answers
