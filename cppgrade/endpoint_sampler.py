"""Sample from an OpenAI-compatible endpoint (SGLang/vLLM on Modal) as if it were a Tinker SamplingClient.

Same rendered prompt as the Tinker path (decoded to text, the server re-tokenizes it), same renderer
parsing on the way back, so results are comparable with Tinker-sampled evals.

    sampler = EndpointSampler(url, api_key, tokenizer, stop_token_ids=renderer.get_stop_sequences())
    resp = await sampler.sample(prompt, num_samples=2, sampling_params=params)   # like tinker
    for seq in resp.sequences: seq.tokens, seq.stop_reason                       # "stop" | "length"
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx


@dataclass
class _Seq:
    tokens: list[int]
    stop_reason: str


@dataclass
class _Resp:
    sequences: list[_Seq]


class EndpointSampler:
    def __init__(self, url: str, api_key: str, tokenizer, stop_token_ids: list[int], model: str = "model",
                 timeout_s: float = 3600, retries: int = 3):
        self.url = url.rstrip("/") + "/v1/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.tok = tokenizer
        self.stop_ids = [t for t in stop_token_ids if isinstance(t, int)]
        self.stop_strs = [self.tok.decode([t]) for t in self.stop_ids]
        self.model = model
        self.retries = retries
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def sample(self, prompt, num_samples: int, sampling_params):
        body = {
            "model": self.model,
            "prompt": self.tok.decode(prompt.to_ints()),
            "n": num_samples,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "stop": self.stop_strs,
            "skip_special_tokens": False,
        }
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = await self.client.post(self.url, json=body, headers=self.headers)
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:300]}", request=r.request, response=r)
                r.raise_for_status()
                seqs = []
                for ch in r.json()["choices"]:
                    toks = self.tok.encode(ch["text"], add_special_tokens=False)
                    if ch.get("finish_reason") == "stop" and self.stop_ids:
                        toks.append(self.stop_ids[0])  # the renderer expects the stop token to see a clean end
                    seqs.append(_Seq(toks, "length" if ch.get("finish_reason") == "length" else "stop"))
                return _Resp(seqs)
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                if isinstance(e, httpx.HTTPStatusError) and e.response is not None and e.response.status_code < 500:
                    raise
                await asyncio.sleep(10 * (attempt + 1))
        raise RuntimeError(f"endpoint failed after {self.retries + 1} attempts: {last}")
