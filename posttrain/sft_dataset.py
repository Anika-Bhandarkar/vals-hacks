"""Cookbook dataset builder for our SFT JSONL files.

Differences from the cookbook's FromConversationFileBuilder:
  - several files, each with an optional cap (mix sources without editing files)
  - multi-turn conversations become one example per assistant turn (renderer.build_supervised_examples),
    so each turn is trained in the context the model really sees (earlier thinking removed)
  - examples longer than max_length are DROPPED, not truncated: truncating a long reasoning trace
    would cut off the code at the end and teach the model to stop mid-thought
"""

from __future__ import annotations

import json
import logging
import random

import chz
import datasets
import tinker
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

from posttrain.sft_data import read_jsonl, to_renderer_messages

logger = logging.getLogger(__name__)


@chz.chz
class SFTJsonlBuilder(ChatDatasetBuilder):
    files: tuple[str, ...]
    caps: tuple[int, ...] = ()  # per-file max examples (0 = all), aligned with `files`
    test_size: int = 0
    shuffle_seed: int = 0

    def _load(self) -> list[dict]:
        rows = []
        rng = random.Random(self.shuffle_seed)
        for i, path in enumerate(self.files):
            recs = list(read_jsonl(path))
            cap = self.caps[i] if i < len(self.caps) else 0
            if cap and len(recs) > cap:
                recs = rng.sample(recs, cap)
            for r in recs:
                rows.append({"messages_json": json.dumps(r["messages"]), "source": r.get("meta", {}).get("source", path)})
            logger.info("SFT data: %s -> %d conversations", path, len(recs))
        return rows

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        ds = datasets.Dataset.from_list(self._load()).shuffle(seed=self.shuffle_seed)
        if self.test_size and len(ds) > self.test_size:
            test_ds, train_ds = ds.take(self.test_size), ds.skip(self.test_size)
        else:
            train_ds, test_ds = ds, None
        renderer = self.renderer
        max_len = self.common_config.max_length
        dropped = {"n": 0}

        def flatmap(row: dict) -> list[tinker.Datum]:
            msgs = to_renderer_messages(json.loads(row["messages_json"]))
            out = []
            for model_input, weights in renderer.build_supervised_examples(msgs, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES):
                if max_len and model_input.length > max_len:
                    dropped["n"] += 1
                    continue
                out.append(datum_from_model_input_weights(model_input, weights, None, reduction="mean"))
            return out

        bs = self.common_config.batch_size
        train = SupervisedDatasetFromHFDataset(train_ds, batch_size=bs, flatmap_fn=flatmap)
        test = SupervisedDatasetFromHFDataset(test_ds, batch_size=len(test_ds), flatmap_fn=flatmap) if test_ds else None
        return train, test


def count_examples(files: list[str], model: str, renderer_name: str, max_length: int) -> dict:
    """Offline check: per-turn examples, token lengths, how many would be dropped. No Tinker calls."""
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    r = renderers.get_renderer(renderer_name, get_tokenizer(model), model_name=model)
    stats = {"conversations": 0, "examples": 0, "dropped_too_long": 0, "trained_tokens": 0, "max_len": 0}
    for path in files:
        for rec in read_jsonl(path):
            stats["conversations"] += 1
            for mi, w in r.build_supervised_examples(to_renderer_messages(rec["messages"]), train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES):
                if mi.length > max_length:
                    stats["dropped_too_long"] += 1
                    continue
                stats["examples"] += 1
                stats["trained_tokens"] += int((w > 0).sum())
                stats["max_len"] = max(stats["max_len"], mi.length)
    return stats
