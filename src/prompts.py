from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

from .utils import slugify


@dataclass
class PromptEntry:
    index: int
    template: str
    formatted: str
    slug: str


@dataclass
class PromptSet:
    subject_identifier: str
    style_prefix: str
    prompts: List[PromptEntry]

    @property
    def as_serializable(self):
        return {
            "subject_identifier": self.subject_identifier,
            "style_prefix": self.style_prefix,
            "prompts": [
                {
                    "index": p.index,
                    "template": p.template,
                    "formatted": p.formatted,
                    "slug": p.slug,
                }
                for p in self.prompts
            ],
        }


def load_prompt_set(path: str | Path) -> PromptSet:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    subject_identifier = data["subject_identifier"]
    style_prefix = data.get("style_prefix", "").strip()

    prompts = []
    for idx, template in enumerate(data.get("positive_prompts", [])):
        formatted = template.format(subject_identifier=subject_identifier)
        if style_prefix:
            formatted = f"{style_prefix}, {formatted}"
        prompts.append(
            PromptEntry(
                index=idx,
                template=template,
                formatted=formatted,
                slug=slugify(formatted),
            )
        )

    return PromptSet(subject_identifier=subject_identifier, style_prefix=style_prefix, prompts=prompts)
