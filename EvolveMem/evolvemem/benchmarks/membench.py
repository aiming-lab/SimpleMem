"""MemBench adapter (arXiv 2506.21605, ACL Findings 2025).

Data: data/membench/repo/MemData/{FirstAgent,ThirdAgent}/<category>.json
Each file: { topic: [ {tid, message_list, QA}, ... ] }
  - message_list: list of sessions, each a list of turns:
        {sid, user_message, assistant_message, time, place}
  - QA: single dict {qid, question, answer, target_step_id, choices, ground_truth, time}

Scoring: multiple-choice accuracy. `ground_truth` is the letter (A-D).

Categories (per paper):
  LowLevel (Participation-Factual):    simple, comparative, aggregative,
                                       conditional, knowledge_update,
                                       post_processing, noisy
  HighLevel (Participation-Reflective): highlevel
  Receptive (ThirdAgent overhearing):  same names under ThirdAgent/
  FirstAgent _rec variants             (optional, paper's receptive w/ history)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .base import (
    BenchmarkAdapter,
    BenchmarkSample,
    QuestionMeta,
    accuracy_with_choices,
)
from .metrics import compute_mcq_metrics

LOWLEVEL_CATS = [
    "simple", "comparative", "aggregative", "conditional",
    "knowledge_update", "post_processing", "noisy",
]
HIGHLEVEL_CATS = ["highlevel"]

# Canonical category id for diagnostics (small-int space distinct from LoCoMo's 1-5)
_CAT_ID = {
    "simple": 11,
    "comparative": 12,
    "aggregative": 13,
    "conditional": 14,
    "knowledge_update": 15,
    "post_processing": 16,
    "noisy": 17,
    "highlevel": 18,
    "highlevel_rec": 19,
    "lowlevel_rec": 20,
    "RecMultiSession": 21,
}


def _turns_from_session(session_raw: list[dict]) -> list[dict]:
    """A MemBench session stores paired user/assistant messages per turn.
    We split them into two EvolveMem turns per original turn.
    """
    turns = []
    for msg in session_raw:
        time_str = msg.get("time", "")
        place = msg.get("place", "")
        prefix = ""
        if time_str or place:
            prefix = f"[{time_str}" + (f" @ {place}" if place else "") + "] "
        if msg.get("user_message"):
            turns.append({
                "speaker": "User",
                "text": prefix + msg["user_message"],
            })
        if msg.get("assistant_message"):
            turns.append({
                "speaker": "Assistant",
                "text": prefix + msg["assistant_message"],
            })
    return turns


class MemBenchAdapter(BenchmarkAdapter):
    name = "membench"
    primary_metric = "accuracy"
    # Use qtype (string category name e.g. "simple"/"highlevel") not the int id
    subcategory_keys = ("agent", "qtype", "topic")

    def score_all(self, prediction: str, reference: str, qa: dict) -> dict:
        meta = qa.get("meta", {}) or {}
        extras = meta.get("extras", {}) if isinstance(meta, dict) else {}
        choices = (extras or {}).get("choices", {}) or {}
        gt = (extras or {}).get("ground_truth", "") or reference
        return compute_mcq_metrics(prediction, gt, choices)

    def load(
        self,
        path: str,
        *,
        agent: str = "FirstAgent",          # or "ThirdAgent"
        categories: list[str] | None = None,  # e.g. ["simple", "comparative"]
        topics: list[str] | None = None,      # filter to specific topics
        max_samples_per_file: int | None = None,
    ) -> list[BenchmarkSample]:
        """
        Args:
            path: base data dir, e.g. `data/membench/repo/MemData`.
            agent: `FirstAgent` or `ThirdAgent`.
            categories: subset of filenames (without `.json`); default = all 7
                        LowLevel categories (comparable with paper main table).
            topics: optional filter within each file's topic dict.
            max_samples_per_file: cap per-file for quick sanity runs.
        """
        agent_dir = os.path.join(path, agent)
        if not os.path.isdir(agent_dir):
            raise FileNotFoundError(agent_dir)

        if categories is None:
            categories = LOWLEVEL_CATS
        samples: list[BenchmarkSample] = []

        for cat in categories:
            fp = os.path.join(agent_dir, f"{cat}.json")
            if not os.path.exists(fp):
                continue
            with open(fp) as f:
                blob = json.load(f)
            for topic, records in blob.items():
                if topics and topic not in topics:
                    continue
                if max_samples_per_file is not None:
                    records = records[:max_samples_per_file]
                for rec in records:
                    tid = rec.get("tid", 0)
                    ml = rec.get("message_list", [])
                    sessions = []
                    for si, ses in enumerate(ml):
                        sid = f"{cat}_{topic}_{tid}_s{si}"
                        turns = _turns_from_session(ses)
                        # Use the first turn's time as the session date marker
                        date_str = ""
                        if ses and ses[0].get("time"):
                            date_str = ses[0]["time"]
                        sessions.append((sid, date_str, turns))

                    qa = rec.get("QA") or {}
                    if not qa:
                        continue
                    qa_pairs = [{
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "category": _CAT_ID.get(cat, 0),
                        "meta": QuestionMeta(
                            qid=f"{agent}/{cat}/{topic}/{tid}",
                            qtype=cat,
                            extras={
                                "topic": topic,
                                "agent": agent,
                                "choices": qa.get("choices", {}),
                                "ground_truth": qa.get("ground_truth", ""),
                                "target_step_id": qa.get("target_step_id", []),
                                "question_time": qa.get("time", ""),
                            },
                        ).__dict__,
                    }]
                    samples.append(BenchmarkSample(
                        sample_id=f"{agent}_{cat}_{topic}_{tid}",
                        sessions=sessions,
                        qa_pairs=qa_pairs,
                        benchmark=self.name,
                        metadata={
                            "agent": agent, "category": cat, "topic": topic,
                            "n_sessions": len(sessions),
                        },
                    ))
        return samples

    def score(self, prediction: str, reference: str, qa: dict) -> float:
        meta = qa.get("meta", {}) or {}
        extras = meta.get("extras", {}) if isinstance(meta, dict) else {}
        choices = (extras or {}).get("choices", {})
        gt_letter = (extras or {}).get("ground_truth", "") or reference
        return accuracy_with_choices(prediction, gt_letter, choices)

    def build_answer_prompt(
        self,
        question: str,
        context: str,
        qa: dict,
    ) -> tuple[str, str]:
        meta = qa.get("meta", {}) or {}
        extras = meta.get("extras", {}) if isinstance(meta, dict) else {}
        choices = (extras or {}).get("choices", {}) or {}
        qtype = meta.get("qtype", "") if isinstance(meta, dict) else ""
        choices_text = "\n".join(
            f"  {letter}) {text}" for letter, text in sorted(choices.items())
        )
        system = (
            "You are a memory-grounded multiple-choice question answerer. "
            "You MUST pick exactly ONE letter (A/B/C/D). JSON only."
        )

        cat_hint = ""
        if qtype == "aggregative":
            cat_hint = (
                "\nThis is an AGGREGATION/COUNTING question. Before picking "
                "an answer:\n"
                "  a) List EVERY entity in the context that matches the "
                "question's criteria.\n"
                "  b) Count them carefully.\n"
                "  c) Pick the option matching your count.\n"
                "Common trap: missing entities that appear in different "
                "sessions or under variant names.\n"
            )
        elif qtype == "knowledge_update":
            cat_hint = (
                "\nThis is a KNOWLEDGE UPDATE question. Facts may have "
                "changed during the conversation.\n"
                "  - Always use the MOST RECENT version of any fact.\n"
                "  - If the same attribute appears multiple times with "
                "different values, the LATEST one is correct.\n"
                "  - Pay attention to timestamps/session order.\n"
            )
        elif qtype == "post_processing":
            cat_hint = (
                "\nThis question involves DERIVED or TRANSFORMED facts.\n"
                "  - Look for modifications, computations, or inferences "
                "from the raw conversation data.\n"
                "  - Consider email domains, time calculations, address "
                "patterns, or other derived attributes.\n"
                "  - If the direct fact is not stated, infer it from "
                "related context details.\n"
            )
        elif qtype == "noisy":
            cat_hint = (
                "\nThe context may contain CONTRADICTORY or AMBIGUOUS info.\n"
                "  - Focus on the most CONSISTENT and DEFINITIVE statements.\n"
                "  - Prefer information stated as fact over hedged or "
                "uncertain mentions.\n"
                "  - If conflicting, prefer the more recent statement.\n"
            )
        elif qtype == "comparative":
            cat_hint = (
                "\nThis is a COMPARISON question.\n"
                "  - Identify both entities being compared.\n"
                "  - Find the relevant attribute for EACH entity.\n"
                "  - Compare directly, then pick the matching option.\n"
            )

        user = (
            f"Question: {question}\n\n"
            f"Options:\n{choices_text}\n\n"
            f"Context (memory snippets):\n{context}\n"
            f"{cat_hint}\n"
            f"Rules:\n"
            f"1. Pick EXACTLY one letter from {{A,B,C,D}}.\n"
            f"2. Base your answer on the context; if context is incomplete "
            f"still pick the most plausible option.\n"
            f"3. Think step-by-step in 'reasoning' before picking.\n"
            f"4. Return JSON: {{\"reasoning\":\"step-by-step\",\"answer\":\"X\"}} "
            f"where X is a single letter."
        )
        return system, user
