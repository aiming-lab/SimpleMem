"""LongMemEval adapter (ICLR 2025, arXiv 2410.10813).

Data: data/longmemeval/longmemeval_{oracle,s,m}.json — 500 questions each.
  - oracle: only the gold sessions present (easiest)
  - s: ~50 sessions haystack
  - m: ~500 sessions haystack

Schema: list of records, each:
  {question_id, question_type, question, answer, question_date,
   haystack_dates, haystack_session_ids, haystack_sessions, answer_session_ids}

  haystack_sessions: list of session-transcripts, each a list of
      {role: user|assistant, content, has_answer}

Each record maps to one BenchmarkSample (sessions from its own haystack,
one QA pair).

Scoring: token-level F1 against gold answer.
  (LongMemEval paper uses GPT-4-judge; we implement both — default F1 for
   cheap iteration, LLM-judge available when needed.)

Question-type -> canonical category mapping:
  temporal-reasoning -> 2 (Temporal)
  multi-session     -> 3 (MultiHop)
  knowledge-update  -> 4 (OpenDomain-ish; knowledge updates over time)
  single-session-user        -> 1 (SingleHop)
  single-session-assistant   -> 1
  single-session-preference  -> 1
"""

from __future__ import annotations

import json
from typing import Any

from .base import (
    BenchmarkAdapter,
    BenchmarkSample,
    QuestionMeta,
    token_f1,
)
from .metrics import compute_text_metrics, llm_judge

_QTYPE_TO_CAT = {
    "temporal-reasoning": 2,
    "multi-session": 3,
    "knowledge-update": 4,
    "single-session-user": 1,
    "single-session-assistant": 1,
    "single-session-preference": 1,
}


def _turns_from_session(session: list[dict]) -> list[dict]:
    """Convert LongMemEval session turns to EvolveMem's {speaker, text}."""
    out = []
    for t in session:
        role = t.get("role", "user")
        speaker = "User" if role == "user" else "Assistant"
        out.append({
            "speaker": speaker,
            "text": t.get("content", ""),
            "has_answer": t.get("has_answer", False),
        })
    return out


class LongMemEvalAdapter(BenchmarkAdapter):
    name = "longmemeval"
    primary_metric = "f1"
    subcategory_keys = ("qtype",)

    # Optionally attach an llm_call at run-time to enable llm_judge in score_all
    llm_judge_call: Any = None

    def score_all(self, prediction: str, reference: str, qa: dict) -> dict:
        m = compute_text_metrics(prediction, str(reference), want_sbert=False)
        if self.llm_judge_call is not None:
            v = llm_judge(
                prediction, str(reference),
                qa.get("question", ""), self.llm_judge_call,
            )
            if v is not None:
                m["llm_judge"] = v
        return m

    def load(
        self,
        path: str,
        *,
        max_samples: int | None = None,
        qtype_filter: list[str] | None = None,
        stratify: bool = False,
        seed: int = 42,
    ) -> list[BenchmarkSample]:
        with open(path) as f:
            raw = json.load(f)

        if qtype_filter:
            raw = [r for r in raw if r.get("question_type") in qtype_filter]

        if max_samples is not None:
            if stratify:
                # Evenly distribute across qtypes; the oracle file is sorted
                # by qtype so naive head-slice biases to one class.
                import random
                rng = random.Random(seed)
                from collections import defaultdict
                by_q: dict[str, list] = defaultdict(list)
                for r in raw:
                    by_q[r.get("question_type", "")].append(r)
                per = max(1, max_samples // max(1, len(by_q)))
                picked: list = []
                for q, items in by_q.items():
                    rng.shuffle(items)
                    picked.extend(items[:per])
                # Top up to exactly max_samples if needed (extras from largest qtypes)
                extras_pool = [r for q, items in by_q.items() for r in items[per:]]
                rng.shuffle(extras_pool)
                if len(picked) < max_samples:
                    picked.extend(extras_pool[: max_samples - len(picked)])
                raw = picked[:max_samples]
            else:
                raw = raw[:max_samples]

        samples: list[BenchmarkSample] = []
        for r in raw:
            sessions_raw = r.get("haystack_sessions", [])
            session_ids = r.get("haystack_session_ids", [])
            session_dates = r.get("haystack_dates", [])

            sessions = []
            for idx, ses in enumerate(sessions_raw):
                sid = session_ids[idx] if idx < len(session_ids) else f"sess_{idx}"
                date_str = session_dates[idx] if idx < len(session_dates) else ""
                sessions.append((sid, date_str, _turns_from_session(ses)))

            qtype = r.get("question_type", "")
            qa_pairs = [{
                "question": r["question"],
                "answer": r.get("answer", ""),
                "category": _QTYPE_TO_CAT.get(qtype, 0),
                "meta": QuestionMeta(
                    qid=r.get("question_id", ""),
                    qtype=qtype,
                    extras={
                        "question_date": r.get("question_date", ""),
                        "answer_session_ids": r.get("answer_session_ids", []),
                    },
                ).__dict__,
            }]

            samples.append(BenchmarkSample(
                sample_id=r.get("question_id", ""),
                sessions=sessions,
                qa_pairs=qa_pairs,
                benchmark=self.name,
                metadata={"qtype": qtype, "n_sessions": len(sessions)},
            ))
        return samples

    def score(self, prediction: str, reference: str, qa: dict) -> float:
        return token_f1(prediction, reference)

    def build_answer_prompt(
        self,
        question: str,
        context: str,
        qa: dict,
    ) -> tuple[str, str]:
        meta = qa.get("meta", {}) or {}
        qtype = meta.get("qtype", "")
        extras = meta.get("extras", {}) if isinstance(meta, dict) else {}
        question_date = (extras or {}).get("question_date", "")

        system = (
            "You are a personal memory assistant answering a user's question "
            "based on past conversation snippets. Answer concisely and "
            "specifically. JSON output only."
        )
        date_prefix = (
            f"(The question is asked on {question_date}.)\n\n" if question_date else ""
        )
        if qtype == "temporal-reasoning":
            rule = (
                "This is a TEMPORAL question. Pay attention to dates in the "
                "context. Prefer the most recent relevant event."
            )
        elif qtype == "knowledge-update":
            rule = (
                "This is a KNOWLEDGE-UPDATE question. The user's situation "
                "may have changed over time — use the LATEST stated fact."
            )
        elif qtype == "multi-session":
            rule = (
                "This is a MULTI-SESSION AGGREGATION question. It typically "
                "asks HOW MANY / HOW MUCH / HOW OLD — the answer is almost "
                "always a single number. Carefully enumerate the relevant "
                "items across ALL sessions (do NOT double-count, do NOT miss "
                "any), then output ONLY the final integer or a compact numeric "
                "phrase. Examples of correct forms: '3', '43', '33 years'. "
                "Do NOT write prose like 'Three plants' or 'about 42' — "
                "write digits. If the question asks a count, output the count "
                "as an Arabic numeral only."
            )
        elif qtype == "single-session-preference":
            rule = (
                "Report the USER'S stated preference exactly as they expressed "
                "it (e.g., a color, a brand name, a style)."
            )
        else:
            rule = (
                "Answer using exact facts from the context. Be specific."
            )

        user = (
            f"{date_prefix}"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            f"Rules:\n"
            f"1. {rule}\n"
            f"2. Answer in 1-15 words. Use EXACT words/phrases from context.\n"
            f"3. If the context genuinely lacks the answer, still give a best "
            f"inference — do not say 'not mentioned'.\n"
            f"Return JSON: {{\"reasoning\":\"brief\",\"answer\":\"concise\"}}"
        )
        return system, user
