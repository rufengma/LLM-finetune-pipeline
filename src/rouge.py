"""Dependency-free ROUGE scoring for the summarization subset of eval.

ROUGE-N (n = 1, 2) counts overlapping n-grams between a candidate summary
and its reference; ROUGE-L uses the longest common subsequence. All three
are reported as precision / recall / F1, following the standard definitions
used by the `rouge-score` package, but implemented here in pure stdlib so
evaluation does not gain another heavy dependency.

The summarization subset is identified by instruction text: seed tasks have
no task-type label, so an instruction counts as a summarization task when
it contains a word starting with "summar" (summarize / summary /
summarization). This is deliberately narrow -- a generic "explain" or
"rewrite" task is not a summary task, and scoring it as one would inflate
the metric with tasks ROUGE was never meant for.
"""

import re
from collections import Counter

_SUMMARIZE_RE = re.compile(r"\bsummar\w*\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    """Lowercase word tokens; punctuation and case are ignored."""
    return _TOKEN_RE.findall(text.lower())


def is_summarization(instruction: str) -> bool:
    """True when the instruction asks for a summary."""
    return bool(_SUMMARIZE_RE.search(instruction or ""))


def summarization_subset(examples) -> list:
    """Filter an iterable of {"instruction", ...} dicts to summary tasks."""
    return [ex for ex in examples if is_summarization(ex.get("instruction", ""))]


def _ngrams(tokens: list, n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _prf(overlap: int, cand_total: int, ref_total: int) -> tuple:
    """Precision, recall, F1 from overlap counts (0.0s when undefined)."""
    precision = overlap / cand_total if cand_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def rouge_n(candidate: str, reference: str, n: int) -> tuple:
    """ROUGE-N (precision, recall, F1) for one candidate/reference pair."""
    cand = _ngrams(tokenize(candidate), n)
    ref = _ngrams(tokenize(reference), n)
    overlap = sum((cand & ref).values())
    return _prf(overlap, sum(cand.values()), sum(ref.values()))


def _lcs_length(a: list, b: list) -> int:
    """Length of the longest common subsequence (token-level, DP)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> tuple:
    """ROUGE-L (precision, recall, F1) for one candidate/reference pair."""
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    lcs = _lcs_length(cand_tokens, ref_tokens)
    return _prf(lcs, len(cand_tokens), len(ref_tokens))


def score_rouge(candidate: str, reference: str) -> dict:
    """All three ROUGE F1 scores (plus P/R) for one candidate/reference pair."""
    scores = {}
    for name, fn in (("rouge1", lambda c, r: rouge_n(c, r, 1)),
                     ("rouge2", lambda c, r: rouge_n(c, r, 2)),
                     ("rougeL", rouge_l)):
        precision, recall, f1 = fn(candidate, reference)
        scores[name] = {"precision": precision, "recall": recall, "f1": f1}
    return scores


def mean_rouge(per_example_scores: list) -> dict:
    """Average each metric over a list of score_rouge() results."""
    if not per_example_scores:
        return {}
    keys = per_example_scores[0].keys()
    return {
        name: {
            metric: sum(s[name][metric] for s in per_example_scores) / len(per_example_scores)
            for metric in ("precision", "recall", "f1")
        }
        for name in keys
    }
