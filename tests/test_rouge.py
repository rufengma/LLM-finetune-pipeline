"""Tests for src/rouge.py: tokenization, ROUGE-N/L, subset selection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rouge import (  # noqa: E402
    is_summarization,
    mean_rouge,
    rouge_l,
    rouge_n,
    score_rouge,
    summarization_subset,
    tokenize,
)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Plants Use Chlorophyll, to convert Sunlight!") == [
        "plants", "use", "chlorophyll", "to", "convert", "sunlight"]


def test_rouge_n_identical_is_perfect():
    p, r, f1 = rouge_n("the cat sat", "the cat sat", 1)
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_rouge_n_partial_overlap():
    # candidate shares 2 of its 3 unigrams and 2 of 3 reference unigrams
    p, r, f1 = rouge_n("the cat sat", "the cat ran", 1)
    assert abs(p - 2 / 3) < 1e-9 and abs(r - 2 / 3) < 1e-9
    assert abs(f1 - 2 / 3) < 1e-9


def test_rouge_n_no_overlap_is_zero():
    assert rouge_n("cats dogs", "birds fish", 1) == (0.0, 0.0, 0.0)


def test_rouge_n_counts_repeated_ngrams_once_per_occurrence():
    # "the the" appears twice in candidate, once in reference -> overlap 1
    p, r, f1 = rouge_n("the the cat", "the cat", 1)
    assert abs(p - 2 / 3) < 1e-9 and abs(r - 1.0) < 1e-9


def test_rouge_2_requires_adjacent_pairs():
    # same unigrams, different order -> ROUGE-1 perfect, ROUGE-2 zero
    p1, _, _ = rouge_n("cat sat", "sat cat", 1)
    p2, _, _ = rouge_n("cat sat", "sat cat", 2)
    assert p1 == 1.0 and p2 == 0.0


def test_rouge_n_empty_candidate_or_reference():
    assert rouge_n("", "some reference", 1) == (0.0, 0.0, 0.0)
    assert rouge_n("some candidate", "", 1) == (0.0, 0.0, 0.0)


def test_rouge_l_subsequence_not_substring():
    # LCS "cat sat" (2 of 3 tokens each side) despite the gap
    p, r, f1 = rouge_l("the cat sat down", "cat sat up")
    assert abs(p - 2 / 4) < 1e-9 and abs(r - 2 / 3) < 1e-9


def test_rouge_l_identical_is_perfect():
    assert rouge_l("a b c", "a b c") == (1.0, 1.0, 1.0)


def test_score_rouge_keys_and_perfect_case():
    scores = score_rouge("plants use chlorophyll", "plants use chlorophyll")
    assert set(scores) == {"rouge1", "rouge2", "rougeL"}
    for metric in scores.values():
        assert metric == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_mean_rouge_averages_f1():
    scores = [score_rouge("a b", "a b"), score_rouge("c d", "e f")]
    mean = mean_rouge(scores)
    assert mean["rouge1"]["f1"] == 0.5
    assert mean["rouge2"]["f1"] == 0.5
    assert mean["rougeL"]["f1"] == 0.5


def test_mean_rouge_empty_is_empty():
    assert mean_rouge([]) == {}


def test_is_summarization_variants():
    assert is_summarization("Summarize the following paragraph in one sentence.")
    assert is_summarization("Write a summary of this article.")
    assert is_summarization("Give a summarization of the plot.")
    assert not is_summarization("Explain what a REST API is.")
    assert not is_summarization("Classify the sentiment of this review.")
    assert not is_summarization("")


def test_summarization_subset_filters_by_instruction():
    examples = [
        {"instruction": "Summarize this complaint in under 20 words.", "response": "x"},
        {"instruction": "Explain photosynthesis.", "response": "y"},
        {"instruction": "Summarize the plot without spoilers.", "response": "z"},
    ]
    subset = summarization_subset(examples)
    assert [ex["response"] for ex in subset] == ["x", "z"]
