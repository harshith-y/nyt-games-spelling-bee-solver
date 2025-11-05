#!/usr/bin/env python3
"""
NYT Spelling Bee Solver (curated)
- Letters can be reused.
- Must include center.
- >= 4 letters.
- Only from the 7 letters.
Scoring:
- len=4 => 1 point, else length
- Pangram (+7)

Extras:
- min_zipf: filter rare words using wordfreq Zipf scores (higher = more common)
- max_len: drop very long words (NYT list rarely includes >9–10)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Iterable, Tuple, Optional
import re

# Optional dependency: wordfreq
try:
    from wordfreq import zipf_frequency  # type: ignore
except Exception:
    zipf_frequency = None  # gracefully degrade if not installed

@dataclass(frozen=True)
class WordResult:
    word: str
    score: int
    is_pangram: bool
    length: int

_WORD_RE = re.compile(r"^[a-z]+$")

def _load_words(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if _WORD_RE.match(w):
                yield w

def _is_valid_word(word: str, letters: Sequence[str], center: str, min_len: int = 4) -> bool:
    if len(word) < min_len:
        return False
    if center not in word:
        return False
    return set(word).issubset(set(letters))

def _is_pangram(word: str, letters: Sequence[str]) -> bool:
    return set(word).issuperset(set(letters))

def _score_word(word: str, letters: Sequence[str], pangram_bonus: int = 7) -> Tuple[int, bool]:
    is_pg = _is_pangram(word, letters)
    base = 1 if len(word) == 4 else len(word)
    return base + (pangram_bonus if is_pg else 0), is_pg

def solve_spellingbee(
    letters: Sequence[str],
    center: str,
    wordlist_path: Path | str,
    min_len: int = 4,
    pangram_bonus: int = 7,
    *,
    # NEW:
    min_zipf: Optional[float] = None,  # e.g., 3.5 (requires `wordfreq`)
    max_len: Optional[int] = 10,       # e.g., 9 or 10 to trim long forms
) -> List[WordResult]:
    """
    Returns sorted results by (score desc, word asc).

    min_zipf:
      - If provided and wordfreq is installed, keeps only words with
        zipf_frequency(word, 'en') >= min_zipf.
      - If wordfreq isn't installed, this parameter is ignored.

    max_len:
      - If provided, drop words longer than this length.
    """
    letters = [c.lower() for c in letters]
    center = center.lower()

    uniq_letters = sorted(set(letters))
    if len(uniq_letters) != 7:
        raise ValueError(f"Expected 7 distinct letters; got {len(uniq_letters)}")
    if center not in uniq_letters:
        raise ValueError(f"Center letter '{center}' must be among {uniq_letters}")

    results: List[WordResult] = []
    for word in _load_words(Path(wordlist_path)):
        # Basic legality
        if not _is_valid_word(word, uniq_letters, center, min_len=min_len):
            continue

        # Length cap (helps remove “alternateness”-style words)
        if max_len is not None and len(word) > max_len:
            continue

        # Frequency filter (if requested and available)
        if min_zipf is not None and zipf_frequency is not None:
            if zipf_frequency(word, "en") < min_zipf:
                continue

        score, is_pg = _score_word(word, uniq_letters, pangram_bonus)
        results.append(WordResult(word=word, score=score, is_pangram=is_pg, length=len(word)))

    results.sort(key=lambda r: (-r.score, r.word))
    return results

# Pretty printing helpers (unchanged)
def print_best(results: List[WordResult], n: int = 20) -> None:
    print(f"=== Top {n} words ===")
    for r in results[:n]:
        tag = " (PANGRAM)" if r.is_pangram else ""
        print(f"{r.word.upper():<20}  len={r.length:<2}  score={r.score:<2}{tag}")
    print()

def summarize(results: List[WordResult]) -> None:
    total_points = sum(r.score for r in results)
    n_pangrams = sum(1 for r in results if r.is_pangram)
    by_len = {}
    for r in results:
        by_len[r.length] = by_len.get(r.length, 0) + 1
    print("=== Summary ===")
    print(f"Words: {len(results)}   Total points: {total_points}   Pangrams: {n_pangrams}")
    print("By length:", dict(sorted(by_len.items())))
    print()

if __name__ == "__main__":
    # Sanity run
    letters = "AESLTRN"
    center = "A"
    wordlist = Path("wordlist.txt")
    results = solve_spellingbee(
        letters, center, wordlist,
        min_len=4,
        pangram_bonus=7,
        min_zipf=3.6,   # tweak 3.5–4.0
        max_len=10,     # 9–10 is a good cap
    )
    summarize(results)
    print_best(results, 20)
