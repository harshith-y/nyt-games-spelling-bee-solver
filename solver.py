from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Iterable, Tuple, Optional
from collections import Counter
from functools import lru_cache
import re
import sys

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
    - max_len: drop very long words (NYT list rarely includes >9-10)
    - filter_plurals: exclude obvious plurals (words ending in S)
    - filter_obscure: additional heuristics for filtering uncommon words
"""

# ============================================================
# PUZZLE INPUT - CHANGE THESE FOR EACH NEW PUZZLE
# ============================================================
LETTERS = "AELNRST"    # All 7 letters in the puzzle
CENTER = "N"           # The required center letter

MIN_ZIPF = 3.6         # Word frequency threshold (3.5-4.5 recommended)
                       # Higher = fewer, more common words
SHOW_TOP_N = 50        # How many words to display
SHOW_HINTS = False     # Set to True for hints without spoilers
# ============================================================

try:
    from wordfreq import zipf_frequency
    WORDFREQ_AVAILABLE = True
except ImportError:
    zipf_frequency = None
    WORDFREQ_AVAILABLE = False

WORDLIST = Path(__file__).parent / "wordlist.txt"

@dataclass(frozen=True)
class WordResult:
    word: str
    score: int
    is_pangram: bool
    length: int
    zipf_score: float = 0.0

_WORD_RE = re.compile(r"^[a-z]+$")
EXCLUDE_PATTERNS = [
    r"^.{15,}$",  # extremely long words
    r".*[qxz]{2,}.*",  # multiple rare letters
]

@lru_cache(maxsize=50000)
def _get_zipf(word: str) -> float:
    """Cached Zipf frequency lookup for performance"""
    if zipf_frequency is None:
        return 0.0
    return zipf_frequency(word, "en")

def _load_words(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if _WORD_RE.match(w):
                yield w

def _is_likely_plural(word: str) -> bool:
    """Filter obvious plurals that NYT typically excludes."""
    if not word.endswith('s') or len(word) <= 4:
        return False
    if word.endswith('ss'):
        return False
    non_plural_endings = ['ous', 'ness', 'less', 'ious', 'us']
    for ending in non_plural_endings:
        if word.endswith(ending):
            return False
    return True

def _is_obscure_word(word: str) -> bool:
    """Additional heuristics to filter uncommon words"""
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, word):
            return True
    if len(word) >= 6:
        for char in set(word):
            if word.count(char) >= len(word) * 0.6:
                return True
    return False

def _is_valid_word(
    word: str, 
    letters_set: set, 
    center: str, 
    min_len: int = 4,
    filter_plurals: bool = True,
    filter_obscure: bool = True
) -> bool:
    if len(word) < min_len:
        return False
    if center not in word:
        return False
    if not set(word).issubset(letters_set):
        return False
    if filter_plurals and _is_likely_plural(word):
        return False
    if filter_obscure and _is_obscure_word(word):
        return False
    return True

def _is_pangram(word: str, letters_set: set) -> bool:
    return set(word).issuperset(letters_set)

def _score_word(word: str, letters_set: set, pangram_bonus: int = 7) -> Tuple[int, bool]:
    is_pg = _is_pangram(word, letters_set)
    base = 1 if len(word) == 4 else len(word)
    return base + (pangram_bonus if is_pg else 0), is_pg

def solve_spellingbee(
    letters: Sequence[str],
    center: str,
    wordlist_path: Path | str,
    min_len: int = 4,
    pangram_bonus: int = 7,
    *,
    min_zipf: Optional[float] = None,
    max_len: Optional[int] = 10,
    filter_plurals: bool = True,
    filter_obscure: bool = True,
    use_aggressive_filtering: bool = False,
) -> List[WordResult]:
    """Returns sorted results by (score desc, word asc)."""
    letters = [c.lower() for c in letters]
    center = center.lower()

    uniq_letters = sorted(set(letters))
    uniq_letters_set = set(uniq_letters)
    
    if len(uniq_letters) != 7:
        raise ValueError(f"Expected 7 distinct letters; got {len(uniq_letters)}")
    if center not in uniq_letters:
        raise ValueError(f"Center letter '{center}' must be among {uniq_letters}")

    results: List[WordResult] = []
    
    for word in _load_words(Path(wordlist_path)):
        if not _is_valid_word(word, uniq_letters_set, center, min_len, filter_plurals, filter_obscure):
            continue

        if max_len is not None and len(word) > max_len:
            continue

        zipf_score = 0.0
        if min_zipf is not None and zipf_frequency is not None:
            zipf_score = _get_zipf(word)
            if zipf_score < min_zipf:
                continue
            
            if use_aggressive_filtering:
                if len(word) >= 8 and zipf_score < min_zipf + 0.3:
                    continue
                if len(word) >= 9 and zipf_score < min_zipf + 0.5:
                    continue

        score, is_pg = _score_word(word, uniq_letters_set, pangram_bonus)
        results.append(WordResult(
            word=word, 
            score=score, 
            is_pangram=is_pg, 
            length=len(word),
            zipf_score=zipf_score
        ))

    results.sort(key=lambda r: (-r.score, r.word))
    return results

def print_hints(results: List[WordResult]) -> None:
    """Print hint statistics without revealing words"""
    by_first_letter = Counter(r.word[0].upper() for r in results)
    two_letter_starts = Counter(r.word[:2].upper() for r in results)
    
    print("=== Hints (Word Counts) ===")
    print("By first letter:", dict(sorted(by_first_letter.items())))
    print("\nMost common 2-letter starts:")
    for start, count in two_letter_starts.most_common(10):
        print(f"  {start}: {count} words")
    print()

def print_results(results: List[WordResult], show_top_n: int = 50, show_hints: bool = False):
    """Print comprehensive results"""
    total_points = sum(r.score for r in results)
    n_pangrams = sum(1 for r in results if r.is_pangram)
    by_len = Counter(r.length for r in results)
    
    print("=" * 60)
    print("=== Summary ===")
    print(f"Words found: {len(results)}")
    print(f"Total points available: {total_points}")
    print(f"Pangrams: {n_pangrams}")
    print(f"By length: {dict(sorted(by_len.items()))}")
    
    if results and results[0].zipf_score > 0:
        avg_zipf = sum(r.zipf_score for r in results) / len(results)
        print(f"Average word frequency (Zipf): {avg_zipf:.2f}")
    print()
    
    # Show hints if requested
    if show_hints:
        print_hints(results)
    
    # Show top words
    print(f"=== Top {min(show_top_n, len(results))} Words ===")
    for r in results[:show_top_n]:
        tag = " (PANGRAM)" if r.is_pangram else ""
        zipf_info = f"  zipf={r.zipf_score:.1f}" if r.zipf_score > 0 else ""
        print(f"{r.word.upper():<20}  len={r.length:<2}  score={r.score:<2}{tag}{zipf_info}")
    print()
    
    # Show all pangrams separately if any exist
    pangrams = [r for r in results if r.is_pangram]
    if pangrams:
        print("=== All Pangrams ===")
        for r in pangrams:
            zipf_info = f"  zipf={r.zipf_score:.1f}" if r.zipf_score > 0 else ""
            print(f"{r.word.upper():<20}  score={r.score}{zipf_info}")
        print()

def main():
    print("=" * 60)
    print("NYT Spelling Bee Solver")
    print("=" * 60)
    print(f"Letters: {LETTERS}")
    print(f"Center:  {CENTER}")
    print()
    
    # Validate configuration
    if len(set(LETTERS)) != 7:
        print(f"ERROR: You must have exactly 7 distinct letters.")
        print(f"You have: {len(set(LETTERS))} distinct letters in '{LETTERS}'")
        sys.exit(1)
    
    if CENTER not in LETTERS:
        print(f"ERROR: Center letter '{CENTER}' must be one of the 7 letters: {LETTERS}")
        sys.exit(1)
    
    if not WORDLIST.exists():
        print(f"ERROR: Wordlist not found at {WORDLIST.absolute()}")
        print("Make sure wordlist.txt is in the same directory as this script.")
        sys.exit(1)
    
    # Check wordfreq
    if not WORDFREQ_AVAILABLE:
        print("⚠ WARNING: wordfreq not installed - you'll see many obscure words!")
        print("  Install with: pip install wordfreq")
        print()
        use_zipf = None
    else:
        print("✓ Using word frequency filtering for quality results")
        print()
        use_zipf = MIN_ZIPF
    
    # Solve
    results = solve_spellingbee(
        letters=list(LETTERS),
        center=CENTER,
        wordlist_path=WORDLIST,
        min_len=4,
        pangram_bonus=7,
        min_zipf=use_zipf,
        max_len=10,
        filter_plurals=True,
        filter_obscure=True,
        use_aggressive_filtering=WORDFREQ_AVAILABLE,
    )
    
    # Display results
    print_results(results, show_top_n=SHOW_TOP_N, show_hints=SHOW_HINTS)
    
    # Tips
    print("=" * 60)
    if WORDFREQ_AVAILABLE:
        print("TIPS:")
        print(f"- Adjust MIN_ZIPF (currently {MIN_ZIPF}) for more/fewer words")
        print("- Set SHOW_HINTS=True to see counts without spoilers")
        print("- Change LETTERS and CENTER at the top for new puzzles")
    else:
        print("Install wordfreq for better results: pip install wordfreq")
    print("=" * 60)

if __name__ == "__main__":
    main()
