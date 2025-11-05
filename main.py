from pathlib import Path
from solver import solve_spellingbee

LETTERS = "AESLTRN"   # any order; 7 distinct letters
CENTER  = "A"
WORDLIST = Path("wordlist.txt")  # your dictionary file
TOP_K = 20  # how many “best options” to show


def print_top(results, k=20):
    print("=== Top candidates ===")
    for r in results[:k]:
        tag = "  (PANGRAM)" if r.is_pangram else ""
        print(f"{r.word.upper():<20}  len={r.length:<2}  score={r.score:<2}{tag}")
    print()


def print_summary(results):
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
    results = solve_spellingbee(
        letters=list(LETTERS),
        center=CENTER,
        wordlist_path=WORDLIST,
        min_len=4,
        pangram_bonus=7,
        min_zipf=3.6,   # keep only common words (higher = stricter)
        max_len=10,     # ignore very long words
    )

    print_summary(results)
    print_top(results, TOP_K)
