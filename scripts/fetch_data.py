"""Download replacement training data from Hugging Face / Universal Dependencies.

Brown (1961, edited American prose) is the wrong register for a typing
assistant: it has no modern vocabulary and only 1.1M tokens, so most trigram
contexts a user types have never been seen. This script builds three raw
files instead:

    data/raw/modern_english.txt   language-model corpus (Phases 1-3)
    data/raw/ud_english.conll     tagged English, web genre (Phase 4)
    data/raw/ud_hindi.conll       tagged Hindi (Phase 5)

Sources
    HuggingFaceFW/fineweb-edu (sample-10BT)  modern web prose, quality-filtered
    knkarthick/dialogsum                     messenger-style dialogue
    UD_English-EWT + UD_English-GUM          UPOS-tagged web/blog/review text
    UD_Hindi-HDTB                            UPOS-tagged Devanagari

    python scripts/fetch_data.py                      # everything, 6M tokens
    python scripts/fetch_data.py --target-tokens 12000000
    python scripts/fetch_data.py --skip-lm            # tagged corpora only
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

UD_ENGLISH = [
    ("UD_English-EWT", "en_ewt-ud-train.conllu"),
    ("UD_English-EWT", "en_ewt-ud-dev.conllu"),
    ("UD_English-GUM", "en_gum-ud-train.conllu"),
]
UD_HINDI = [
    ("UD_Hindi-HDTB", "hi_hdtb-ud-train.conllu"),
    ("UD_Hindi-HDTB", "hi_hdtb-ud-dev.conllu"),
]
UD_RAW = "https://raw.githubusercontent.com/UniversalDependencies/{repo}/master/{name}"
DIALOGSUM = "https://huggingface.co/datasets/knkarthick/dialogsum/resolve/main/train.csv"

# A line is kept only if it looks like a sentence somebody typed: sentence
# punctuation at the end, no markup, no navigation furniture, mostly letters.
BOILERPLATE = re.compile(
    r"(cookie|privacy policy|all rights reserved|click here|sign up|log in"
    r"|posted (on|by)|read more|©|\bhttps?://)", re.I
)
MARKUP = re.compile(r"[<>{}|\^~=*#_\[\]]")
SPEAKER = re.compile(r"#Person\d+#:\s*")
ENDS_OK = re.compile(r"[.!?\"')]$")


def keep_line(line: str, min_words: int = 4, max_words: int = 40) -> bool:
    words = line.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if not ENDS_OK.search(line) or MARKUP.search(line) or BOILERPLATE.search(line):
        return False
    letters = sum(ch.isalpha() for ch in line)
    return letters >= 0.7 * len(line)


def split_lines(text: str) -> list[str]:
    """Crude sentence-ish split. Phase 1 re-segments properly; this only
    decides what is worth writing to disk."""
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para))
    return out


def fetch_fineweb(target_tokens: int, handle) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets not installed -- run: pip install datasets")

    print(f"  fineweb-edu (streaming, target {target_tokens:,} tokens) ...")
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT",
        split="train", streaming=True,
    )
    written = 0
    for i, doc in enumerate(stream):
        for line in split_lines(doc["text"]):
            if keep_line(line):
                handle.write(line + "\n")
                written += len(line.split())
        if written >= target_tokens:
            break
        if i % 2000 == 0 and i:
            print(f"    {i:,} docs -> {written:,} tokens")
    return written


def fetch_dialogsum(handle) -> int:
    print("  dialogsum (messenger-style dialogue) ...")
    with urllib.request.urlopen(DIALOGSUM) as resp:
        data = resp.read().decode("utf-8", "replace")
    written = 0
    for row in csv.DictReader(io.StringIO(data)):
        for turn in row.get("dialogue", "").split("\n"):
            line = SPEAKER.sub("", turn).strip()
            if keep_line(line, min_words=3):
                handle.write(line + "\n")
                written += len(line.split())
    return written


def fetch_ud(targets, out_path: Path) -> int:
    sentences = 0
    tokens = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for repo, name in targets:
            url = UD_RAW.format(repo=repo, name=name)
            print(f"  {repo}/{name} ...")
            with urllib.request.urlopen(url) as resp:
                text = resp.read().decode("utf-8")
            empty = True
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    if not empty:
                        fh.write("\n")
                        sentences += 1
                        empty = True
                    continue
                if line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) < 4 or "-" in cols[0] or "." in cols[0]:
                    continue          # multiword range or empty node
                form, upos = cols[1], cols[3]
                if upos == "_":
                    continue
                fh.write(f"{form}\t{upos}\n")
                tokens += 1
                empty = False
            if not empty:
                fh.write("\n")
                sentences += 1
    print(f"    -> {out_path.name}: {sentences:,} sentences, {tokens:,} tokens")
    return tokens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-tokens", type=int, default=6_000_000,
                    help="fineweb-edu tokens to keep (RAM at fit time scales "
                         "with this; 6M ~ 2 GB peak, Brown was 1.1M)")
    ap.add_argument("--skip-lm", action="store_true")
    ap.add_argument("--skip-tagged", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    if not args.skip_lm:
        out = RAW / "modern_english.txt"
        print(f"Language-model corpus -> {out}")
        with out.open("w", encoding="utf-8") as fh:
            n = fetch_dialogsum(fh)
            print(f"    dialogsum: {n:,} tokens")
            n += fetch_fineweb(args.target_tokens, fh)
        print(f"  total {n:,} tokens, {out.stat().st_size / 1e6:.1f} MB\n")

    if not args.skip_tagged:
        print("Tagged corpora (Universal Dependencies, UPOS)")
        fetch_ud(UD_ENGLISH, RAW / "ud_english.conll")
        fetch_ud(UD_HINDI, RAW / "ud_hindi.conll")

    print("\nNext: python scripts/retrain.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
