import json
import random
import re
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_OUTPUT = Path("data/datasets/pairs_mvp.jsonl")
DEFAULT_STATS_OUTPUT = Path("data/datasets/pairs_mvp.stats.json")


def load_timelines(
    processed_dir: Path,
    langs: Tuple[str, ...],
    include_merged: bool,
    merged_lang: str,
    min_events_required: int,
) -> Dict[Tuple[str, str], Dict]:
    """
    Loads files like Entity_Slug.en.timeline.json and optionally Entity_Slug.merged.timeline.json.
    Returns dict keyed by (entity_slug, lang).
    """
    timelines = {}
    wanted_langs = set(langs)
    for fp in processed_dir.glob("*.timeline.json"):
        name = fp.name  # e.g. Angelina_Jolie.en.timeline.json
        parts = name.split(".")
        if len(parts) < 4:
            continue

        if parts[-3] == "merged":
            if not include_merged:
                continue
            entity_slug = ".".join(parts[:-3])
            lang = merged_lang
        else:
            entity_slug, lang = parts[0], parts[1]
            if lang not in wanted_langs:
                continue

        data = json.loads(fp.read_text(encoding="utf-8"))
        events = data.get("events", [])
        if not isinstance(events, list) or len(events) < min_events_required:
            continue
        timelines[(entity_slug, lang)] = data
    return timelines

def event_to_line(e: Dict) -> str:
    y = e.get("year", None)
    y_str = str(y) if isinstance(y, int) else "NA"
    ev = (e.get("event") or "").strip()
    # make it stable / single line
    ev = re.sub(r"\s+", " ", ev)
    return f"{y_str}: {ev}"

def timeline_to_text(tl: Dict, max_events_used: int) -> str:
    events = tl.get("events", [])[:max_events_used]
    lines = [event_to_line(e) for e in events if (e.get("event") or "").strip()]
    # keep it as a narrative-like paragraph
    return " | ".join(lines)

def build_positive_pairs(
    timelines: Dict[Tuple[str, str], Dict],
    langs: Tuple[str, ...],
    include_merged: bool,
    merged_lang: str,
    max_events_used: int,
) -> List[Dict]:
    pos = []

    # Base positives between canonical langs (e.g. en-fr).
    if len(langs) >= 2:
        a_lang, b_lang = langs[0], langs[1]
        entities = sorted({
            slug for (slug, _) in timelines.keys()
            if (slug, a_lang) in timelines and (slug, b_lang) in timelines
        })
        for slug in entities:
            tl_a = timelines[(slug, a_lang)]
            tl_b = timelines[(slug, b_lang)]
            pos.append({
                "id": f"{slug}_{a_lang}_{b_lang}",
                "label": 1,
                "entity_a": slug,
                "lang_a": a_lang,
                "text_a": timeline_to_text(tl_a, max_events_used=max_events_used),
                "entity_b": slug,
                "lang_b": b_lang,
                "text_b": timeline_to_text(tl_b, max_events_used=max_events_used),
                "pair_type": f"pos_same_entity_{a_lang}_{b_lang}",
            })

    # Additional positives between merged timeline and each source language.
    if include_merged:
        entities_with_merged = sorted({
            slug for (slug, lang) in timelines.keys() if lang == merged_lang
        })
        for slug in entities_with_merged:
            merged_tl = timelines[(slug, merged_lang)]
            for base_lang in langs:
                if (slug, base_lang) not in timelines:
                    continue
                tl_lang = timelines[(slug, base_lang)]
                pos.append({
                    "id": f"{slug}_{base_lang}_{merged_lang}",
                    "label": 1,
                    "entity_a": slug,
                    "lang_a": base_lang,
                    "text_a": timeline_to_text(tl_lang, max_events_used=max_events_used),
                    "entity_b": slug,
                    "lang_b": merged_lang,
                    "text_b": timeline_to_text(merged_tl, max_events_used=max_events_used),
                    "pair_type": f"pos_same_entity_{base_lang}_{merged_lang}",
                })

    return pos

def build_negative_pairs(
    pos_pairs: List[Dict],
    timelines: Dict[Tuple[str, str], Dict],
    neg_multiplier: int,
    max_events_used: int,
) -> List[Dict]:
    # candidate timelines list
    candidates = [(slug, lang) for (slug, lang) in timelines.keys()]
    neg = []
    target = len(pos_pairs) * neg_multiplier
    langs_available = sorted({lang for _, lang in candidates})

    # To keep negatives meaningful, sample same language half the time when possible.
    while len(neg) < target:
        if random.random() < 0.5 and langs_available:
            # same language negative
            lang = random.choice(langs_available)
            pool = [x for x in candidates if x[1] == lang]
            if len(pool) < 2:
                continue
            (a_slug, a_lang), (b_slug, b_lang) = random.sample(pool, 2)
        else:
            (a_slug, a_lang), (b_slug, b_lang) = random.sample(candidates, 2)

        if a_slug == b_slug:
            continue

        a_tl = timelines[(a_slug, a_lang)]
        b_tl = timelines[(b_slug, b_lang)]

        neg.append({
            "id": f"neg_{len(neg)}_{a_slug}_{a_lang}__{b_slug}_{b_lang}",
            "label": 0,
            "entity_a": a_slug,
            "lang_a": a_lang,
            "text_a": timeline_to_text(a_tl, max_events_used=max_events_used),
            "entity_b": b_slug,
            "lang_b": b_lang,
            "text_b": timeline_to_text(b_tl, max_events_used=max_events_used),
            "pair_type": "neg_diff_entity",
        })

    return neg

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build narrative similarity pairs from timeline files."
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--langs", nargs="+", default=["en", "fr"])
    parser.add_argument("--include-merged", action="store_true")
    parser.add_argument("--merged-lang", type=str, default="multi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg-multiplier", type=int, default=3)
    parser.add_argument("--max-events-used", type=int, default=35)
    parser.add_argument("--min-events-required", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS_OUTPUT)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not args.processed_dir.exists():
        raise SystemExit(f"Processed directory not found: {args.processed_dir}")
    if args.neg_multiplier < 0:
        raise SystemExit("--neg-multiplier must be >= 0")
    if args.max_events_used < 1:
        raise SystemExit("--max-events-used must be >= 1")
    if args.min_events_required < 1:
        raise SystemExit("--min-events-required must be >= 1")

    random.seed(args.seed)

    langs = tuple(str(x).strip().lower() for x in args.langs if str(x).strip())
    if len(langs) < 2:
        raise SystemExit("Need at least 2 langs for positive pair construction.")

    timelines = load_timelines(
        processed_dir=args.processed_dir,
        langs=langs,
        include_merged=args.include_merged,
        merged_lang=args.merged_lang,
        min_events_required=args.min_events_required,
    )
    print(f"Loaded timelines: {len(timelines)} files")

    pos = build_positive_pairs(
        timelines=timelines,
        langs=langs,
        include_merged=args.include_merged,
        merged_lang=args.merged_lang,
        max_events_used=args.max_events_used,
    )
    pair_type_counts = Counter(p["pair_type"] for p in pos)
    print(f"Positive pairs: {len(pos)}")
    print(f"Positive pair types: {dict(sorted(pair_type_counts.items()))}")

    neg = build_negative_pairs(
        pos_pairs=pos,
        timelines=timelines,
        neg_multiplier=args.neg_multiplier,
        max_events_used=args.max_events_used,
    )
    print(f"Negative pairs: {len(neg)} (multiplier={args.neg_multiplier})")

    pairs = pos + neg
    random.shuffle(pairs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # small stats file
    stats = {
        "timelines_loaded": len(timelines),
        "positive_pairs": len(pos),
        "negative_pairs": len(neg),
        "total_pairs": len(pairs),
        "neg_multiplier": args.neg_multiplier,
        "max_events_used": args.max_events_used,
        "min_events_required": args.min_events_required,
        "langs": list(langs),
        "include_merged": bool(args.include_merged),
        "merged_lang": args.merged_lang if args.include_merged else "",
        "positive_pair_type_counts": dict(sorted(pair_type_counts.items())),
        "seed": args.seed,
    }
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved: {args.output}")
    print(f"Saved: {args.stats_out}")

if __name__ == "__main__":
    main()
