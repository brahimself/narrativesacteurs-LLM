import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List


DEFAULT_INPUT = Path("data/datasets/pairs_mvp.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/datasets/autotrain_pair_score")
SPLITS = ("train", "validation", "test")


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_events(text: str, max_events: int) -> str:
    if not text:
        return ""
    parts = [p.strip() for p in text.split("|")]
    parts = [p for p in parts if p]
    if max_events > 0:
        parts = parts[:max_events]
    return " | ".join(parts)


def assign_entity_split(
    entities: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, str]:
    rng = random.Random(seed)
    shuffled = entities[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_train = max(1, n_train) if n_total >= 3 else n_train
    n_val = max(1, n_val) if n_total >= 5 else n_val
    if n_train + n_val >= n_total:
        n_val = max(0, n_total - n_train - 1)

    split_map: Dict[str, str] = {}
    for i, entity in enumerate(shuffled):
        if i < n_train:
            split_map[entity] = "train"
        elif i < n_train + n_val:
            split_map[entity] = "validation"
        else:
            split_map[entity] = "test"
    return split_map


def add_row_with_dedup(
    out_rows: Dict[str, List[Dict]],
    seen: Dict[str, set],
    split_name: str,
    row: Dict,
) -> None:
    key = (
        row["score"],
        row["entity_a"],
        row["lang_a"],
        row["entity_b"],
        row["lang_b"],
        row["sentence1"],
        row["sentence2"],
    )
    if key in seen[split_name]:
        return
    seen[split_name].add(key)
    out_rows[split_name].append(row)


def build_rows(
    rows: List[Dict],
    entity_split: Dict[str, str],
    max_events: int,
    sym_positives: bool,
) -> Dict[str, List[Dict]]:
    out_rows: Dict[str, List[Dict]] = {s: [] for s in SPLITS}
    seen: Dict[str, set] = {s: set() for s in SPLITS}

    for src in rows:
        label = int(src.get("label", 0))
        if label != 1:
            continue

        entity_a = str(src.get("entity_a", "")).strip()
        entity_b = str(src.get("entity_b", "")).strip()
        lang_a = str(src.get("lang_a", "")).strip()
        lang_b = str(src.get("lang_b", "")).strip()

        # Keep only cross-lingual pairs, otherwise model learns language shortcut.
        if not lang_a or not lang_b or lang_a == lang_b:
            continue

        split_a = entity_split.get(entity_a)
        split_b = entity_split.get(entity_b)
        if split_a is None or split_b is None or split_a != split_b:
            continue

        # Keep only positive same-entity pairs from source.
        if entity_a != entity_b:
            continue

        text_a = truncate_events(str(src.get("text_a", "")), max_events)
        text_b = truncate_events(str(src.get("text_b", "")), max_events)
        if not text_a or not text_b:
            continue

        base = {
            "sentence1": text_a,
            "sentence2": text_b,
            "score": 1.0,
            "entity_a": entity_a,
            "lang_a": lang_a,
            "entity_b": entity_b,
            "lang_b": lang_b,
            "source_id": str(src.get("id", "")),
            "pair_type": str(src.get("pair_type", "")),
        }
        add_row_with_dedup(out_rows, seen, split_a, base)

        if sym_positives:
            mirrored = {
                "sentence1": text_b,
                "sentence2": text_a,
                "score": 1.0,
                "entity_a": entity_b,
                "lang_a": lang_b,
                "entity_b": entity_a,
                "lang_b": lang_a,
                "source_id": f"{src.get('id', '')}_rev",
                "pair_type": f"{src.get('pair_type', '')}_rev",
            }
            add_row_with_dedup(out_rows, seen, split_a, mirrored)

    return out_rows


def generate_crossling_negatives(
    split_rows: Dict[str, List[Dict]],
    neg_multiplier: float,
    seed: int,
) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    out: Dict[str, List[Dict]] = {}
    for split_name, rows in split_rows.items():
        positives = [r for r in rows if r["score"] >= 0.5]
        target_neg = int(round(len(positives) * max(0.0, neg_multiplier)))

        en_text_by_entity: Dict[str, str] = {}
        fr_text_by_entity: Dict[str, str] = {}
        for row in positives:
            if row["lang_a"] == "en":
                en_text_by_entity[row["entity_a"]] = row["sentence1"]
            elif row["lang_a"] == "fr":
                fr_text_by_entity[row["entity_a"]] = row["sentence1"]

            if row["lang_b"] == "en":
                en_text_by_entity[row["entity_b"]] = row["sentence2"]
            elif row["lang_b"] == "fr":
                fr_text_by_entity[row["entity_b"]] = row["sentence2"]

        entities = sorted(set(en_text_by_entity.keys()) & set(fr_text_by_entity.keys()))
        negatives: List[Dict] = []
        seen_neg = set()

        if len(entities) >= 2 and target_neg > 0:
            max_attempts = target_neg * 40
            attempts = 0
            while len(negatives) < target_neg and attempts < max_attempts:
                attempts += 1
                entity_a, entity_b = rng.sample(entities, 2)
                if rng.random() < 0.5:
                    lang_a, lang_b = "en", "fr"
                    sentence1 = en_text_by_entity[entity_a]
                    sentence2 = fr_text_by_entity[entity_b]
                else:
                    lang_a, lang_b = "fr", "en"
                    sentence1 = fr_text_by_entity[entity_a]
                    sentence2 = en_text_by_entity[entity_b]

                key = (entity_a, lang_a, entity_b, lang_b)
                if key in seen_neg:
                    continue
                seen_neg.add(key)

                negatives.append(
                    {
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                        "score": 0.0,
                        "entity_a": entity_a,
                        "lang_a": lang_a,
                        "entity_b": entity_b,
                        "lang_b": lang_b,
                        "source_id": f"generated_neg_{split_name}_{len(negatives)}",
                        "pair_type": "neg_generated_cross_entity",
                    }
                )

        merged = positives + negatives
        rng.shuffle(merged)
        out[split_name] = merged
    return out


def summarize(rows: List[Dict]) -> Dict:
    labels = Counter(int(r["score"] >= 0.5) for r in rows)
    combos = Counter(f"{r['lang_a']}-{r['lang_b']}" for r in rows)
    entities = len({r["entity_a"] for r in rows} | {r["entity_b"] for r in rows})
    return {
        "rows": len(rows),
        "positives": labels.get(1, 0),
        "negatives": labels.get(0, 0),
        "entities": entities,
        "lang_combos": dict(sorted(combos.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build AutoTrain pair_score datasets with strict entity split."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-events", type=int, default=14)
    parser.add_argument("--neg-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--no-sym-positives",
        action="store_true",
        help="Disable mirrored positive pairs.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input dataset not found: {args.input}")
    if args.train_ratio <= 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise SystemExit("Invalid split ratios. Need train>0, val>=0, train+val<1.")

    rows = read_jsonl(args.input)
    positives = [
        r for r in rows
        if int(r.get("label", 0)) == 1
        and r.get("entity_a") == r.get("entity_b")
    ]
    entities = sorted({str(r["entity_a"]) for r in positives})
    if len(entities) < 10:
        raise SystemExit(f"Not enough entities for robust split: {len(entities)}")

    entity_split = assign_entity_split(
        entities=entities,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    split_rows = build_rows(
        rows=rows,
        entity_split=entity_split,
        max_events=args.max_events,
        sym_positives=not args.no_sym_positives,
    )
    split_rows = generate_crossling_negatives(
        split_rows=split_rows,
        neg_multiplier=args.neg_multiplier,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLITS:
        out_path = args.output_dir / f"{split_name}.jsonl"
        write_jsonl(out_path, split_rows[split_name])

    split_entity_counts = Counter(entity_split.values())
    stats = {
        "source": str(args.input),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "max_events": args.max_events,
        "neg_multiplier": args.neg_multiplier,
        "sym_positives": not args.no_sym_positives,
        "entities_total": len(entities),
        "entity_split_counts": dict(split_entity_counts),
        "splits": {s: summarize(split_rows[s]) for s in SPLITS},
    }

    stats_path = args.output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {args.output_dir / 'train.jsonl'}")
    print(f"Saved {args.output_dir / 'validation.jsonl'}")
    print(f"Saved {args.output_dir / 'test.jsonl'}")
    print(f"Saved {stats_path}")
    print(json.dumps(stats["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
