import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_INPUT = Path("data/datasets/pairs_mvp.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/datasets/autotrain_pair_score")
DEFAULT_HARD_NEG_MODEL = "BAAI/bge-m3"
SPLITS = ("train", "validation", "test")
MOJIBAKE_MARKERS = ("Ã", "Â", "�")


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


def fix_mojibake(text: str) -> str:
    if not text or not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text

    candidates = [text]
    try:
        candidates.append(text.encode("latin-1").decode("utf-8"))
    except UnicodeError:
        pass
    try:
        candidates.append(text.encode("cp1252").decode("utf-8"))
    except UnicodeError:
        pass

    def badness(value: str) -> int:
        return sum(value.count(marker) for marker in MOJIBAKE_MARKERS)

    return min(candidates, key=badness)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_events(
    text: str,
    max_events: int,
    drop_na_events: bool,
    dedup_events: bool,
    apply_mojibake_fix: bool,
) -> str:
    if not text:
        return ""

    raw_parts = [p.strip() for p in text.split("|")]
    parts: List[str] = []
    seen = set()

    for part in raw_parts:
        if not part:
            continue
        if apply_mojibake_fix:
            part = fix_mojibake(part)
        part = normalize_ws(part)
        if not part:
            continue
        if drop_na_events and part.lower().startswith("na:"):
            continue
        key = part.lower()
        if dedup_events and key in seen:
            continue
        seen.add(key)
        parts.append(part)

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
    drop_na_events: bool,
    dedup_events: bool,
    apply_mojibake_fix: bool,
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

        text_a = truncate_events(
            text=str(src.get("text_a", "")),
            max_events=max_events,
            drop_na_events=drop_na_events,
            dedup_events=dedup_events,
            apply_mojibake_fix=apply_mojibake_fix,
        )
        text_b = truncate_events(
            text=str(src.get("text_b", "")),
            max_events=max_events,
            drop_na_events=drop_na_events,
            dedup_events=dedup_events,
            apply_mojibake_fix=apply_mojibake_fix,
        )
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


def collect_lang_texts(
    positives: List[Dict],
    lang_a: str,
    lang_b: str,
) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    lang_a_text_by_entity: Dict[str, str] = {}
    lang_b_text_by_entity: Dict[str, str] = {}
    for row in positives:
        if row["lang_a"] == lang_a:
            lang_a_text_by_entity[row["entity_a"]] = row["sentence1"]
        elif row["lang_a"] == lang_b:
            lang_b_text_by_entity[row["entity_a"]] = row["sentence1"]

        if row["lang_b"] == lang_a:
            lang_a_text_by_entity[row["entity_b"]] = row["sentence2"]
        elif row["lang_b"] == lang_b:
            lang_b_text_by_entity[row["entity_b"]] = row["sentence2"]

    entities = sorted(set(lang_a_text_by_entity.keys()) & set(lang_b_text_by_entity.keys()))
    return lang_a_text_by_entity, lang_b_text_by_entity, entities


def build_hard_negative_pairs(
    split_name: str,
    entities: List[str],
    lang_a_text_by_entity: Dict[str, str],
    lang_b_text_by_entity: Dict[str, str],
    lang_a: str,
    lang_b: str,
    target_count: int,
    top_k: int,
    rng: random.Random,
    model,
    batch_size: int,
) -> List[Dict]:
    if target_count <= 0 or len(entities) < 2:
        return []

    texts_a = [lang_a_text_by_entity[e] for e in entities]
    texts_b = [lang_b_text_by_entity[e] for e in entities]
    emb_a = model.encode(
        texts_a,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    emb_b = model.encode(
        texts_b,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    sims = np.matmul(emb_a, emb_b.T)
    np.fill_diagonal(sims, -1.0)

    index_candidates: Dict[int, List[int]] = {}
    n_entities = len(entities)
    cap = max(1, min(top_k, n_entities - 1))
    for i in range(n_entities):
        order = np.argsort(-sims[i])
        cands = [int(j) for j in order if j != i][:cap]
        index_candidates[i] = cands

    negatives: List[Dict] = []
    seen = set()
    attempts = 0
    max_attempts = max(500, target_count * 60)

    while len(negatives) < target_count and attempts < max_attempts:
        attempts += 1
        idx_a = rng.randrange(n_entities)
        cands = index_candidates.get(idx_a, [])
        if not cands:
            continue
        idx_b = rng.choice(cands)
        if idx_a == idx_b:
            continue

        entity_a = entities[idx_a]
        entity_b = entities[idx_b]
        if entity_a == entity_b:
            continue

        if rng.random() < 0.5:
            sentence1 = lang_a_text_by_entity[entity_a]
            sentence2 = lang_b_text_by_entity[entity_b]
            neg_lang_a = lang_a
            neg_lang_b = lang_b
        else:
            sentence1 = lang_b_text_by_entity[entity_a]
            sentence2 = lang_a_text_by_entity[entity_b]
            neg_lang_a = lang_b
            neg_lang_b = lang_a

        key = (entity_a, neg_lang_a, entity_b, neg_lang_b)
        if key in seen:
            continue
        seen.add(key)

        negatives.append(
            {
                "sentence1": sentence1,
                "sentence2": sentence2,
                "score": 0.0,
                "entity_a": entity_a,
                "lang_a": neg_lang_a,
                "entity_b": entity_b,
                "lang_b": neg_lang_b,
                "source_id": f"generated_hard_neg_{split_name}_{len(negatives)}",
                "pair_type": "neg_generated_cross_entity_hard",
            }
        )

    return negatives


def build_random_negative_pairs(
    split_name: str,
    entities: List[str],
    lang_a_text_by_entity: Dict[str, str],
    lang_b_text_by_entity: Dict[str, str],
    lang_a: str,
    lang_b: str,
    target_count: int,
    rng: random.Random,
    existing_keys: set,
) -> List[Dict]:
    if target_count <= 0 or len(entities) < 2:
        return []

    negatives: List[Dict] = []
    attempts = 0
    max_attempts = max(500, target_count * 60)

    while len(negatives) < target_count and attempts < max_attempts:
        attempts += 1
        entity_a, entity_b = rng.sample(entities, 2)
        if entity_a == entity_b:
            continue

        if rng.random() < 0.5:
            sentence1 = lang_a_text_by_entity[entity_a]
            sentence2 = lang_b_text_by_entity[entity_b]
            neg_lang_a = lang_a
            neg_lang_b = lang_b
        else:
            sentence1 = lang_b_text_by_entity[entity_a]
            sentence2 = lang_a_text_by_entity[entity_b]
            neg_lang_a = lang_b
            neg_lang_b = lang_a

        key = (entity_a, neg_lang_a, entity_b, neg_lang_b)
        if key in existing_keys:
            continue
        existing_keys.add(key)

        negatives.append(
            {
                "sentence1": sentence1,
                "sentence2": sentence2,
                "score": 0.0,
                "entity_a": entity_a,
                "lang_a": neg_lang_a,
                "entity_b": entity_b,
                "lang_b": neg_lang_b,
                "source_id": f"generated_random_neg_{split_name}_{len(negatives)}",
                "pair_type": "neg_generated_cross_entity_random",
            }
        )

    return negatives


def generate_crossling_negatives(
    split_rows: Dict[str, List[Dict]],
    lang_a: str,
    lang_b: str,
    neg_multiplier: float,
    seed: int,
    hard_neg_ratio: float,
    hard_neg_top_k: int,
    hard_neg_model_id: str,
    hard_neg_batch_size: int,
    hard_neg_max_seq_length: Optional[int],
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, int]]]:
    rng = random.Random(seed)
    out: Dict[str, List[Dict]] = {}
    neg_stats: Dict[str, Dict[str, int]] = {}

    hard_model = None
    if hard_neg_ratio > 0.0:
        from sentence_transformers import SentenceTransformer

        hard_model = SentenceTransformer(hard_neg_model_id)
        if hard_neg_max_seq_length is not None:
            hard_model.max_seq_length = hard_neg_max_seq_length

    for split_name, rows in split_rows.items():
        positives = [r for r in rows if r["score"] >= 0.5]
        target_neg = int(round(len(positives) * max(0.0, neg_multiplier)))
        hard_target = int(round(target_neg * hard_neg_ratio))
        random_target = max(0, target_neg - hard_target)

        lang_a_text_by_entity, lang_b_text_by_entity, entities = collect_lang_texts(
            positives=positives,
            lang_a=lang_a,
            lang_b=lang_b,
        )

        hard_negatives: List[Dict] = []
        if hard_model is not None:
            hard_negatives = build_hard_negative_pairs(
                split_name=split_name,
                entities=entities,
                lang_a_text_by_entity=lang_a_text_by_entity,
                lang_b_text_by_entity=lang_b_text_by_entity,
                lang_a=lang_a,
                lang_b=lang_b,
                target_count=hard_target,
                top_k=hard_neg_top_k,
                rng=rng,
                model=hard_model,
                batch_size=hard_neg_batch_size,
            )

        existing_keys = {
            (r["entity_a"], r["lang_a"], r["entity_b"], r["lang_b"])
            for r in hard_negatives
        }
        random_needed = random_target + max(0, hard_target - len(hard_negatives))
        random_negatives = build_random_negative_pairs(
            split_name=split_name,
            entities=entities,
            lang_a_text_by_entity=lang_a_text_by_entity,
            lang_b_text_by_entity=lang_b_text_by_entity,
            lang_a=lang_a,
            lang_b=lang_b,
            target_count=random_needed,
            rng=rng,
            existing_keys=existing_keys,
        )

        negatives = hard_negatives + random_negatives
        merged = positives + negatives
        rng.shuffle(merged)
        out[split_name] = merged
        neg_stats[split_name] = {
            "positives": len(positives),
            "target_negatives": target_neg,
            "hard_negatives": len(hard_negatives),
            "random_negatives": len(random_negatives),
            "total_negatives": len(negatives),
        }

    return out, neg_stats


def summarize(rows: List[Dict]) -> Dict:
    labels = Counter(int(r["score"] >= 0.5) for r in rows)
    combos = Counter(f"{r['lang_a']}-{r['lang_b']}" for r in rows)
    pair_types = Counter(str(r.get("pair_type", "")) for r in rows)
    entities = len({r["entity_a"] for r in rows} | {r["entity_b"] for r in rows})
    return {
        "rows": len(rows),
        "positives": labels.get(1, 0),
        "negatives": labels.get(0, 0),
        "entities": entities,
        "lang_combos": dict(sorted(combos.items())),
        "pair_types": dict(sorted(pair_types.items())),
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
        "--neg-lang-a",
        type=str,
        default="en",
        help="First language label used for generated cross-lingual negatives.",
    )
    parser.add_argument(
        "--neg-lang-b",
        type=str,
        default="fr",
        help="Second language label used for generated cross-lingual negatives.",
    )
    parser.add_argument(
        "--no-sym-positives",
        action="store_true",
        help="Disable mirrored positive pairs.",
    )
    parser.add_argument(
        "--drop-na-events",
        action="store_true",
        help="Drop timeline events that start with 'NA:' during text construction.",
    )
    parser.add_argument(
        "--no-dedup-events",
        action="store_true",
        help="Keep duplicate events after truncation.",
    )
    parser.add_argument(
        "--no-fix-mojibake",
        action="store_true",
        help="Disable heuristic mojibake cleanup (e.g. CÃ©sar -> Cesar).",
    )
    parser.add_argument(
        "--hard-neg-ratio",
        type=float,
        default=0.7,
        help="Fraction of generated negatives built as semantic hard negatives (0..1).",
    )
    parser.add_argument(
        "--hard-neg-top-k",
        type=int,
        default=8,
        help="Sample hard negatives among top-k most similar wrong entities.",
    )
    parser.add_argument(
        "--hard-neg-model",
        type=str,
        default=DEFAULT_HARD_NEG_MODEL,
        help="Sentence-transformer used to mine hard negatives.",
    )
    parser.add_argument(
        "--hard-neg-batch-size",
        type=int,
        default=128,
        help="Batch size for hard negative embedding inference.",
    )
    parser.add_argument(
        "--hard-neg-max-seq-length",
        type=int,
        default=None,
        help="Optional max sequence length when mining hard negatives.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input dataset not found: {args.input}")
    if args.train_ratio <= 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise SystemExit("Invalid split ratios. Need train>0, val>=0, train+val<1.")
    if args.hard_neg_ratio < 0.0 or args.hard_neg_ratio > 1.0:
        raise SystemExit("--hard-neg-ratio must be between 0 and 1.")
    if args.hard_neg_top_k < 1:
        raise SystemExit("--hard-neg-top-k must be >= 1.")

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
        drop_na_events=bool(args.drop_na_events),
        dedup_events=not args.no_dedup_events,
        apply_mojibake_fix=not args.no_fix_mojibake,
    )
    split_rows, neg_stats = generate_crossling_negatives(
        split_rows=split_rows,
        lang_a=args.neg_lang_a,
        lang_b=args.neg_lang_b,
        neg_multiplier=args.neg_multiplier,
        seed=args.seed,
        hard_neg_ratio=args.hard_neg_ratio,
        hard_neg_top_k=args.hard_neg_top_k,
        hard_neg_model_id=args.hard_neg_model,
        hard_neg_batch_size=args.hard_neg_batch_size,
        hard_neg_max_seq_length=args.hard_neg_max_seq_length,
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
        "neg_lang_a": args.neg_lang_a,
        "neg_lang_b": args.neg_lang_b,
        "sym_positives": not args.no_sym_positives,
        "drop_na_events": bool(args.drop_na_events),
        "dedup_events": not args.no_dedup_events,
        "fix_mojibake": not args.no_fix_mojibake,
        "hard_neg_ratio": args.hard_neg_ratio,
        "hard_neg_top_k": args.hard_neg_top_k,
        "hard_neg_model": args.hard_neg_model,
        "hard_neg_batch_size": args.hard_neg_batch_size,
        "hard_neg_max_seq_length": args.hard_neg_max_seq_length,
        "entities_total": len(entities),
        "entity_split_counts": dict(split_entity_counts),
        "negatives": neg_stats,
        "splits": {s: summarize(split_rows[s]) for s in SPLITS},
    }

    stats_path = args.output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {args.output_dir / 'train.jsonl'}")
    print(f"Saved {args.output_dir / 'validation.jsonl'}")
    print(f"Saved {args.output_dir / 'test.jsonl'}")
    print(f"Saved {stats_path}")
    print(json.dumps(stats["splits"], ensure_ascii=False, indent=2))
    print(json.dumps(stats["negatives"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
