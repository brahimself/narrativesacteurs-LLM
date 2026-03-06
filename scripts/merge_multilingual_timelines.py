import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_INPUT_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_LANGS = ("en", "fr")
DEFAULT_TARGET_LANG = "en"
DEFAULT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

WS_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")


@dataclass
class EventRecord:
    year: Optional[int]
    event: str
    evidence: str
    source_langs: set[str] = field(default_factory=set)
    source_count: int = 1


def normalize_space(value: str) -> str:
    return WS_RE.sub(" ", value or "").strip()


def normalize_text_key(value: str) -> str:
    value = normalize_space(value)
    value = unicodedata.normalize("NFKD", value).lower()
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = WS_RE.sub(" ", value).strip()
    return value


def parse_year(raw_year) -> Optional[int]:
    if raw_year is None:
        return None
    if isinstance(raw_year, float) and raw_year.is_integer():
        raw_year = int(raw_year)
    if isinstance(raw_year, int):
        return raw_year if 1800 <= raw_year <= 2100 else None

    text = str(raw_year).strip()
    if not text:
        return None
    if text.isdigit():
        year = int(text)
        return year if 1800 <= year <= 2100 else None

    match = YEAR_RE.search(text)
    if not match:
        return None
    year = int(match.group(0))
    return year if 1800 <= year <= 2100 else None


def choose_best_record(a: EventRecord, b: EventRecord, target_lang: str) -> EventRecord:
    def score(rec: EventRecord) -> Tuple[int, int, int]:
        has_target = 1 if target_lang in rec.source_langs else 0
        return (has_target, len(rec.event), len(rec.evidence))

    return a if score(a) >= score(b) else b


def merge_record(a: EventRecord, b: EventRecord, target_lang: str) -> EventRecord:
    best = choose_best_record(a, b, target_lang=target_lang)
    merged = EventRecord(
        year=best.year,
        event=best.event,
        evidence=best.evidence,
        source_langs=set(a.source_langs) | set(b.source_langs),
        source_count=a.source_count + b.source_count,
    )
    return merged


def load_timeline(path: Path) -> Tuple[str, List[Dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entity = normalize_space(str(data.get("entity", "")))
    events = data.get("events", [])
    if not isinstance(events, list):
        events = []
    return entity, events


def group_timeline_files(input_dir: Path, langs: List[str]) -> Dict[str, Dict[str, Path]]:
    wanted = set(langs)
    grouped: Dict[str, Dict[str, Path]] = {}

    for fp in sorted(input_dir.glob("*.timeline.json")):
        parts = fp.name.split(".")
        if len(parts) < 4:
            continue
        if parts[-2] != "timeline" or parts[-1] != "json":
            continue
        lang = parts[-3].strip()
        if lang not in wanted:
            continue
        entity_slug = ".".join(parts[:-3]).strip()
        if not entity_slug:
            continue
        grouped.setdefault(entity_slug, {})[lang] = fp

    return grouped


def read_records_for_entity(files_by_lang: Dict[str, Path]) -> Tuple[str, List[EventRecord]]:
    all_records: List[EventRecord] = []
    entity_name = ""

    for lang, path in sorted(files_by_lang.items()):
        entity, events = load_timeline(path)
        if entity and not entity_name:
            entity_name = entity

        for raw_event in events:
            if not isinstance(raw_event, dict):
                continue
            event_text = normalize_space(str(raw_event.get("event", "")))
            if not event_text:
                continue
            evidence_text = normalize_space(str(raw_event.get("evidence", "")))
            year = parse_year(raw_event.get("year"))
            all_records.append(
                EventRecord(
                    year=year,
                    event=event_text,
                    evidence=evidence_text,
                    source_langs={lang},
                    source_count=1,
                )
            )

    return entity_name, all_records


def exact_dedup(records: List[EventRecord], target_lang: str) -> List[EventRecord]:
    dedup: Dict[Tuple[Optional[int], str], EventRecord] = {}
    ordered: List[Tuple[Optional[int], str]] = []

    for rec in records:
        key = (rec.year, normalize_text_key(rec.event))
        prev = dedup.get(key)
        if prev is None:
            dedup[key] = rec
            ordered.append(key)
        else:
            dedup[key] = merge_record(prev, rec, target_lang=target_lang)

    return [dedup[k] for k in ordered]


def year_compatible(y1: Optional[int], y2: Optional[int]) -> bool:
    if y1 is None and y2 is None:
        return True
    if y1 is None or y2 is None:
        return False
    return y1 == y2


def semantic_merge(
    records: List[EventRecord],
    model: SentenceTransformer,
    threshold: float,
    target_lang: str,
    batch_size: int,
) -> List[EventRecord]:
    if len(records) <= 1:
        return records

    texts = [r.event for r in records]
    emb = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    emb = np.asarray(emb, dtype=np.float32)

    clusters: List[List[int]] = []
    for i, rec in enumerate(records):
        best_cluster = None
        best_sim = -1.0

        for c_idx, cluster in enumerate(clusters):
            base = records[cluster[0]]
            if not year_compatible(rec.year, base.year):
                continue

            c_emb = emb[cluster]
            sims = np.dot(c_emb, emb[i])
            sim = float(np.max(sims))
            if sim > best_sim:
                best_sim = sim
                best_cluster = c_idx

        if best_cluster is not None and best_sim >= threshold:
            clusters[best_cluster].append(i)
        else:
            clusters.append([i])

    merged: List[EventRecord] = []
    for cluster in clusters:
        current = records[cluster[0]]
        for idx in cluster[1:]:
            current = merge_record(current, records[idx], target_lang=target_lang)
        merged.append(current)

    return merged


def sort_events(records: List[EventRecord]) -> List[EventRecord]:
    return sorted(
        records,
        key=lambda r: (
            r.year is None,
            r.year if isinstance(r.year, int) else 9999,
            normalize_text_key(r.event),
        ),
    )


def to_output_payload(
    entity: str,
    entity_slug: str,
    merged_events: List[EventRecord],
    langs: List[str],
    target_lang: str,
    semantic_threshold: float,
    source_event_count: int,
) -> Dict:
    events_out = []
    for rec in merged_events:
        events_out.append(
            {
                "year": rec.year,
                "event": rec.event,
                "evidence": rec.evidence,
                "source_langs": sorted(rec.source_langs),
                "source_event_count": rec.source_count,
            }
        )

    return {
        "entity": entity,
        "entity_slug": entity_slug,
        "target_lang": target_lang,
        "langs_merged": langs,
        "merge_meta": {
            "source_events_count": source_event_count,
            "merged_events_count": len(events_out),
            "semantic_threshold": semantic_threshold,
        },
        "events": events_out,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multilingual timelines into one exhaustive timeline per entity."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--langs", nargs="+", default=list(DEFAULT_LANGS))
    parser.add_argument("--target-lang", type=str, default=DEFAULT_TARGET_LANG)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--semantic-threshold", type=float, default=0.80)
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--entity-slug", type=str, default=None)
    parser.add_argument("--no-semantic-merge", action="store_true")
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=Path("data/datasets/merged_timelines.stats.json"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    if args.max_events < 1:
        raise SystemExit("--max-events must be >= 1.")
    if args.semantic_threshold < -1.0 or args.semantic_threshold > 1.0:
        raise SystemExit("--semantic-threshold must be in [-1, 1].")

    langs = [normalize_space(str(l)).lower() for l in args.langs if normalize_space(str(l))]
    if len(langs) < 2:
        raise SystemExit("Need at least 2 languages in --langs.")

    grouped = group_timeline_files(args.input_dir, langs=langs)
    if args.entity_slug:
        grouped = {args.entity_slug: grouped.get(args.entity_slug, {})}

    candidates = {
        slug: files
        for slug, files in grouped.items()
        if all(lang in files for lang in langs)
    }
    if not candidates:
        raise SystemExit("No entity with complete language set found.")

    model = None
    if not args.no_semantic_merge:
        print(f"Loading sentence model: {args.model}")
        model = SentenceTransformer(args.model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "langs": langs,
        "target_lang": args.target_lang,
        "semantic_merge": not args.no_semantic_merge,
        "semantic_threshold": args.semantic_threshold,
        "model": args.model if model is not None else "",
        "entities_total": len(candidates),
        "entities_written": 0,
        "avg_source_events": 0.0,
        "avg_merged_events": 0.0,
        "entities": [],
    }

    source_total = 0
    merged_total = 0

    for entity_slug in sorted(candidates.keys()):
        files_by_lang = candidates[entity_slug]
        entity, records = read_records_for_entity(files_by_lang)
        if not entity:
            entity = entity_slug.replace("_", " ")

        source_count = len(records)
        if source_count == 0:
            continue

        merged = exact_dedup(records, target_lang=args.target_lang)
        if model is not None:
            merged = semantic_merge(
                records=merged,
                model=model,
                threshold=args.semantic_threshold,
                target_lang=args.target_lang,
                batch_size=args.batch_size,
            )
        merged = sort_events(merged)[: args.max_events]

        payload = to_output_payload(
            entity=entity,
            entity_slug=entity_slug,
            merged_events=merged,
            langs=langs,
            target_lang=args.target_lang,
            semantic_threshold=args.semantic_threshold,
            source_event_count=source_count,
        )
        out_path = args.output_dir / f"{entity_slug}.merged.timeline.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        stats["entities_written"] += 1
        source_total += source_count
        merged_total += len(merged)
        stats["entities"].append(
            {
                "entity_slug": entity_slug,
                "source_events": source_count,
                "merged_events": len(merged),
                "out_path": str(out_path),
            }
        )
        print(
            f"Merged {entity_slug}: source={source_count} -> merged={len(merged)}"
        )

    if stats["entities_written"] > 0:
        stats["avg_source_events"] = source_total / stats["entities_written"]
        stats["avg_merged_events"] = merged_total / stats["entities_written"]

    if args.stats_out is not None:
        args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved stats: {args.stats_out}")

    print(f"Done. Files written: {stats['entities_written']}")


if __name__ == "__main__":
    main()
