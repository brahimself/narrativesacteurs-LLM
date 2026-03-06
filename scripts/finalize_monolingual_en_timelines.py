import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI


load_dotenv(find_dotenv(usecwd=True), override=True)

DEFAULT_INPUT_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_STATS_OUT = Path("data/datasets/final_en_timelines.stats.json")

DEFAULT_API_KEY = os.getenv("IAFA_INFERENCE", "")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_BILL_TO = os.getenv("BILL_TO", "IAFA-UT")
DEFAULT_MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "1800"))

WS_RE = re.compile(r"\s+")


def normalize_space(value: str) -> str:
    return WS_RE.sub(" ", value or "").strip()


def fix_mojibake(value: str) -> str:
    text = value or ""
    if "Ã" not in text and "â" not in text:
        return text
    try:
        repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return text
    repaired = repaired.strip()
    return repaired if repaired else text


def is_likely_english(text: str) -> bool:
    t = normalize_space(fix_mojibake(text)).lower()
    if not t:
        return True

    # Fast signal: accented letters usually indicate FR (or other non-EN text).
    if re.search(r"[àâäçéèêëîïôöùûüÿœæ]", t):
        return False

    fr_markers = {
        " le ", " la ", " les ", " des ", " une ", " un ", " dans ", " avec ",
        " pour ", " est ", " était ", " né ", " née ", " nommé ", " nommée ",
        " devient ", " remporte ", " soutien ", " soutient ", " fait ", " dont ",
        " puis ", " par ", " sur ", " son ", " sa ", " ses ", " du ", " de ",
    }
    padded = f" {t} "
    hits = sum(1 for m in fr_markers if m in padded)
    return hits <= 2


def extract_json_block(raw: str) -> str:
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(raw[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    raise ValueError("No complete JSON object found in model output")


def safe_json_load(raw: str) -> Dict[str, Any]:
    candidates = [raw]
    try:
        candidates.append(extract_json_block(raw))
    except Exception:
        pass

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"items": obj}
    raise ValueError("Could not parse model output as JSON object.")


def needs_translation(event: Dict[str, Any]) -> bool:
    src_langs = event.get("source_langs", [])
    if isinstance(src_langs, list):
        src_langs_norm = {str(x).strip().lower() for x in src_langs}
    else:
        src_langs_norm = set()

    event_text = normalize_space(str(event.get("event", "")))
    evidence_text = normalize_space(str(event.get("evidence", "")))

    # If there is no explicit EN source, force translation to guarantee final monolingual EN.
    if "en" not in src_langs_norm:
        return True

    # If EN is present but selected text still looks non-EN, translate it too.
    return (not is_likely_english(event_text)) or (evidence_text and not is_likely_english(evidence_text))


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    out = []
    for i in range(0, len(items), chunk_size):
        out.append(items[i:i + chunk_size])
    return out


def entity_slug_from_merged_path(path: Path) -> str:
    parts = path.name.split(".")
    if len(parts) >= 4 and parts[-3] == "merged":
        return ".".join(parts[:-3])
    return path.stem


def call_translation_batch(
    client: OpenAI,
    model_id: str,
    bill_to: str,
    batch: List[Dict[str, Any]],
    max_completion_tokens: int,
    retries: int = 3,
    retry_delay: float = 0.8,
) -> Dict[int, Dict[str, str]]:
    system = (
        "You are a strict translation engine. "
        "Translate French timeline facts into concise English. "
        "Keep facts unchanged (no invention). "
        "Return JSON only."
    )
    user_payload = {
        "task": "Translate each item to English.",
        "rules": [
            "Preserve factual content and chronology.",
            "Keep names, titles, years, and numbers unchanged.",
            "If text is already English, keep it as is.",
            "Return strict JSON with key 'items'.",
            "Each item must include: idx, event_en, evidence_en.",
        ],
        "items": batch,
        "output_schema": {
            "items": [
                {"idx": 0, "event_en": "string", "evidence_en": "string"}
            ]
        },
    }

    headers = {"X-HF-Bill-To": bill_to} if bill_to else None
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                "temperature": 0,
                "top_p": 1,
                "max_completion_tokens": max_completion_tokens,
            }
            if headers:
                kwargs["extra_headers"] = headers

            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content
            if not isinstance(raw, str):
                raw = str(raw)
            obj = safe_json_load(raw)
            items = obj.get("items", [])
            if not isinstance(items, list):
                raise ValueError("Missing 'items' list in translation output.")

            mapped: Dict[int, Dict[str, str]] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                idx = item.get("idx", None)
                if not isinstance(idx, int):
                    continue
                event_en = normalize_space(str(item.get("event_en", "")))
                evidence_en = normalize_space(str(item.get("evidence_en", "")))
                mapped[idx] = {"event_en": event_en, "evidence_en": evidence_en}
            return mapped
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_delay * attempt)

    raise RuntimeError(f"Translation batch failed after retries: {last_error}")


def call_translation_single_fallback(
    client: OpenAI,
    model_id: str,
    bill_to: str,
    item: Dict[str, Any],
    max_completion_tokens: int,
    retries: int = 2,
    retry_delay: float = 0.8,
) -> Dict[int, Dict[str, str]]:
    system = "Translate French timeline text to concise English. No extra facts."
    user = (
        "Translate this item to English.\n"
        f"Event: {item.get('event', '')}\n"
        f"Evidence: {item.get('evidence', '')}\n\n"
        "Return exactly two lines:\n"
        "EVENT_EN: ...\n"
        "EVIDENCE_EN: ..."
    )

    headers = {"X-HF-Bill-To": bill_to} if bill_to else None
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "top_p": 1,
                "max_completion_tokens": max_completion_tokens,
            }
            if headers:
                kwargs["extra_headers"] = headers

            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content
            if not isinstance(raw, str):
                raw = str(raw)
            text = raw.strip()

            m_event = re.search(r"EVENT_EN:\s*(.+)", text, flags=re.IGNORECASE)
            m_evidence = re.search(r"EVIDENCE_EN:\s*(.+)", text, flags=re.IGNORECASE)
            if not m_event:
                raise ValueError("Missing EVENT_EN in fallback response.")

            event_en = normalize_space(m_event.group(1))
            evidence_en = normalize_space(m_evidence.group(1) if m_evidence else item.get("evidence", ""))
            return {
                int(item["idx"]): {
                    "event_en": event_en or normalize_space(str(item.get("event", ""))),
                    "evidence_en": evidence_en or normalize_space(str(item.get("evidence", ""))),
                }
            }
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_delay * attempt)

    raise RuntimeError(f"Single-item fallback translation failed: {last_error}")


def translate_batch_resilient(
    client: OpenAI,
    model_id: str,
    bill_to: str,
    batch: List[Dict[str, Any]],
    max_completion_tokens: int,
) -> Dict[int, Dict[str, str]]:
    if not batch:
        return {}
    try:
        return call_translation_batch(
            client=client,
            model_id=model_id,
            bill_to=bill_to,
            batch=batch,
            max_completion_tokens=max_completion_tokens,
        )
    except Exception:
        if len(batch) == 1:
            item = batch[0]
            try:
                return call_translation_single_fallback(
                    client=client,
                    model_id=model_id,
                    bill_to=bill_to,
                    item=item,
                    max_completion_tokens=max_completion_tokens,
                )
            except Exception:
                return {
                    int(item["idx"]): {
                        "event_en": normalize_space(str(item.get("event", ""))),
                        "evidence_en": normalize_space(str(item.get("evidence", ""))),
                    }
                }

        mid = len(batch) // 2
        left = translate_batch_resilient(
            client=client,
            model_id=model_id,
            bill_to=bill_to,
            batch=batch[:mid],
            max_completion_tokens=max_completion_tokens,
        )
        right = translate_batch_resilient(
            client=client,
            model_id=model_id,
            bill_to=bill_to,
            batch=batch[mid:],
            max_completion_tokens=max_completion_tokens,
        )
        left.update(right)
        return left


def process_file(
    path: Path,
    output_dir: Path,
    client: OpenAI,
    model_id: str,
    bill_to: str,
    max_completion_tokens: int,
    translation_batch_size: int,
) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events", [])
    if not isinstance(events, list):
        events = []

    events_out = []
    pending = []
    translated_count = 0

    for idx, e in enumerate(events):
        if not isinstance(e, dict):
            continue
        year = e.get("year", None)
        event_text = normalize_space(fix_mojibake(str(e.get("event", ""))))
        evidence_text = normalize_space(fix_mojibake(str(e.get("evidence", ""))))

        out_event = dict(e)
        out_event["year"] = year
        out_event["event"] = event_text
        out_event["evidence"] = evidence_text
        out_event["translated_to_en"] = False
        events_out.append(out_event)

        if needs_translation(out_event):
            pending.append(
                {
                    "idx": idx,
                    "event": event_text,
                    "evidence": evidence_text,
                }
            )

    for batch in chunk_list(pending, translation_batch_size):
        translated_map = translate_batch_resilient(
            client=client,
            model_id=model_id,
            bill_to=bill_to,
            batch=batch,
            max_completion_tokens=max_completion_tokens,
        )
        for item in batch:
            idx = item["idx"]
            translated = translated_map.get(idx)
            if not translated:
                continue
            event_en = translated.get("event_en", "") or item["event"]
            evidence_en = translated.get("evidence_en", "") or item["evidence"]
            events_out[idx]["event"] = event_en
            events_out[idx]["evidence"] = evidence_en
            events_out[idx]["translated_to_en"] = True
            translated_count += 1

    entity_slug = data.get("entity_slug", "") or entity_slug_from_merged_path(path)

    payload = {
        "entity": data.get("entity", ""),
        "entity_slug": entity_slug,
        "lang": "en",
        "source_file": str(path),
        "source_kind": "merged_timeline",
        "events": events_out,
        "final_meta": {
            "total_events": len(events_out),
            "translated_events": translated_count,
            "already_english_events": len(events_out) - translated_count,
        },
    }

    out_path = output_dir / f"{entity_slug}.final_en.timeline.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "entity_slug": entity_slug,
        "in_path": str(path),
        "out_path": str(out_path),
        "total_events": len(events_out),
        "translated_events": translated_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert merged multilingual timelines into final monolingual English timelines."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", type=str, default="*.merged.timeline.json")
    parser.add_argument("--entity-slug", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--bill-to", type=str, default=DEFAULT_BILL_TO)
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--translation-batch-size", type=int, default=30)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--start-after", type=str, default=None, help="Process files whose slug is lexicographically greater than this value.")
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS_OUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    if not args.api_key:
        raise SystemExit("Missing API key. Set IAFA_INFERENCE in .env or use --api-key.")
    if args.translation_batch_size < 1:
        raise SystemExit("--translation-batch-size must be >= 1.")

    files = sorted(args.input_dir.glob(args.pattern))
    if args.entity_slug:
        files = [fp for fp in files if fp.name.startswith(f"{args.entity_slug}.")]
    if args.start_after:
        files = [fp for fp in files if entity_slug_from_merged_path(fp) > args.start_after]
    if args.skip_existing:
        filtered = []
        for fp in files:
            slug = entity_slug_from_merged_path(fp)
            out_path = args.output_dir / f"{slug}.final_en.timeline.json"
            if not out_path.exists():
                filtered.append(fp)
        files = filtered
    if args.max_files is not None and args.max_files >= 0:
        files = files[:args.max_files]
    if not files:
        raise SystemExit("No merged timeline files found for conversion.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=args.api_key)

    stats_items: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    total_events = 0
    translated_events = 0

    for fp in files:
        try:
            row = process_file(
                path=fp,
                output_dir=args.output_dir,
                client=client,
                model_id=args.model,
                bill_to=args.bill_to,
                max_completion_tokens=args.max_completion_tokens,
                translation_batch_size=args.translation_batch_size,
            )
            stats_items.append(row)
            total_events += row["total_events"]
            translated_events += row["translated_events"]
            print(
                f"Finalized {row['entity_slug']}: translated={row['translated_events']}/{row['total_events']}"
            )
        except Exception as exc:
            errors.append({"file": str(fp), "error": str(exc)})
            print(f"Failed {fp.name}: {exc}")

    stats = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "pattern": args.pattern,
        "model": args.model,
        "entities_total": len(files),
        "entities_succeeded": len(stats_items),
        "entities_failed": len(errors),
        "events_total": total_events,
        "events_translated": translated_events,
        "events_already_english": total_events - translated_events,
        "translation_ratio": (translated_events / total_events) if total_events else 0.0,
        "entities": stats_items,
        "errors": errors,
    }

    if args.stats_out is not None:
        args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved stats: {args.stats_out}")

    print(
        f"Done. Final EN timelines: {len(stats_items)} | "
        f"translated events: {translated_events}/{total_events}"
    )


if __name__ == "__main__":
    main()
