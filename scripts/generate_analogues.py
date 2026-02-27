import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


load_dotenv(find_dotenv(usecwd=True), override=True)

DEFAULT_TIMELINE_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("data/results")
DEFAULT_SIMILARITY_MODEL = "sidbrahim/autotrain-NarraAnalogues15events"
DEFAULT_GENERATOR_MODEL = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "1400"))
DEFAULT_BILL_TO = os.getenv("BILL_TO", "IAFA-UT")
DEFAULT_API_KEY = os.getenv("IAFA_INFERENCE", "")

WS_RE = re.compile(r"\s+")


def normalize_space(value: str) -> str:
    return WS_RE.sub(" ", value or "").strip()


def slugify(name: str) -> str:
    value = name.strip()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value)
    return value


def timeline_events_to_text(events: List[Dict[str, Any]], max_events: int) -> str:
    lines: List[str] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        event = normalize_space(str(item.get("event", "")))
        if not event:
            continue

        year = item.get("year")
        if isinstance(year, float) and year.is_integer():
            year = int(year)
        if isinstance(year, int):
            year_str = str(year)
        else:
            year_str = "NA"

        lines.append(f"{year_str}: {event}")

    if max_events > 0:
        lines = lines[:max_events]
    return " | ".join(lines)


def normalize_narrative_text(text: str, max_events: int) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    if "|" in text:
        parts = [normalize_space(p) for p in text.split("|")]
        parts = [p for p in parts if p]
        if max_events > 0:
            parts = parts[:max_events]
        return " | ".join(parts)

    lines: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\).\s-]+", "", line).strip()
        line = re.sub(r"^[-*]\s+", "", line).strip()
        line = normalize_space(line)
        if line:
            lines.append(line)

    if not lines:
        lines = [normalize_space(text)]
    if max_events > 0:
        lines = lines[:max_events]
    return " | ".join(lines)


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
                return raw[start : i + 1]

    raise ValueError("No complete JSON object found in model output")


def parse_candidate_output(raw: str, max_events: int) -> Dict[str, str]:
    payload: Optional[Dict[str, Any]] = None

    candidates = [raw]
    try:
        candidates.append(extract_json_block(raw))
    except ValueError:
        pass

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            payload = obj
            break

    if payload is None:
        narrative = normalize_narrative_text(raw, max_events=max_events)
        return {
            "candidate_name": "",
            "candidate_narrative": narrative,
            "rationale": "",
        }

    narrative = ""
    if isinstance(payload.get("candidate_narrative"), str):
        narrative = payload["candidate_narrative"]
    elif isinstance(payload.get("narrative"), str):
        narrative = payload["narrative"]
    elif isinstance(payload.get("text"), str):
        narrative = payload["text"]
    elif isinstance(payload.get("events"), list):
        narrative = timeline_events_to_text(payload["events"], max_events=max_events)

    narrative = normalize_narrative_text(narrative, max_events=max_events)
    return {
        "candidate_name": normalize_space(str(payload.get("candidate_name", payload.get("entity", "")))),
        "candidate_narrative": narrative,
        "rationale": normalize_space(str(payload.get("why_this_should_match", payload.get("rationale", "")))),
    }


def find_source_file(entity: str, lang: str, timeline_dir: Path) -> Path:
    slug = slugify(entity)
    path = timeline_dir / f"{slug}.{lang}.timeline.json"
    if not path.exists():
        raise FileNotFoundError(f"Source timeline not found: {path}")
    return path


def load_source_from_file(path: Path, max_events: int) -> Dict[str, str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = raw.strip()
    if not raw:
        raise ValueError(f"Source file is empty: {path}")

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            if isinstance(data.get("events"), list):
                text = timeline_events_to_text(data["events"], max_events=max_events)
                if text:
                    return {
                        "source_text": text,
                        "entity": normalize_space(str(data.get("entity", ""))),
                        "lang": normalize_space(str(data.get("lang", ""))),
                    }
            for key in ("candidate_narrative", "narrative", "text", "source_text"):
                if isinstance(data.get(key), str) and data[key].strip():
                    return {
                        "source_text": normalize_narrative_text(data[key], max_events=max_events),
                        "entity": "",
                        "lang": "",
                    }

    return {
        "source_text": normalize_narrative_text(raw, max_events=max_events),
        "entity": "",
        "lang": "",
    }


def as_text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                maybe = item.get("text")
                if isinstance(maybe, str):
                    chunks.append(maybe)
        return "\n".join(chunks).strip()
    return str(content or "")


def call_generator(
    client: OpenAI,
    model_id: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: int,
    temperature: float,
    bill_to: str,
    retries: int = 3,
    retry_delay: float = 0.8,
) -> str:
    last_error: Optional[Exception] = None
    headers = {"X-HF-Bill-To": bill_to} if bill_to else None

    for attempt in range(1, retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "top_p": 1,
                "max_completion_tokens": max_completion_tokens,
            }
            if headers:
                kwargs["extra_headers"] = headers
            resp = client.chat.completions.create(**kwargs)
            text = as_text_from_message_content(resp.choices[0].message.content).strip()
            if not text:
                raise ValueError("Empty model content")
            return text
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_delay * attempt)

    raise RuntimeError(f"Generator call failed after retries: {last_error}")


def score_similarity(
    model: SentenceTransformer,
    text_a: str,
    text_b: str,
    batch_size: int,
) -> float:
    emb = model.encode(
        [text_a, text_b],
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return float(np.dot(emb[0], emb[1]))


def attempt_summary_for_prompt(attempts: List[Dict[str, Any]], max_chars: int = 420) -> str:
    if not attempts:
        return "No previous attempts."

    lines: List[str] = []
    for item in attempts[-3:]:
        excerpt = item.get("candidate_narrative", "")[:max_chars]
        excerpt = excerpt.replace("\n", " ")
        lines.append(
            f"- Attempt {item['attempt']}: score={item['score']:.4f}; excerpt={excerpt}"
        )
    return "\n".join(lines)


def build_messages(
    source_text: str,
    adaptation: str,
    threshold: float,
    attempt_idx: int,
    attempts: List[Dict[str, Any]],
    max_events: int,
) -> List[Dict[str, str]]:
    system = (
        "You generate analogous life-cycle narratives. "
        "Return only valid JSON. No markdown. No extra text."
    )
    user = f"""
Goal:
Generate ONE candidate narrative analogous to the source narrative, while respecting the adaptation request.

Current attempt: {attempt_idx}
Target similarity threshold: {threshold:.3f}

Adaptation request:
{adaptation}

Source narrative (format: 'YEAR: event | YEAR: event | ...'):
{source_text}

Previous failed attempts and scores:
{attempt_summary_for_prompt(attempts)}

Hard constraints:
- Output strict JSON with this schema:
  {{
    "candidate_name": "short title or entity name",
    "candidate_narrative": "YEAR: event | YEAR: event | ...",
    "why_this_should_match": "one short sentence"
  }}
- candidate_narrative must be chronological, concise, and include 8 to {max_events} events.
- Keep life-cycle coverage: background, training/early phase, milestones, distinctions, public/personal turning points.
- Keep structural analogies with source (tempo and progression), but do not copy it verbatim.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate analogous narratives with iterative similarity filtering."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-entity", type=str, help="Entity name used to load data/processed/<entity>.<lang>.timeline.json")
    source_group.add_argument("--source-file", type=Path, help="Path to source narrative file (.json or plain text)")
    source_group.add_argument("--source-text", type=str, help="Raw source narrative string")

    parser.add_argument("--source-lang", type=str, default="en", help="Language for --source-entity (default: en)")
    parser.add_argument("--timeline-dir", type=Path, default=DEFAULT_TIMELINE_DIR, help="Timeline directory for --source-entity")
    parser.add_argument("--adaptation", type=str, required=True, help="Requested adaptation to apply")

    parser.add_argument("--generator-model", type=str, default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--similarity-model", type=str, default=DEFAULT_SIMILARITY_MODEL)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--max-tries", type=int, default=5)
    parser.add_argument("--max-events", type=int, default=14)
    parser.add_argument("--max-source-chars", type=int, default=3500)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--bill-to", type=str, default=DEFAULT_BILL_TO)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-retry-delay", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.threshold < -1 or args.threshold > 1:
        raise SystemExit("--threshold must be between -1 and 1 for cosine similarity.")
    if args.max_tries < 1:
        raise SystemExit("--max-tries must be >= 1.")
    if args.max_events < 1:
        raise SystemExit("--max-events must be >= 1.")

    source_meta: Dict[str, str]
    source_path: Optional[Path] = None
    if args.source_entity:
        source_path = find_source_file(args.source_entity, args.source_lang, args.timeline_dir)
        source_meta = load_source_from_file(source_path, max_events=args.max_events)
        if not source_meta.get("entity"):
            source_meta["entity"] = normalize_space(args.source_entity)
        if not source_meta.get("lang"):
            source_meta["lang"] = normalize_space(args.source_lang)
    elif args.source_file:
        if not args.source_file.exists():
            raise SystemExit(f"Source file not found: {args.source_file}")
        source_path = args.source_file
        source_meta = load_source_from_file(args.source_file, max_events=args.max_events)
    else:
        source_meta = {
            "source_text": normalize_narrative_text(args.source_text, max_events=args.max_events),
            "entity": "",
            "lang": "",
        }

    source_text = source_meta.get("source_text", "")
    source_text = source_text[: args.max_source_chars].strip()
    if not source_text:
        raise SystemExit("Could not build a non-empty source narrative text.")

    if not args.api_key:
        raise SystemExit("Missing API key. Set IAFA_INFERENCE in .env or use --api-key.")

    print(f"Loading similarity model: {args.similarity_model}")
    similarity_model = SentenceTransformer(args.similarity_model)
    if args.max_seq_length is not None:
        similarity_model.max_seq_length = args.max_seq_length

    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=args.api_key)

    attempts: List[Dict[str, Any]] = []
    accepted = False
    best_attempt: Optional[Dict[str, Any]] = None

    for attempt_idx in range(1, args.max_tries + 1):
        messages = build_messages(
            source_text=source_text,
            adaptation=args.adaptation,
            threshold=args.threshold,
            attempt_idx=attempt_idx,
            attempts=attempts,
            max_events=args.max_events,
        )

        raw = call_generator(
            client=client,
            model_id=args.generator_model,
            messages=messages,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            bill_to=args.bill_to,
            retries=args.llm_retries,
            retry_delay=args.llm_retry_delay,
        )

        parsed = parse_candidate_output(raw, max_events=args.max_events)
        candidate_text = parsed["candidate_narrative"]
        if not candidate_text:
            candidate_text = normalize_narrative_text(raw, max_events=args.max_events)

        score = score_similarity(
            model=similarity_model,
            text_a=source_text,
            text_b=candidate_text,
            batch_size=args.batch_size,
        )
        is_ok = score >= args.threshold

        attempt_record = {
            "attempt": attempt_idx,
            "score": score,
            "accepted": is_ok,
            "candidate_name": parsed.get("candidate_name", ""),
            "candidate_narrative": candidate_text,
            "rationale": parsed.get("rationale", ""),
            "raw_response": raw,
        }
        attempts.append(attempt_record)

        if best_attempt is None or score > float(best_attempt["score"]):
            best_attempt = attempt_record

        print(
            f"Attempt {attempt_idx}/{args.max_tries}: score={score:.4f} "
            f"(threshold={args.threshold:.4f}) {'ACCEPTED' if is_ok else 'retry'}"
        )

        if is_ok:
            accepted = True
            break

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.output is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_name = datetime.now(timezone.utc).strftime("analogue_result_%Y%m%d_%H%M%S.json")
        out_path = DEFAULT_OUTPUT_DIR / out_name
    else:
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "created_at_utc": timestamp,
        "accepted": accepted,
        "threshold": args.threshold,
        "max_tries": args.max_tries,
        "attempts_count": len(attempts),
        "best_score": None if best_attempt is None else best_attempt["score"],
        "accepted_attempt": next((a["attempt"] for a in attempts if a["accepted"]), None),
        "source": {
            "entity": source_meta.get("entity", ""),
            "lang": source_meta.get("lang", ""),
            "source_file": str(source_path) if source_path else "",
            "source_text": source_text,
        },
        "models": {
            "generator_model": args.generator_model,
            "similarity_model": args.similarity_model,
            "max_seq_length": args.max_seq_length,
        },
        "generation": {
            "adaptation": args.adaptation,
            "temperature": args.temperature,
            "max_completion_tokens": args.max_completion_tokens,
            "max_events": args.max_events,
        },
        "best_attempt": best_attempt,
        "attempts": attempts,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved result: {out_path}")


if __name__ == "__main__":
    main()
