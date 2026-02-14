import argparse
import json
import os
import re
import shutil
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)
from openai import OpenAI


CHUNK_DIR = Path("data/chunks")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IAFA_KEY = os.getenv("IAFA_INFERENCE", "")
MODEL_ID = os.getenv("MODEL_ID", "")
BILL_TO = os.getenv("BILL_TO", "IAFA-UT")
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "1800"))
MIN_EVENTS_TOTAL = int(os.getenv("MIN_EVENTS_TOTAL", "15"))
MAX_EVENTS_TOTAL = int(os.getenv("MAX_EVENTS_TOTAL", "40"))
MAX_EVENTS_PER_CHUNK = int(os.getenv("MAX_EVENTS_PER_CHUNK", "10"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "200"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "3"))
LLM_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "0.8"))
EVIDENCE_MAX_CHARS = int(os.getenv("EVIDENCE_MAX_CHARS", "90"))
CURRENT_YEAR = datetime.now(timezone.utc).year
SAVE_RAW_SUCCESS_DEFAULT = os.getenv("SAVE_RAW_SUCCESS", "").strip().lower() in {"1", "true", "yes", "y"}

if not IAFA_KEY:
    raise SystemExit("Missing IAFA_INFERENCE in .env")


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=IAFA_KEY,
)

def call_llm(messages):
    last_error = None
    for attempt in range(1, LLM_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=0,
                top_p=1,
                extra_headers={"X-HF-Bill-To": BILL_TO},
            )
            content = resp.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty model content")
            return content
        except Exception as exc:
            last_error = exc
            if attempt >= LLM_RETRIES:
                break
            time.sleep(LLM_RETRY_DELAY * attempt)
    raise last_error

def extract_json_block(text: str) -> str:
    """
    Essaie d'isoler un bloc JSON meme si le modele ajoute du texte autour.
    Extraction robuste par scan (respecte les chaines et les echappements).
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    brace_count = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
        else:
            if ch == "\"":
                in_string = True
            elif ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i + 1]

    raise ValueError("No complete JSON object found in model output")

def sanitize_json(text: str) -> str:
    """
    Repare quelques erreurs JSON frequentes dans la sortie LLM.
    - Echappe les backslashes invalides
    - Normalise les sauts de ligne
    """
    text = text.replace("```json", "").replace("```", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Echappe les backslashes non valides
    text = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text)
    return text

def build_prompt(entity: str, lang: str, chunk_text: str) -> List[Dict[str, str]]:
    system = (
        "You are an information extraction system. "
        "Return ONLY valid JSON (RFC 8259). "
        "No extra text, no markdown. "
        "Use double quotes for all keys and string values. "
        "Do not use single quotes. "
        "Do not include trailing commas."
    )

    user = f"""
Entity: {entity}
Language: {lang}

Goal:
Extract a LIFE-CYCLE timeline (not only filmography): birth, family/background, education/training,
career milestones, awards/distinctions, humanitarian roles, and major personal public events if present.

Output (STRICT JSON ONLY):
{{
  "entity": "{entity}",
  "lang": "{lang}",
  "events": [
    {{"year": 0 or null, "event": "short fact", "evidence": "short excerpt"}}
  ]
}}

Hard requirements (must include IF present in the text):
1) Birth (date/place) -> set year to birth year
2) Education / training (schools, institutes) -> year if present else null
3) First career milestone (debut / first leading role) -> year if present else null
4) Major awards / nominations -> year if present
5) Humanitarian role (e.g., UNHCR Special Envoy) -> year if present else null
6) Major legal/public changes (name change, marriage/divorce) -> year if present

Rules:
- Only facts explicitly present in the provided text (no invention).
- Prefer covering different life stages over listing many similar films.
- If a sentence lists multiple works/years, extract SEPARATE events with the correct year for each.
- At most {MAX_EVENTS_PER_CHUNK} events for this chunk.
- "event" <= 120 chars.
- "evidence" must be a SHORT substring copied from the text (<= {EVIDENCE_MAX_CHARS} chars).
- "evidence" should not use ellipsis ("..." or "…").
- If the year isn't explicit, set year=null.
- Keep "entity" exactly "{entity}" and "lang" exactly "{lang}".
- Return JSON only.

Text:
\"\"\"{chunk_text}\"\"\"
""".strip()

    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]



YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
WS_RE = re.compile(r"\s+")
HEADER_ENTITY_RE = re.compile(r'"entity"\s*:\s*"([^"]+)"')
HEADER_LANG_RE = re.compile(r'"lang"\s*:\s*"([^"]+)"')


def normalize_space(text: str) -> str:
    return WS_RE.sub(" ", text or "").strip()


def extract_years(text: str) -> List[int]:
    return [int(y) for y in YEAR_RE.findall(text or "")]


def normalize_event(e: Dict[str, Any]) -> Dict[str, Any]:
    year = e.get("year", None)
    if isinstance(year, float) and year.is_integer():
        year = int(year)
    elif isinstance(year, str):
        y = year.strip()
        if y.isdigit():
            year = int(y)
        else:
            m_year = YEAR_RE.search(y)
            year = int(m_year.group(0)) if m_year else None

    event = normalize_space(e.get("event", "") or "")
    evidence = normalize_space(e.get("evidence", "") or "")

    if evidence.startswith('"') and evidence.endswith('"') and len(evidence) >= 2:
        evidence = evidence[1:-1].strip()

    if year is None and evidence:
        years_in_evidence = extract_years(evidence)
        if len(years_in_evidence) == 1:
            year = years_in_evidence[0]

    # Fallback: si l'evidence ne contient pas d'annee explicite, on accepte une
    # annee unique presente dans l'enonce de l'evenement.
    if year is None and event:
        years_in_event = extract_years(event)
        if len(years_in_event) == 1:
            year = years_in_event[0]
        elif len(years_in_event) == 2:
            y0, y1 = years_in_event[0], years_in_event[1]
            if y0 <= y1 and (y1 - y0) <= 20:
                year = y0

    if isinstance(year, int) and (year < 1800 or year > CURRENT_YEAR + 1):
        year = None

    if len(event) > 120:
        event = event[:120].rstrip()
    if len(evidence) > EVIDENCE_MAX_CHARS:
        evidence = evidence[:EVIDENCE_MAX_CHARS].rstrip()

    return {"year": year, "event": event, "evidence": evidence}


def normalize_text_key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").lower()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = WS_RE.sub(" ", text).strip()
    return text


def evidence_supported_by_chunk(evidence: str, chunk_text: str) -> bool:
    evidence_norm = normalize_space(evidence)
    if not evidence_norm:
        return False

    chunk_norm = normalize_space(chunk_text).lower()
    ev = evidence_norm.lower()
    if ev in chunk_norm:
        return True

    for suffix in ("...", "…"):
        if ev.endswith(suffix):
            prefix = ev[: -len(suffix)].rstrip()
            if len(prefix) >= 20 and prefix in chunk_norm:
                return True
    return False


def raw_header_info(raw: str) -> Tuple[Optional[str], Optional[str]]:
    m_entity = HEADER_ENTITY_RE.search(raw or "")
    m_lang = HEADER_LANG_RE.search(raw or "")
    entity = m_entity.group(1).strip() if m_entity else None
    lang = m_lang.group(1).strip() if m_lang else None
    return entity, lang


def raw_header_matches_expected(raw: str, entity: str, lang: str) -> bool:
    raw_entity, raw_lang = raw_header_info(raw)
    if raw_entity and normalize_text_key(raw_entity) != normalize_text_key(entity):
        return False
    if raw_lang and raw_lang.lower() != lang.lower():
        return False
    return True

def is_event_plausible(e: Dict[str, Any]) -> bool:
    event = (e.get("event") or "").strip()
    evidence = (e.get("evidence") or "").strip()
    year = e.get("year")

    if not event or not evidence:
        return False
    if len(event) < 8:
        return False
    if isinstance(year, int):
        years_in_evidence = {int(y) for y in re.findall(r"(?:18|19|20)\d{2}", evidence)}
        if years_in_evidence and year not in years_in_evidence:
            return False
        years_in_event = {int(y) for y in re.findall(r"(?:18|19|20)\d{2}", event)}
        if years_in_event and year not in years_in_event:
            return False
    return True

def event_quality_score(e: Dict[str, Any]) -> int:
    score = 0
    event = (e.get("event") or "").lower()
    evidence = e.get("evidence") or ""
    year = e.get("year")

    if len(event) >= 20:
        score += 1
    if len(evidence) >= 30:
        score += 1

    if isinstance(year, int):
        years_in_evidence = {int(y) for y in re.findall(r"(?:18|19|20)\d{2}", evidence)}
        if years_in_evidence:
            score += 2 if year in years_in_evidence else -3

    milestone_words = (
        "born", "debut", "first", "won", "award", "married", "divorce",
        "adopt", "direct", "appointed", "promoted", "starred", "mastectomy",
        "naissance", "nomination", "prix", "oscar", "mariage", "divorc",
        "réalis", "humanitaire", "baptisé", "baptized",
    )
    if any(w in event for w in milestone_words):
        score += 1

    noisy_words = ("bisexual", "kissed", "wild child", "tabloid", "rumeur")
    if any(w in event for w in noisy_words):
        score -= 3

    return score

def dedup_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dedup simple par (year, event lower). Suffisant pour MVP.
    """
    seen = set()
    seen_evidence = set()
    out = []
    for e in events:
        if not is_event_plausible(e):
            continue
        year = e.get("year")
        event_key = normalize_text_key(e.get("event", ""))
        evidence_key = normalize_text_key(e.get("evidence", ""))
        key = (year, event_key)
        ev_key = (year, evidence_key[:EVIDENCE_MAX_CHARS])
        if key in seen:
            continue
        if ev_key in seen_evidence:
            continue
        if not e.get("event"):
            continue
        seen.add(key)
        seen_evidence.add(ev_key)
        out.append(e)
    return out

def sort_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tri chronologique: annees connues d'abord (asc), puis sans annee.
    """
    return sorted(
        events,
        key=lambda e: (
            e.get("year") is None,
            e.get("year") if isinstance(e.get("year"), int) else 9999,
            (e.get("event") or "").lower(),
        ),
    )

def cap_events_with_coverage(events: List[Dict[str, Any]], max_total: int) -> List[Dict[str, Any]]:
    """
    Evite qu'un debut de timeline monopolise tout le budget:
    - 1 meilleur evenement par annee d'abord
    - puis remplissage par score de qualite.
    """
    events = sort_events(events)
    if len(events) <= max_total:
        return events

    by_year: Dict[int, List[Dict[str, Any]]] = {}
    without_year: List[Dict[str, Any]] = []
    for e in events:
        y = e.get("year")
        if isinstance(y, int):
            by_year.setdefault(y, []).append(e)
        else:
            without_year.append(e)

    selected: List[Dict[str, Any]] = []
    used_ids = set()

    for y in sorted(by_year.keys()):
        best = sorted(by_year[y], key=event_quality_score, reverse=True)[0]
        selected.append(best)
        used_ids.add(id(best))
        if len(selected) >= max_total:
            return sort_events(selected[:max_total])

    rest: List[Dict[str, Any]] = []
    for y in sorted(by_year.keys()):
        for e in by_year[y]:
            if id(e) not in used_ids:
                rest.append(e)
    rest.extend(without_year)

    rest = sorted(
        rest,
        key=lambda e: (
            -event_quality_score(e),
            e.get("year") is None,
            e.get("year") if isinstance(e.get("year"), int) else 9999,
        ),
    )

    for e in rest:
        if len(selected) >= max_total:
            break
        selected.append(e)

    return sort_events(selected[:max_total])

def salvage_events_from_partial(raw: str) -> List[Dict[str, Any]]:
    """
    Fallback: recupere les objets event complets deja presents
    meme si la sortie est tronquee avant la fin du JSON global.
    """
    marker = raw.find("\"events\"")
    if marker == -1:
        return []

    arr_start = raw.find("[", marker)
    if arr_start == -1:
        return []

    s = raw[arr_start + 1 :]
    in_string = False
    escape = False
    depth = 0
    obj_start = None
    events: List[Dict[str, Any]] = []

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue

        if ch == "\"":
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
            continue

        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start is not None:
                    cand = s[obj_start:i + 1]
                    try:
                        obj = json.loads(cand)
                    except json.JSONDecodeError:
                        try:
                            obj = json.loads(sanitize_json(cand))
                        except json.JSONDecodeError:
                            obj = None
                    if isinstance(obj, dict):
                        events.append(obj)
                    obj_start = None
            continue

        if ch == "]" and depth == 0:
            break

    return events

def parse_timeline(raw: str, chunk_text: str = "") -> Dict[str, Any]:
    data: Optional[Dict[str, Any]] = None

    try:
        js = extract_json_block(raw)
    except ValueError:
        js = None

    if js is not None:
        for candidate in (js, sanitize_json(js)):
            try:
                loaded = json.loads(candidate)
                if isinstance(loaded, dict):
                    data = loaded
                    break
            except json.JSONDecodeError:
                continue

    if data is None:
        partial_events = salvage_events_from_partial(raw)
        if not partial_events:
            raise ValueError("Failed to parse model output as JSON and could not salvage events.")
        data = {"events": partial_events}

    events = [normalize_event(e) for e in data.get("events", []) if isinstance(e, dict)]
    if chunk_text:
        events = [e for e in events if evidence_supported_by_chunk(e.get("evidence", ""), chunk_text)]
    events = dedup_events(events)
    events = cap_events_with_coverage(events, MAX_EVENTS_PER_CHUNK)
    data["events"] = events
    return data


def is_noise_chunk(chunk_text: str) -> bool:
    text = normalize_space(chunk_text).lower()
    if not text:
        return True
    if len(text) < 300 and "portail" in text:
        return True

    noise_markers = (
        "== external links ==",
        "== see also ==",
        "== notes et références ==",
        "=== liens externes ===",
        "=== références ===",
    )
    if any(marker in text for marker in noise_markers) and len(text) < 1200:
        return True
    return False

def write_debug_paths(entity_slug: str, lang: str, fp_name: str) -> Tuple[Path, Path]:
    err = OUT_DIR / f"{entity_slug}.{lang}.{fp_name}.error.txt"
    raw_path = OUT_DIR / f"{entity_slug}.{lang}.{fp_name}.raw.txt"
    return err, raw_path


def archive_existing_debug_files() -> Optional[Path]:
    debug_files = sorted(OUT_DIR.glob("*.raw.txt")) + sorted(OUT_DIR.glob("*.error.txt"))
    if not debug_files:
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_dir = OUT_DIR / "archive" / stamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    for src in debug_files:
        dst = archive_dir / src.name
        if dst.exists():
            base = src.stem
            suffix = src.suffix
            k = 1
            while (archive_dir / f"{base}.{k}{suffix}").exists():
                k += 1
            dst = archive_dir / f"{base}.{k}{suffix}"
        shutil.move(str(src), str(dst))
    return archive_dir


def extract_for_chunk(fp: Path, save_raw_success: bool = SAVE_RAW_SUCCESS_DEFAULT) -> None:
    parts = fp.stem.split(".")
    if len(parts) < 3:
        raise SystemExit(f"Invalid chunk filename: {fp.name}")
    entity_slug = parts[0]
    lang = parts[1]
    entity_name = entity_slug.replace("_", " ")

    chunk_text = fp.read_text(encoding="utf-8", errors="ignore").strip()
    if len(chunk_text) < MIN_CHUNK_CHARS:
        raise SystemExit(f"Chunk too short: {fp}")
    if is_noise_chunk(chunk_text):
        raise SystemExit(f"Chunk appears non-biographical/noisy: {fp}")

    messages = build_prompt(entity_name, lang, chunk_text)
    err, raw_path = write_debug_paths(entity_slug, lang, fp.name)

    try:
        raw = call_llm(messages)
        if not raw_header_matches_expected(raw, entity_name, lang):
            raise ValueError("Model output entity/lang header does not match requested chunk.")
        timeline = parse_timeline(raw, chunk_text=chunk_text)
        if save_raw_success:
            raw_path.write_text(raw, encoding="utf-8")
        events = cap_events_with_coverage(timeline.get("events", []), MAX_EVENTS_TOTAL)
        out = {
            "entity": entity_name,
            "lang": lang,
            "events": events
        }
        out_path = OUT_DIR / f"{entity_slug}.{lang}.timeline.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        n_events = len(out["events"])
        print(f"Saved: {out_path}  events={n_events}")
        if n_events < MIN_EVENTS_TOTAL:
            print(f"Warning: events below target ({n_events} < {MIN_EVENTS_TOTAL})")
    except Exception as e:
        err.write_text(str(e), encoding="utf-8")
        try:
            raw_path.write_text(raw, encoding="utf-8")
        except Exception:
            pass
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=str, help="Path to a single chunk file")
    parser.add_argument("--save-raw-success", action="store_true", help="Save .raw.txt for successful chunk extractions")
    parser.add_argument("--archive-debug-files", action="store_true", help="Archive existing .raw.txt/.error.txt in data/processed before extraction")
    parser.add_argument("--archive-only", action="store_true", help="Archive debug files and exit")
    args = parser.parse_args()
    save_raw_success = args.save_raw_success or SAVE_RAW_SUCCESS_DEFAULT

    if args.archive_debug_files:
        archived_to = archive_existing_debug_files()
        if archived_to:
            print(f"Archived debug files to: {archived_to}")
        else:
            print("No debug files to archive.")
        if args.archive_only:
            return

    if args.chunk:
        fp = Path(args.chunk)
        if not fp.is_file():
            raise SystemExit(f"Chunk file not found: {fp}")
        extract_for_chunk(fp, save_raw_success=save_raw_success)
        return

    # On traite chunk par chunk, puis on fusionne au niveau (entity,lang).
    chunks = sorted(CHUNK_DIR.glob("*.chunk*.txt"))
    if not chunks:
        raise SystemExit("No chunks found in data/chunks. Run chunk_texts.py first.")

    grouped: Dict[tuple, List[Path]] = {}
    for fp in chunks:
        # fp.stem exemple: Angelina_Jolie.fr.chunk001
        parts = fp.stem.split(".")
        if len(parts) < 3:
            continue
        entity_slug = parts[0]            # Angelina_Jolie
        lang = parts[1]                   # fr
        grouped.setdefault((entity_slug, lang), []).append(fp)

    for (entity_slug, lang), files in grouped.items():
        all_events: List[Dict[str, Any]] = []
        entity_name = entity_slug.replace("_", " ")

        for fp in tqdm(files, desc=f"Extract {entity_slug}.{lang}"):
            chunk_text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue
            if is_noise_chunk(chunk_text):
                continue

            messages = build_prompt(entity_name, lang, chunk_text)
            err, raw_path = write_debug_paths(entity_slug, lang, fp.name)

            try:
                raw = call_llm(messages)
                if not raw_header_matches_expected(raw, entity_name, lang):
                    raise ValueError("Model output entity/lang header does not match requested chunk.")
                timeline = parse_timeline(raw, chunk_text=chunk_text)
                if save_raw_success:
                    raw_path.write_text(raw, encoding="utf-8")
                all_events.extend(timeline.get("events", []))
            except Exception as e:
                # log d'erreur pour debug + sortie brute
                err.write_text(str(e), encoding="utf-8")
                try:
                    raw_path.write_text(raw, encoding="utf-8")
                except Exception:
                    pass
                continue

        all_events = dedup_events([normalize_event(e) for e in all_events])
        all_events = cap_events_with_coverage(all_events, MAX_EVENTS_TOTAL)

        out = {
            "entity": entity_name,
            "lang": lang,
            "events": all_events
        }
        out_path = OUT_DIR / f"{entity_slug}.{lang}.timeline.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        n_events = len(all_events)
        print(f"Saved: {out_path}  events={n_events}")
        if n_events < MIN_EVENTS_TOTAL:
            print(f"Warning: events below target ({n_events} < {MIN_EVENTS_TOTAL})")

if __name__ == "__main__":
    main()
