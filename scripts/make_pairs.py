import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

PROCESSED_DIR = Path("data/processed")
OUT_DIR = Path("data/datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANGS = ("en", "fr")
SEED = 42
POS_PER_ENTITY = 1          # EN-FR same entity = 1 pair
NEG_MULTIPLIER = 3          # negatives = POS * 3 (MVP). Increase later.
MAX_EVENTS_USED = 35        # keep narratives compact
MIN_EVENTS_REQUIRED = 5    # skip weak timelines

random.seed(SEED)

def load_timelines() -> Dict[Tuple[str, str], Dict]:
    """
    Loads files like Entity_Slug.en.timeline.json
    Returns dict keyed by (entity_slug, lang).
    """
    timelines = {}
    for fp in PROCESSED_DIR.glob("*.timeline.json"):
        name = fp.name  # e.g. Angelina_Jolie.en.timeline.json
        parts = name.split(".")
        if len(parts) < 4:
            continue
        entity_slug, lang = parts[0], parts[1]
        if lang not in LANGS:
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        events = data.get("events", [])
        if not isinstance(events, list) or len(events) < MIN_EVENTS_REQUIRED:
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

def timeline_to_text(tl: Dict) -> str:
    events = tl.get("events", [])[:MAX_EVENTS_USED]
    lines = [event_to_line(e) for e in events if (e.get("event") or "").strip()]
    # keep it as a narrative-like paragraph
    return " | ".join(lines)

def build_positive_pairs(timelines: Dict[Tuple[str, str], Dict]) -> List[Dict]:
    pos = []
    # entities that have both en and fr
    entities = sorted({slug for (slug, lang) in timelines.keys()
                       if (slug, "en") in timelines and (slug, "fr") in timelines})
    for slug in entities:
        en = timelines[(slug, "en")]
        fr = timelines[(slug, "fr")]
        pos.append({
            "id": f"{slug}_en_fr",
            "label": 1,
            "entity_a": slug,
            "lang_a": "en",
            "text_a": timeline_to_text(en),
            "entity_b": slug,
            "lang_b": "fr",
            "text_b": timeline_to_text(fr),
            "pair_type": "pos_same_entity_en_fr",
        })
    return pos

def build_negative_pairs(pos_pairs: List[Dict], timelines: Dict[Tuple[str, str], Dict]) -> List[Dict]:
    # candidate timelines list
    candidates = [(slug, lang) for (slug, lang) in timelines.keys()]
    neg = []
    target = len(pos_pairs) * NEG_MULTIPLIER

    # To keep negatives meaningful, we sample different entities, same language (en-en or fr-fr) half the time.
    while len(neg) < target:
        if random.random() < 0.5:
            # same language negative
            lang = random.choice(LANGS)
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
            "text_a": timeline_to_text(a_tl),
            "entity_b": b_slug,
            "lang_b": b_lang,
            "text_b": timeline_to_text(b_tl),
            "pair_type": "neg_diff_entity",
        })

    return neg

def main():
    timelines = load_timelines()
    print(f"Loaded timelines: {len(timelines)} files")

    pos = build_positive_pairs(timelines)
    print(f"Positive pairs: {len(pos)} (entities with both EN+FR)")

    neg = build_negative_pairs(pos, timelines)
    print(f"Negative pairs: {len(neg)} (multiplier={NEG_MULTIPLIER})")

    pairs = pos + neg
    random.shuffle(pairs)

    out_jsonl = OUT_DIR / "pairs_mvp.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # small stats file
    stats = {
        "timelines_loaded": len(timelines),
        "positive_pairs": len(pos),
        "negative_pairs": len(neg),
        "total_pairs": len(pairs),
        "neg_multiplier": NEG_MULTIPLIER,
        "max_events_used": MAX_EVENTS_USED,
        "min_events_required": MIN_EVENTS_REQUIRED,
        "langs": list(LANGS),
    }
    (OUT_DIR / "pairs_mvp.stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved: {out_jsonl}")
    print(f"Saved: {OUT_DIR / 'pairs_mvp.stats.json'}")

if __name__ == "__main__":
    main()
