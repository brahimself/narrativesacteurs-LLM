import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import gradio as gr


ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "scripts" / "generate_analogues.py"
TIMELINE_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "data" / "results"
SIMILARITY_MODEL_CHOICES = [
    "sidbrahim/narrativesAnalogues-MPNet",
    "sidbrahim/narrativesAnalogues-allMiniLM",
]
CUSTOM_MODEL_OPTION = "custom"
DEFAULT_SIMILARITY_MODEL = SIMILARITY_MODEL_CHOICES[0]
DEFAULT_THRESHOLD_BY_MODEL = {
    "sidbrahim/narrativesAnalogues-MPNet": 0.46,
    "sidbrahim/narrativesAnalogues-allMiniLM": 0.28,
}
DEFAULT_THRESHOLD = DEFAULT_THRESHOLD_BY_MODEL[DEFAULT_SIMILARITY_MODEL]
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTATION_PRESET_CUSTOM = "Custom (write your own instruction)"
ADAPTATION_PRESETS = [
    "Keep the same career arc and pacing, but move the person into European arthouse cinema with regular Cannes and Venice entries.",
    "Preserve early struggles and breakthrough structure, but adapt milestones toward socially engaged documentaries and human-rights advocacy.",
    "Keep a mainstream-to-prestige trajectory, but shift the narrative to stage acting first, then prestige TV, then auteur films.",
    "Maintain the same rhythm of awards and setbacks, but adapt achievements toward international co-productions and multilingual roles.",
    "Keep personal-life turning points analogous, but adapt professional milestones toward animation voice acting and family-oriented franchises.",
    "Preserve timeline density and chronology, but reframe the profile as an actor-producer focused on climate and sustainability themes.",
    "Keep the same rise-fall-recovery pattern, but adapt it to an independent film circuit with Sundance and Berlinale breakthroughs.",
    "Maintain similar career longevity and turning points, but orient the narrative to action-thriller franchises and stunt-driven roles.",
    "Keep early training and mentorship structure, but adapt later years to directing and screenwriting recognition.",
    "Preserve major public recognition moments, but adapt the domain from film-first to streaming-series-first international visibility.",
    "Keep the same number of major milestones, but make the trajectory centered on biopics and historical dramas.",
    "Maintain the same balance of professional and personal events, but adapt public image toward philanthropy and educational initiatives.",
    "Keep breakthrough timing and award cadence, but shift geography from Hollywood-centered to UK-France-Italy collaborations.",
    "Preserve the structure of critical acclaim followed by commercial success, but adapt genres toward psychological drama and noir.",
    "Keep analogous career inflection points, but adapt controversies into reputation recovery through selective high-quality projects.",
    "Maintain family-background influence, but adapt the person into a first-generation artist who builds a career through scholarships and theater.",
    "Keep the arc of rapid fame then strategic slowdown, but adapt to a profile that prioritizes selective indie projects over blockbusters.",
    "Preserve early supporting roles before lead status, but adapt toward science-fiction and speculative cinema.",
    "Keep career acceleration in the 20s, but adapt peak recognition in the 30s through festival-driven performances.",
    "Maintain the same number of award events, but adapt them from US institutions to European and Asian festivals.",
    "Keep the same chronology, but adapt milestones to include periodic career breaks for activism and social campaigns.",
    "Preserve the arc of collaboration with one key director, but adapt that collaboration to two recurring auteur partners.",
    "Keep education-to-career transition similar, but adapt training to conservatory theater and classical acting workshops.",
    "Maintain a similar media visibility curve, but adapt public communication around mental health and anti-harassment advocacy.",
    "Preserve first major success timing, but adapt the breakthrough title into a politically themed drama.",
    "Keep the same pattern of international recognition, but adapt primary markets toward Latin America and Europe.",
    "Maintain analogous personal turning points, but adapt the narrative to minimize gossip and focus on professional craft.",
    "Keep a comparable list of career highs, but adapt lows to include box-office failures followed by critic-led comeback.",
    "Preserve long-term relevance, but adapt later career to mentoring younger performers and producing debut films.",
    "Keep timeline granularity and event count, but adapt content toward TV miniseries and anthology formats.",
    "Maintain the same pacing of life events, but adapt key milestones toward legal advocacy and public policy engagement.",
    "Keep a similar number of collaborations, but adapt collaborators to international female directors and writers.",
    "Preserve breakthrough and consolidation phases, but adapt signature genre to dark comedy and satire.",
    "Keep career turning points analogous, but adapt domain to choreography, dance films, and performance art cinema.",
    "Maintain broad audience appeal, but adapt public persona toward low-profile, craft-first communication.",
    "Keep the same chronology, but adapt to a transnational career split between London, Paris, and Seoul.",
    "Preserve award cadence, but adapt award types toward ensemble cast and screenplay-oriented honors.",
    "Keep the trajectory from newcomer to established figure, but adapt the path through recurring supporting roles before lead fame.",
    "Maintain the same number of major projects, but adapt focus toward literary adaptations and period pieces.",
    "Keep crisis and comeback structure, but adapt comeback trigger to an acclaimed limited series role.",
    "Preserve early commercial projects, but adapt mid-career pivot toward activist documentaries and public speaking.",
    "Maintain personal stability events, but adapt professional experimentation with genre switches every 3-4 years.",
    "Keep timeline shape similar, but adapt outcomes toward teaching, workshops, and film-school partnerships.",
    "Preserve high-visibility milestones, but adapt them around international jury memberships and festival leadership roles.",
    "Maintain the same sequence of growth, but adapt to a profile balancing acting with entrepreneurship in film-tech.",
    "Keep breakout and recognition timing, but adapt the arc toward underrepresented-language cinema and subtitled global hits.",
    "Preserve the pattern of one iconic role, but adapt to two medium-impact roles distributed across film and series.",
    "Maintain a consistent year-by-year rhythm, but adapt to include major charity campaign leadership and NGO partnerships.",
]


def timeline_events_to_text(events: List[Dict[str, Any]], max_events: int) -> str:
    lines: List[str] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        event = str(item.get("event", "")).strip()
        if not event:
            continue
        year = item.get("year")
        year_str = str(year) if isinstance(year, int) else "NA"
        lines.append(f"{year_str}: {event}")
    if max_events > 0:
        lines = lines[:max_events]
    return " | ".join(lines)


def scan_timelines(timeline_dir: Path) -> Dict[str, Set[str]]:
    entity_langs: Dict[str, Set[str]] = {}
    if not timeline_dir.exists():
        return entity_langs

    for fp in timeline_dir.glob("*.timeline.json"):
        parts = fp.name.split(".")
        # expected: <entity_slug>.<lang>.timeline.json
        if len(parts) < 4:
            continue
        if parts[-2] != "timeline" or parts[-1] != "json":
            continue
        lang = parts[-3].strip()
        entity_slug = ".".join(parts[:-3]).strip()
        if not entity_slug or not lang:
            continue
        entity_langs.setdefault(entity_slug, set()).add(lang)

    return entity_langs


def slug_to_display(slug: str) -> str:
    return slug.replace("_", " ").strip()


def build_display_maps(entity_langs: Dict[str, Set[str]]) -> Dict[str, str]:
    display_to_slug: Dict[str, str] = {}
    for slug in sorted(entity_langs.keys()):
        display = slug_to_display(slug)
        # In case of rare display collisions, keep deterministic first insert.
        display_to_slug.setdefault(display, slug)
    return display_to_slug


ENTITY_LANGS = scan_timelines(TIMELINE_DIR)
DISPLAY_TO_SLUG = build_display_maps(ENTITY_LANGS)
DISPLAY_CHOICES = sorted(DISPLAY_TO_SLUG.keys())
DEFAULT_ENTITY = "Leonardo DiCaprio" if "Leonardo DiCaprio" in DISPLAY_TO_SLUG else (DISPLAY_CHOICES[0] if DISPLAY_CHOICES else "")


def langs_for_entity_display(entity_display: str) -> List[str]:
    slug = DISPLAY_TO_SLUG.get(entity_display, "")
    langs = sorted(ENTITY_LANGS.get(slug, set()))
    return langs or ["en", "fr"]


def on_entity_change(entity_display: str):
    langs = langs_for_entity_display(entity_display)
    return gr.update(choices=langs, value=langs[0])


def load_source_preview(entity_display: str, source_lang: str, max_events: int) -> str:
    source_slug = DISPLAY_TO_SLUG.get(entity_display, "")
    if not source_slug:
        return ""

    timeline_path = TIMELINE_DIR / f"{source_slug}.{source_lang}.timeline.json"
    if not timeline_path.exists():
        return f"Timeline not found: {timeline_path.name}"

    try:
        payload = json.loads(timeline_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Failed to read source timeline ({timeline_path.name}): {exc}"

    events = payload.get("events", [])
    text = timeline_events_to_text(events, max_events=max_events)
    if not text:
        return "No valid events found in source timeline."
    return text


def on_entity_change_with_preview(entity_display: str, max_events: int):
    langs = langs_for_entity_display(entity_display)
    selected_lang = langs[0]
    preview = load_source_preview(entity_display, selected_lang, max_events=max_events)
    return gr.update(choices=langs, value=selected_lang), gr.update(value=preview)


def on_source_context_change(entity_display: str, source_lang: str, max_events: int):
    return gr.update(value=load_source_preview(entity_display, source_lang, max_events=max_events))


def on_similarity_model_change(selected_model: str):
    model_id = (selected_model or "").strip()
    if model_id in DEFAULT_THRESHOLD_BY_MODEL:
        return gr.update(value=DEFAULT_THRESHOLD_BY_MODEL[model_id])
    return gr.update()


def on_adaptation_preset_change(selected_preset: str, current_text: str):
    preset = (selected_preset or "").strip()
    if not preset or preset == ADAPTATION_PRESET_CUSTOM:
        return gr.update(value=current_text or "")
    return gr.update(value=preset)


def run_generation(
    entity_display: str,
    source_lang: str,
    adaptation: str,
    threshold: float,
    max_tries: int,
    max_events: int,
    temperature: float,
    max_completion_tokens: int,
    generator_model: str,
    similarity_model: str,
    similarity_model_custom: str,
):
    if not SCRIPT_PATH.exists():
        raise gr.Error(f"Missing script: {SCRIPT_PATH}")

    if not entity_display:
        raise gr.Error("Select a source entity.")

    adaptation = (adaptation or "").strip()
    if not adaptation:
        raise gr.Error("Adaptation instruction is required.")

    similarity_model = (similarity_model or "").strip()
    if similarity_model == CUSTOM_MODEL_OPTION:
        similarity_model = (similarity_model_custom or "").strip()
    if not similarity_model:
        raise gr.Error("Similarity model is required.")

    source_slug = DISPLAY_TO_SLUG.get(entity_display)
    if not source_slug:
        raise gr.Error("Unknown entity selected.")

    allowed_langs = ENTITY_LANGS.get(source_slug, set())
    if allowed_langs and source_lang not in allowed_langs:
        raise gr.Error(f"Language '{source_lang}' not available for '{entity_display}'.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"space_run_{stamp}.json"

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--source-entity",
        entity_display,
        "--source-lang",
        source_lang,
        "--adaptation",
        adaptation,
        "--threshold",
        str(float(threshold)),
        "--max-tries",
        str(int(max_tries)),
        "--max-events",
        str(int(max_events)),
        "--temperature",
        str(float(temperature)),
        "--max-completion-tokens",
        str(int(max_completion_tokens)),
        "--generator-model",
        generator_model.strip(),
        "--similarity-model",
        similarity_model.strip(),
        "--output",
        str(output_path),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )

    logs = (proc.stdout or "")
    if proc.stderr:
        logs = f"{logs}\n[stderr]\n{proc.stderr}".strip()

    if proc.returncode != 0:
        raise gr.Error(f"Generation failed.\n\n{logs[-4000:]}")

    if not output_path.exists():
        raise gr.Error("Generation finished but output JSON was not created.")

    result = json.loads(output_path.read_text(encoding="utf-8"))

    accepted = bool(result.get("accepted"))
    best_score = result.get("best_score")
    attempts_count = result.get("attempts_count")
    accepted_attempt = result.get("accepted_attempt")

    summary = (
        f"**Accepted:** `{accepted}`\n"
        f"**Best Score:** `{best_score:.4f}`\n" if isinstance(best_score, (int, float)) else f"**Accepted:** `{accepted}`\n**Best Score:** `N/A`\n"
    )
    summary += (
        f"**Attempts:** `{attempts_count}`\n"
        f"**Accepted Attempt:** `{accepted_attempt}`\n"
        f"**Output File:** `{output_path}`"
    )

    best_attempt = result.get("best_attempt") or {}
    best_narrative = str(best_attempt.get("candidate_narrative", "")).strip()

    rows = []
    for a in result.get("attempts", []):
        rows.append([
            a.get("attempt"),
            round(float(a.get("score", 0.0)), 4),
            bool(a.get("accepted", False)),
        ])

    return summary, best_narrative, rows, result, logs[-4000:]


with gr.Blocks(title="Narrative Analogues Generator") as demo:
    gr.Markdown("# Narrative Analogues Generator (MVP)")
    gr.Markdown("Generate candidate narratives and validate them with similarity thresholding.")

    with gr.Row():
        entity_input = gr.Dropdown(
            label="Source Entity",
            choices=DISPLAY_CHOICES,
            value=DEFAULT_ENTITY,
            interactive=True,
        )
        lang_input = gr.Radio(
            label="Source Language",
            choices=langs_for_entity_display(DEFAULT_ENTITY),
            value=langs_for_entity_display(DEFAULT_ENTITY)[0],
            interactive=True,
        )

    source_preview_output = gr.Textbox(
        label="Source Timeline Preview",
        lines=8,
        interactive=False,
        value=load_source_preview(
            DEFAULT_ENTITY,
            langs_for_entity_display(DEFAULT_ENTITY)[0],
            max_events=14,
        ),
    )

    adaptation_preset_input = gr.Dropdown(
        label="Adaptation Preset",
        choices=[ADAPTATION_PRESET_CUSTOM] + ADAPTATION_PRESETS,
        value=ADAPTATION_PRESET_CUSTOM,
        interactive=True,
    )

    adaptation_input = gr.Textbox(
        label="Adaptation Instruction",
        lines=4,
        placeholder="Select a preset above or write your own instruction.",
    )

    with gr.Row():
        threshold_input = gr.Slider(0.1, 0.95, value=DEFAULT_THRESHOLD, step=0.01, label="Similarity Threshold")
        max_tries_input = gr.Slider(1, 8, value=5, step=1, label="Max Tries")
        max_events_input = gr.Slider(8, 35, value=14, step=1, label="Max Events")

    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            temperature_input = gr.Slider(0.0, 1.2, value=0.7, step=0.05, label="Generation Temperature")
            max_tokens_input = gr.Slider(200, 3000, value=1400, step=50, label="Max Completion Tokens")
        generator_model_input = gr.Textbox(label="Generator Model", value=DEFAULT_GENERATOR_MODEL)
        similarity_model_input = gr.Dropdown(
            label="Similarity Model",
            choices=SIMILARITY_MODEL_CHOICES + [CUSTOM_MODEL_OPTION],
            value=DEFAULT_SIMILARITY_MODEL,
            interactive=True,
        )
        similarity_model_custom_input = gr.Textbox(
            label="Custom Similarity Model (used only if 'custom' is selected)",
            placeholder="owner/repo-id",
        )

    run_btn = gr.Button("Generate", variant="primary")

    summary_output = gr.Markdown(label="Run Summary")
    best_narrative_output = gr.Textbox(label="Best Candidate Narrative", lines=8)
    attempts_output = gr.Dataframe(
        headers=["attempt", "score", "accepted"],
        datatype=["number", "number", "bool"],
        label="Attempts",
    )
    json_output = gr.JSON(label="Full Result JSON")
    logs_output = gr.Textbox(label="Execution Logs", lines=10)

    entity_input.change(
        on_entity_change_with_preview,
        inputs=[entity_input, max_events_input],
        outputs=[lang_input, source_preview_output],
    )
    lang_input.change(
        on_source_context_change,
        inputs=[entity_input, lang_input, max_events_input],
        outputs=[source_preview_output],
    )
    max_events_input.change(
        on_source_context_change,
        inputs=[entity_input, lang_input, max_events_input],
        outputs=[source_preview_output],
    )
    adaptation_preset_input.change(
        on_adaptation_preset_change,
        inputs=[adaptation_preset_input, adaptation_input],
        outputs=[adaptation_input],
    )
    similarity_model_input.change(
        on_similarity_model_change,
        inputs=[similarity_model_input],
        outputs=[threshold_input],
    )
    run_btn.click(
        fn=run_generation,
        inputs=[
            entity_input,
            lang_input,
            adaptation_input,
            threshold_input,
            max_tries_input,
            max_events_input,
            temperature_input,
            max_tokens_input,
            generator_model_input,
            similarity_model_input,
            similarity_model_custom_input,
        ],
        outputs=[
            summary_output,
            best_narrative_output,
            attempts_output,
            json_output,
            logs_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
