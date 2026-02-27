import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

import gradio as gr


ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "scripts" / "generate_analogues.py"
TIMELINE_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "data" / "results"
DEFAULT_SIMILARITY_MODEL = "sidbrahim/autotrain-NarraAnalogues15events"
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"


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
):
    if not SCRIPT_PATH.exists():
        raise gr.Error(f"Missing script: {SCRIPT_PATH}")

    if not entity_display:
        raise gr.Error("Select a source entity.")

    adaptation = (adaptation or "").strip()
    if not adaptation:
        raise gr.Error("Adaptation instruction is required.")

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
            str(a.get("candidate_name", "")),
            str(a.get("rationale", "")),
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

    adaptation_input = gr.Textbox(
        label="Adaptation Instruction",
        lines=4,
        placeholder="Ex: Keep a similar career trajectory but shift milestones toward European cinema and climate advocacy.",
    )

    with gr.Row():
        threshold_input = gr.Slider(0.1, 0.95, value=0.40, step=0.01, label="Similarity Threshold")
        max_tries_input = gr.Slider(1, 8, value=5, step=1, label="Max Tries")
        max_events_input = gr.Slider(8, 35, value=14, step=1, label="Max Events")

    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            temperature_input = gr.Slider(0.0, 1.2, value=0.7, step=0.05, label="Generation Temperature")
            max_tokens_input = gr.Slider(200, 3000, value=1400, step=50, label="Max Completion Tokens")
        generator_model_input = gr.Textbox(label="Generator Model", value=DEFAULT_GENERATOR_MODEL)
        similarity_model_input = gr.Textbox(label="Similarity Model", value=DEFAULT_SIMILARITY_MODEL)

    run_btn = gr.Button("Generate", variant="primary")

    summary_output = gr.Markdown(label="Run Summary")
    best_narrative_output = gr.Textbox(label="Best Candidate Narrative", lines=8)
    attempts_output = gr.Dataframe(
        headers=["attempt", "score", "accepted", "candidate_name", "rationale"],
        datatype=["number", "number", "bool", "str", "str"],
        label="Attempts",
    )
    json_output = gr.JSON(label="Full Result JSON")
    logs_output = gr.Textbox(label="Execution Logs", lines=10)

    entity_input.change(on_entity_change, inputs=[entity_input], outputs=[lang_input])
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
