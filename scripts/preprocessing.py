import json
import re
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")

RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Sections biographiques pertinentes
ALLOWED_SECTIONS = {
    "en": [
        "Early life",
        "Career",
        "Personal life"
    ],
    "fr": [
        "Biographie",
        "Jeunesse",
        "Carrière",
        "Vie privée"
    ]
}

# -------------------- NETTOYAGE BAS NIVEAU --------------------

def remove_refs(text: str) -> str:
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/>]*/>", "", text)
    return text

def remove_templates(text: str) -> str:
    return re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)

def clean_links(text: str) -> str:
    # [[Brad Pitt|Pitt]] → Pitt
    return re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)

def remove_files_and_categories(text: str) -> str:
    text = re.sub(r"\[\[(File|Fichier|Image):[^\]]+\]\]", "", text)
    text = re.sub(r"\[\[Category:[^\]]+\]\]", "", text)
    return text

def normalize(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# -------------------- STRUCTURATION --------------------

def split_by_sections(wikitext: str) -> dict:
    sections = {}
    current_section = "intro"
    buffer = []

    for line in wikitext.splitlines():
        match = re.match(r"^==+\s*(.*?)\s*==+$", line)
        if match:
            sections[current_section] = "\n".join(buffer).strip()
            current_section = match.group(1)
            buffer = []
        else:
            buffer.append(line)

    sections[current_section] = "\n".join(buffer).strip()
    return sections

def filter_sections(sections: dict, lang: str) -> dict:
    allowed = ALLOWED_SECTIONS.get(lang, [])
    filtered = {}

    for title, content in sections.items():
        if len(content) < 200:
            continue
        if any(a.lower() in title.lower() for a in allowed):
            filtered[title] = content

    return filtered

# -------------------- PIPELINE COMPLET --------------------

def preprocess_wikitext(wikitext: str, lang: str) -> dict:
    text = remove_refs(wikitext)
    text = remove_templates(text)
    text = clean_links(text)
    text = remove_files_and_categories(text)

    sections = split_by_sections(text)
    sections = filter_sections(sections, lang)

    return {
        title: normalize(content)
        for title, content in sections.items()
    }

# -------------------- MAIN --------------------

def main():
    files = list(RAW_DIR.glob("*.json"))

    for path in tqdm(files, desc="Preprocessing Wikipedia data"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entity = data.get("entity")
        lang = data.get("language")
        wikitext = data.get("wikitext", "")

        if not wikitext or not lang:
            continue

        clean_sections = preprocess_wikitext(wikitext, lang)

        if not clean_sections:
            continue

        clean_data = {
            "entity": entity,
            "language": lang,
            "pageid": data.get("pageid"),
            "sections": clean_sections
        }

        out_path = CLEAN_DIR / path.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()