import re
import os
import time
from pathlib import Path
import requests
from tqdm import tqdm

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def slugify(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name

def fetch_wikipedia_extract(title: str, lang: str = "fr") -> str:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,     
        "redirects": 1,
        "titles": title,
    }
    headers = {
        "User-Agent": "narrativesacteurs",
        "Accept": "application/json",
    }

    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()

    pages = r.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    return page.get("extract", "") or ""


def save_text(entity: str, lang: str, text: str) -> Path:
    filename = f"{slugify(entity)}.{lang}.txt"
    path = RAW_DIR / filename
    path.write_text(text, encoding="utf-8")
    return path

def load_entities(path: str) -> list[str]:
    p = Path(path)
    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l and not l.startswith("#")]

def main():
    entities = load_entities("data/entities.txt")
    langs = ["fr", "en"]

    for entity in tqdm(entities, desc="Fetching Wikipedia"):
        for lang in langs:
            out = RAW_DIR / f"{slugify(entity)}.{lang}.txt"
            if out.exists() and out.stat().st_size > 200:
                continue

            try:
                text = fetch_wikipedia_extract(entity, lang=lang)
                save_text(entity, lang, text)
            except Exception as e:
                # On log et on continue
                err_path = RAW_DIR / f"{slugify(entity)}.{lang}.error.txt"
                err_path.write_text(str(e), encoding="utf-8")
            time.sleep(0.2)  # petite pause pour être poli avec l'API

if __name__ == "__main__":
    main()
