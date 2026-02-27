import random
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 1.5
MAX_BACKOFF_SECONDS = 60.0
REQUEST_PACING_SECONDS = 1.2


def slugify(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name


def _compute_backoff(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            retry_after_value = float(retry_after)
            if retry_after_value > 0:
                return min(MAX_BACKOFF_SECONDS, retry_after_value)
        except ValueError:
            pass

    exponential = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, 0.5)
    return min(MAX_BACKOFF_SECONDS, exponential + jitter)


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
        "User-Agent": "narrativesacteurs-llm/1.0 (wikipedia-fetch-script)",
        "Accept": "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 429:
                if attempt == MAX_RETRIES:
                    response.raise_for_status()
                wait_seconds = _compute_backoff(attempt, response.headers.get("Retry-After"))
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()

            pages = response.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            return page.get("extract", "") or ""
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(_compute_backoff(attempt))

    raise RuntimeError(f"Failed to fetch article after retries: {title} [{lang}]")


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
            err_path = RAW_DIR / f"{slugify(entity)}.{lang}.error.txt"
            if out.exists() and out.stat().st_size > 200:
                if err_path.exists():
                    err_path.unlink()
                continue

            try:
                text = fetch_wikipedia_extract(entity, lang=lang)
                save_text(entity, lang, text)
                if err_path.exists():
                    err_path.unlink()
            except Exception as e:
                err_path.write_text(str(e), encoding="utf-8")

            time.sleep(REQUEST_PACING_SECONDS)


if __name__ == "__main__":
    main()
