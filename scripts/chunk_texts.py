import re
import unicodedata
from pathlib import Path

from tqdm import tqdm

RAW_DIR = Path("data/raw")
CHUNK_DIR = Path("data/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

HEADING_RE = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$")

DROP_SECTION_KEYWORDS = {
    "see also",
    "external links",
    "notes",
    "references",
    "bibliography",
    "appendices",
    "voir aussi",
    "liens externes",
    "notes et references",
    "references et notes",
    "annexes",
    "bibliographie",
    # Filmography/media lists
    "filmography",
    "selected filmography",
    "television",
    "theatre",
    "theater",
    "radio",
    "discography",
    "filmographie",
    "television et radio",
    # Voice/dubbing sections
    "voice work",
    "dubbing",
    "voix francaises",
    "voix francophones",
    "doublage",
    # Awards/honors lists
    "awards",
    "accolades",
    "honours",
    "honors",
    "nominations",
    "distinctions",
    "recompenses",
}


def normalize_heading(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    return value


def should_drop_section(heading: str) -> bool:
    if not heading:
        return False
    return any(keyword in heading for keyword in DROP_SECTION_KEYWORDS)


def strip_non_biographical_tail(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned = []
    skip_level = None

    for line in lines:
        stripped = line.strip()
        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            heading_level = len(heading_match.group(1))
            heading_text = normalize_heading(heading_match.group(2))

            # End skipped block when we return to equal/higher heading level.
            if skip_level is not None and heading_level <= skip_level:
                skip_level = None

            if skip_level is None and should_drop_section(heading_text):
                skip_level = heading_level
                continue

        if skip_level is not None:
            continue

        if stripped.lower().startswith("portail "):
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def split_paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # Consecutive paragraph dedup to avoid visible repetitions.
    deduped = []
    prev = None
    for p in paras:
        if p == prev:
            continue
        deduped.append(p)
        prev = p

    return deduped


def chunk_paragraphs(paras: list[str], max_chars: int = 6000, overlap_paras: int = 1) -> list[str]:
    """
    max_chars ~ simple approximate limit.
    overlap_paras: repeated paragraphs between chunks for continuity.
    """
    chunks = []
    current = []
    current_len = 0

    for p in paras:
        if current_len + len(p) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paras:] if overlap_paras > 0 else []
            current_len = sum(len(x) for x in current) + 2 * max(0, len(current) - 1)

        current.append(p)
        current_len += len(p) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def main():
    files = sorted(RAW_DIR.glob("*.txt"))
    for fp in tqdm(files, desc="Chunking"):
        base = fp.stem  # Example: Angelina_Jolie.fr

        # Remove stale chunks for this base before re-writing.
        for old in CHUNK_DIR.glob(f"{base}.chunk*.txt"):
            old.unlink()

        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        text = strip_non_biographical_tail(text)
        if len(text) < 500:
            continue

        paras = split_paragraphs(text)
        chunks = chunk_paragraphs(paras, max_chars=8000, overlap_paras=0)

        for i, ch in enumerate(chunks, start=1):
            out = CHUNK_DIR / f"{base}.chunk{i:03d}.txt"
            out.write_text(ch, encoding="utf-8")


if __name__ == "__main__":
    main()
