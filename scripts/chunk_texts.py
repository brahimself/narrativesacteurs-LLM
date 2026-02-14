import re
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")
CHUNK_DIR = Path("data/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

TAIL_SECTION_MARKERS = {
    "== see also ==",
    "== external links ==",
    "== notes et références ==",
    "=== références ===",
    "=== liens externes ===",
}


def strip_non_biographical_tail(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cut_idx = None
    for i, line in enumerate(lines):
        marker = line.strip().lower()
        if marker in TAIL_SECTION_MARKERS:
            cut_idx = i
            break
    if cut_idx is not None:
        lines = lines[:cut_idx]

    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Portail "):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()

def split_paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # Dédup consécutif (évite les répétitions visibles)
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
    max_chars ~ limite simple (approx). 6000 chars ~ 1000-1500 tokens selon langue.
    overlap_paras: nb de paragraphes répétés entre chunks pour garder la continuité.
    """
    chunks = []
    current = []
    current_len = 0

    for p in paras:
        if current_len + len(p) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            # overlap: on garde les derniers paragraphes
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
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        text = strip_non_biographical_tail(text)
        if len(text) < 500:
            # Trop court → inutile
            continue

        paras = split_paragraphs(text)
        chunks = chunk_paragraphs(paras, max_chars=8000, overlap_paras=0)

        # Nom base: Angelina_Jolie.fr
        base = fp.stem  # ex: Angelina_Jolie.fr
        for i, ch in enumerate(chunks, start=1):
            out = CHUNK_DIR / f"{base}.chunk{i:03d}.txt"
            out.write_text(ch, encoding="utf-8")

if __name__ == "__main__":
    main()
