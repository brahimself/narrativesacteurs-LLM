import json
from pathlib import Path
from tqdm import tqdm
from typing import List
import nltk

# Téléchargement tokenizer phrase (une seule fois)
nltk.download("punkt")

CLEAN_DIR = Path("data/clean")
CHUNK_DIR = Path("data/chunks")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- PARAMÈTRES --------------------

MAX_TOKENS = 350
OVERLAP = 50

# Tokenisation simple (LLM-friendly)
def count_tokens(text: str) -> int:
    return len(text.split())

def split_sentences(text: str) -> List[str]:
    return nltk.sent_tokenize(text)

# -------------------- CHUNKING --------------------

def chunk_section(
    sentences: List[str],
    max_tokens: int,
    overlap: int
) -> List[str]:
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        tokens = count_tokens(sentence)

        if current_tokens + tokens > max_tokens:
            chunks.append(" ".join(current_chunk))

            # overlap
            overlap_tokens = 0
            overlap_chunk = []

            for s in reversed(current_chunk):
                t = count_tokens(s)
                if overlap_tokens + t > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_tokens += t

            current_chunk = overlap_chunk
            current_tokens = overlap_tokens

        current_chunk.append(sentence)
        current_tokens += tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# -------------------- PIPELINE COMPLET --------------------

def chunk_document(data: dict) -> List[dict]:
    chunks = []
    entity = data["entity"]
    lang = data["language"]

    for section, text in data["sections"].items():
        sentences = split_sentences(text)
        section_chunks = chunk_section(
            sentences,
            max_tokens=MAX_TOKENS,
            overlap=OVERLAP
        )

        for i, chunk in enumerate(section_chunks):
            chunks.append({
                "entity": entity,
                "language": lang,
                "section": section,
                "chunk_id": i,
                "text": chunk
            })

    return chunks

# -------------------- MAIN --------------------

def main():
    files = list(CLEAN_DIR.glob("*.json"))

    for path in tqdm(files, desc="Chunking cleaned data"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = chunk_document(data)

        if not chunks:
            continue

        out_path = CHUNK_DIR / path.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
