import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Example:
    sentence1: str
    sentence2: str
    label: int
    entity_a: str
    entity_b: str
    lang_a: str
    lang_b: str


def read_examples(path: Path) -> List[Example]:
    examples: List[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            s1 = str(row.get("sentence1", row.get("text_a", ""))).strip()
            s2 = str(row.get("sentence2", row.get("text_b", ""))).strip()
            if not s1 or not s2:
                continue

            if "score" in row:
                label = 1 if float(row["score"]) >= 0.5 else 0
            else:
                label = int(row.get("label", 0))

            examples.append(
                Example(
                    sentence1=s1,
                    sentence2=s2,
                    label=label,
                    entity_a=str(row.get("entity_a", "")).strip(),
                    entity_b=str(row.get("entity_b", "")).strip(),
                    lang_a=str(row.get("lang_a", "")).strip(),
                    lang_b=str(row.get("lang_b", "")).strip(),
                )
            )
    return examples


def auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            idx = np.where(inverse == i)[0]
            ranks[idx] = ranks[idx].mean()

    rank_sum_pos = ranks[pos].sum()
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u_stat / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y_sorted = labels[order]
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")

    tp = 0
    fp = 0
    prec_values: List[float] = []
    for y in y_sorted:
        if y == 1:
            tp += 1
            prec_values.append(tp / (tp + fp))
        else:
            fp += 1
    return float(sum(prec_values) / n_pos)


def prf1(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    tp = int(((labels == 1) & (preds == 1)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    thresholds = np.linspace(float(scores.min()), float(scores.max()), 401)
    best = {
        "f1": -1.0,
        "threshold": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }
    for t in thresholds:
        preds = (scores >= t).astype(int)
        m = prf1(labels, preds)
        if m["f1"] > best["f1"]:
            best = {
                "f1": m["f1"],
                "threshold": float(t),
                "precision": m["precision"],
                "recall": m["recall"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
    return best


def retrieval_en_fr(
    model: SentenceTransformer,
    examples: List[Example],
    batch_size: int,
) -> Optional[Dict[str, float]]:
    positives = [
        x for x in examples
        if x.label == 1
        and x.lang_a == "en"
        and x.lang_b == "fr"
        and x.entity_a
        and x.entity_a == x.entity_b
    ]
    if not positives:
        return None

    query_by_entity: Dict[str, str] = {}
    cand_by_entity: Dict[str, str] = {}
    for x in positives:
        if x.entity_a not in query_by_entity:
            query_by_entity[x.entity_a] = x.sentence1
        if x.entity_b not in cand_by_entity:
            cand_by_entity[x.entity_b] = x.sentence2

    entities = sorted(set(query_by_entity.keys()) & set(cand_by_entity.keys()))
    if len(entities) < 2:
        return None

    q_texts = [query_by_entity[e] for e in entities]
    c_texts = [cand_by_entity[e] for e in entities]
    q_emb = model.encode(
        q_texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    c_emb = model.encode(
        c_texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    sim = np.matmul(q_emb, c_emb.T)

    ranks: List[int] = []
    for i in range(len(entities)):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)

    ranks_arr = np.asarray(ranks)
    return {
        "entities": len(entities),
        "recall@1": float((ranks_arr <= 1).mean()),
        "recall@5": float((ranks_arr <= 5).mean()),
        "mrr": float((1.0 / ranks_arr).mean()),
    }


def evaluate_model(
    model_id: str,
    examples: List[Example],
    batch_size: int,
    max_seq_length: Optional[int],
) -> Dict:
    model = SentenceTransformer(model_id)
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    s1 = [x.sentence1 for x in examples]
    s2 = [x.sentence2 for x in examples]
    labels = np.asarray([x.label for x in examples], dtype=int)

    emb1 = model.encode(
        s1,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    emb2 = model.encode(
        s2,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    scores = np.sum(emb1 * emb2, axis=1)

    score_pos = scores[labels == 1]
    score_neg = scores[labels == 0]

    metrics_05 = prf1(labels, (scores >= 0.5).astype(int))
    best = best_f1_threshold(labels, scores)
    retrieval = retrieval_en_fr(model, examples, batch_size=batch_size)

    return {
        "model_id": model_id,
        "rows": len(examples),
        "positives": int((labels == 1).sum()),
        "negatives": int((labels == 0).sum()),
        "score_mean_pos": float(score_pos.mean()) if len(score_pos) else float("nan"),
        "score_mean_neg": float(score_neg.mean()) if len(score_neg) else float("nan"),
        "score_median_pos": float(np.median(score_pos)) if len(score_pos) else float("nan"),
        "score_median_neg": float(np.median(score_neg)) if len(score_neg) else float("nan"),
        "auc_roc": auc_roc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "f1_at_0.5": metrics_05["f1"],
        "precision_at_0.5": metrics_05["precision"],
        "recall_at_0.5": metrics_05["recall"],
        "best_f1": best["f1"],
        "best_threshold": best["threshold"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "retrieval_en_fr": retrieval,
    }


def print_report(report: Dict) -> None:
    print(f"\n=== {report['model_id']} ===")
    print(
        "rows={rows} positives={positives} negatives={negatives}".format(
            **report
        )
    )
    print(
        "mean_pos={:.4f} mean_neg={:.4f} median_pos={:.4f} median_neg={:.4f}".format(
            report["score_mean_pos"],
            report["score_mean_neg"],
            report["score_median_pos"],
            report["score_median_neg"],
        )
    )
    print(
        "auc={:.4f} ap={:.4f}".format(
            report["auc_roc"],
            report["average_precision"],
        )
    )
    print(
        "f1@0.5={:.4f} precision@0.5={:.4f} recall@0.5={:.4f}".format(
            report["f1_at_0.5"],
            report["precision_at_0.5"],
            report["recall_at_0.5"],
        )
    )
    print(
        "best_f1={:.4f} at_threshold={:.4f} (precision={:.4f}, recall={:.4f})".format(
            report["best_f1"],
            report["best_threshold"],
            report["best_precision"],
            report["best_recall"],
        )
    )

    retrieval = report.get("retrieval_en_fr")
    if retrieval:
        print(
            "retrieval en->fr: entities={entities} recall@1={recall@1:.4f} recall@5={recall@5:.4f} mrr={mrr:.4f}".format(
                **retrieval
            )
        )
    else:
        print("retrieval en->fr: not available (missing required positive pairs).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentence embedding similarity models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/datasets/autotrain_pair_score/test.jsonl"),
        help="JSONL dataset with sentence1/sentence2/score or text_a/text_b/label.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model ids to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data}")

    examples = read_examples(args.data)
    if not examples:
        raise SystemExit(f"No valid rows found in {args.data}")

    reports: List[Dict] = []
    for model_id in args.models:
        report = evaluate_model(
            model_id=model_id,
            examples=examples,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        )
        reports.append(report)
        print_report(report)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report: {args.out_json}")


if __name__ == "__main__":
    main()

