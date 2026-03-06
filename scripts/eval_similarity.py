import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    tn = int(((labels == 0) & (preds == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    unique_thresholds = np.unique(scores)
    if len(unique_thresholds) > 3000:
        thresholds = np.linspace(float(scores.min()), float(scores.max()), 1001)
    else:
        thresholds = unique_thresholds

    best = {
        "f1": -1.0,
        "threshold": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        metrics = prf1(labels, preds)
        if metrics["f1"] > best["f1"]:
            best = {
                "f1": metrics["f1"],
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
            }
    return best


def encode_similarity_scores(
    model: SentenceTransformer,
    examples: List[Example],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sentence1 = [x.sentence1 for x in examples]
    sentence2 = [x.sentence2 for x in examples]
    labels = np.asarray([x.label for x in examples], dtype=int)

    emb1 = model.encode(
        sentence1,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    emb2 = model.encode(
        sentence2,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    scores = np.sum(emb1 * emb2, axis=1)
    return labels, scores


def retrieval_crossling(
    model: SentenceTransformer,
    examples: List[Example],
    query_lang: str,
    candidate_lang: str,
    batch_size: int,
) -> Optional[Dict[str, float]]:
    positives = [
        x for x in examples
        if x.label == 1 and x.entity_a and x.entity_a == x.entity_b
    ]
    if not positives:
        return None

    query_by_entity: Dict[str, str] = {}
    cand_by_entity: Dict[str, str] = {}
    for x in positives:
        if x.lang_a == query_lang and x.lang_b == candidate_lang:
            query_by_entity.setdefault(x.entity_a, x.sentence1)
            cand_by_entity.setdefault(x.entity_b, x.sentence2)
        elif x.lang_a == candidate_lang and x.lang_b == query_lang:
            query_by_entity.setdefault(x.entity_b, x.sentence2)
            cand_by_entity.setdefault(x.entity_a, x.sentence1)

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
        "query_lang": query_lang,
        "candidate_lang": candidate_lang,
        "entities": len(entities),
        "recall@1": float((ranks_arr <= 1).mean()),
        "recall@5": float((ranks_arr <= 5).mean()),
        "mrr": float((1.0 / ranks_arr).mean()),
    }


def evaluate_from_scores(
    model_id: str,
    labels: np.ndarray,
    scores: np.ndarray,
    selected_threshold: float,
    retrieval_report: Optional[Dict[str, float]],
) -> Dict:
    score_pos = scores[labels == 1]
    score_neg = scores[labels == 0]

    metrics_05 = prf1(labels, (scores >= 0.5).astype(int))
    best = best_f1_threshold(labels, scores)
    selected_metrics = prf1(labels, (scores >= selected_threshold).astype(int))

    return {
        "model_id": model_id,
        "rows": int(len(labels)),
        "positives": int((labels == 1).sum()),
        "negatives": int((labels == 0).sum()),
        "score_mean_pos": float(score_pos.mean()) if len(score_pos) else float("nan"),
        "score_mean_neg": float(score_neg.mean()) if len(score_neg) else float("nan"),
        "score_median_pos": float(np.median(score_pos)) if len(score_pos) else float("nan"),
        "score_median_neg": float(np.median(score_neg)) if len(score_neg) else float("nan"),
        "auc_roc": auc_roc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "metrics_at_0_5": metrics_05,
        "metrics_at_selected_threshold": {
            "threshold": float(selected_threshold),
            **selected_metrics,
        },
        "best_f1_on_eval": best,
        "retrieval": retrieval_report,
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

    m05 = report["metrics_at_0_5"]
    print(
        "@0.5 -> acc={:.4f} f1={:.4f} precision={:.4f} recall={:.4f}".format(
            m05["accuracy"],
            m05["f1"],
            m05["precision"],
            m05["recall"],
        )
    )

    msel = report["metrics_at_selected_threshold"]
    print(
        "@selected(t={:.4f}) -> acc={:.4f} f1={:.4f} precision={:.4f} recall={:.4f}".format(
            msel["threshold"],
            msel["accuracy"],
            msel["f1"],
            msel["precision"],
            msel["recall"],
        )
    )

    best = report["best_f1_on_eval"]
    print(
        "best_on_eval -> t={:.4f} acc={:.4f} f1={:.4f} precision={:.4f} recall={:.4f}".format(
            best["threshold"],
            best["accuracy"],
            best["f1"],
            best["precision"],
            best["recall"],
        )
    )

    retrieval = report.get("retrieval")
    if retrieval:
        print(
            "retrieval {query_lang}->{candidate_lang}: entities={entities} recall@1={recall@1:.4f} recall@5={recall@5:.4f} mrr={mrr:.4f}".format(
                **retrieval
            )
        )
    else:
        print("retrieval: not available for requested language direction.")



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentence embedding similarity models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/datasets/autotrain_pair_score/test.jsonl"),
        help="Evaluation JSONL dataset with sentence1/sentence2/score or text_a/text_b/label.",
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=None,
        help="Optional validation JSONL. If provided, threshold is selected on validation best-F1 then applied to --data.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model ids to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--retrieval-query-lang", type=str, default="en")
    parser.add_argument("--retrieval-candidate-lang", type=str, default="fr_en")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data}")
    if args.validation_data is not None and not args.validation_data.exists():
        raise SystemExit(f"Validation dataset not found: {args.validation_data}")

    eval_examples = read_examples(args.data)
    if not eval_examples:
        raise SystemExit(f"No valid rows found in {args.data}")

    validation_examples: Optional[List[Example]] = None
    if args.validation_data is not None:
        validation_examples = read_examples(args.validation_data)
        if not validation_examples:
            raise SystemExit(f"No valid rows found in {args.validation_data}")

    reports: List[Dict] = []
    for model_id in args.models:
        model = SentenceTransformer(model_id)
        if args.max_seq_length is not None:
            model.max_seq_length = args.max_seq_length

        selected_threshold = 0.5
        validation_report = None
        if validation_examples is not None:
            val_labels, val_scores = encode_similarity_scores(
                model=model,
                examples=validation_examples,
                batch_size=args.batch_size,
            )
            val_best = best_f1_threshold(val_labels, val_scores)
            selected_threshold = float(val_best["threshold"])
            val_at_05 = prf1(val_labels, (val_scores >= 0.5).astype(int))
            val_at_sel = prf1(val_labels, (val_scores >= selected_threshold).astype(int))
            validation_report = {
                "rows": int(len(val_labels)),
                "best_f1": val_best,
                "metrics_at_0_5": val_at_05,
                "metrics_at_selected_threshold": {
                    "threshold": selected_threshold,
                    **val_at_sel,
                },
            }

        eval_labels, eval_scores = encode_similarity_scores(
            model=model,
            examples=eval_examples,
            batch_size=args.batch_size,
        )
        retrieval_report = retrieval_crossling(
            model=model,
            examples=eval_examples,
            query_lang=args.retrieval_query_lang,
            candidate_lang=args.retrieval_candidate_lang,
            batch_size=args.batch_size,
        )

        report = evaluate_from_scores(
            model_id=model_id,
            labels=eval_labels,
            scores=eval_scores,
            selected_threshold=selected_threshold,
            retrieval_report=retrieval_report,
        )
        report["selection"] = {
            "selected_threshold_source": "validation_best_f1" if validation_report else "fixed_0.5",
            "selected_threshold": selected_threshold,
        }
        if validation_report is not None:
            report["validation"] = validation_report

        reports.append(report)
        print_report(report)

        if validation_report is not None:
            print(
                "validation -> @0.5 acc={:.4f} f1={:.4f}, @selected(t={:.4f}) acc={:.4f} f1={:.4f}".format(
                    validation_report["metrics_at_0_5"]["accuracy"],
                    validation_report["metrics_at_0_5"]["f1"],
                    validation_report["metrics_at_selected_threshold"]["threshold"],
                    validation_report["metrics_at_selected_threshold"]["accuracy"],
                    validation_report["metrics_at_selected_threshold"]["f1"],
                )
            )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report: {args.out_json}")


if __name__ == "__main__":
    main()
