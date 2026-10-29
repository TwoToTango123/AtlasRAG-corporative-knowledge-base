"""Local NLI benchmark, no paid API. Threshold uses calibration groups only."""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from datetime import datetime, timezone
import json
import time
from scripts.run_routerai_eval import ROOT, save
from scripts.validation_fixtures import semantic_cases
from atlas_rag.semantic_support import classification_metrics

OUT = ROOT / "results/validation"
MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
REVISION = "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"


def predict(probabilities, threshold):
    if probabilities["supported"] >= threshold:
        return "supported"
    return "contradicted" if probabilities["contradicted"] > probabilities["insufficient"] else "insufficient"


def main():
    path = OUT / "nli.json"
    if path.exists():
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(8)
    cases = semantic_cases(ROOT)
    print("Loading NLI model", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    # The model card explicitly warns against FP16; the checkpoint defaults to it.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).eval()
    names = {"entailment": "supported", "neutral": "insufficient", "contradiction": "contradicted"}
    order = [names[str(model.config.id2label[i]).lower()] for i in range(3)]
    rows = []
    start = time.perf_counter()
    for case in cases:
        inputs = tokenizer(case["premise"], case["claim"], return_tensors="pt", truncation=False)
        tokens = inputs["input_ids"].shape[1]
        if tokens > 512:
            raise ValueError("Fixture exceeds NLI context; silent truncation forbidden")
        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits[0], dim=-1).tolist()
        rows.append({"id": case["id"], "split": case["split"], "gold": case["label"],
            "tokens": tokens, "probabilities": dict(zip(order, probs))})
        print(case["id"], flush=True)
    candidates = []
    for threshold in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
        metrics = classification_metrics([{**r, "pred": predict(r["probabilities"], threshold)} for r in rows if r["split"] == "calibration"])
        candidates.append({"threshold": threshold, **metrics})
    # No held-out labels influence this selection. 1.0 is an explicit reject-all fallback.
    clean = [r for r in candidates if r["false_supported"] == 0]
    selected = max(clean, key=lambda r: (r["support_recall"] or 0, -r["threshold"]))
    threshold = selected["threshold"]
    for row in rows:
        row["pred"] = predict(row["probabilities"], threshold)
        row["argmax_pred"] = max(row["probabilities"], key=row["probabilities"].get)
    save(path, {"date": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "revision": REVISION, "dtype": str(model.dtype), "label_order": order, "threshold": threshold,
        "threshold_rule": "Max support recall with zero false-supported on 20 calibration pairs; tie uses lowest threshold.",
        "calibration_candidates": candidates, "inference_seconds": time.perf_counter()-start,
        "heldout": classification_metrics([r for r in rows if r["split"] == "heldout"]),
        "heldout_argmax": classification_metrics([{**r, "pred": r["argmax_pred"]} for r in rows if r["split"] == "heldout"]),
        "rows": rows, "limits": ["Assistant labels, no independent human review.",
        "20 held-out pairs, four unseen source sections; softmax is not calibrated probability.",
        "Local diagnostic only; does not change production faithfulness."]})
    print("NLI complete", flush=True)


if __name__ == "__main__":
    main()
