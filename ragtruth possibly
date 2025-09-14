#!/usr/bin/env python3
"""
Evaluate a Self-RAG model on a Hugging Face dataset split (e.g., RAGTruth) via a TGI endpoint.

Requirements:
  pip install datasets requests tqdm numpy

Example usage:
  python evals/ragtruth_hf_eval.py \
    --tgi-url http://localhost:8300 \
    --dataset-name RAGTruth/ragtruth \
    --split validation \
    --output-file outputs/ragtruth_preds.jsonl

If your columns differ, specify:
  --input-column prompt --context-column context --label-column label
or let the script auto-detect from common names:
  - prompt: prompt | question | query | input | instruction
  - context: context | contexts | passages | docs | evidence | source_documents
  - label: label | hallucinated | hallucination | is_hallucination | gold_label

Label mapping:
  - Any of yes,true,1 are treated as positive ('yes')
  - Any of no,false,0 are treated as negative ('no')
  Override with --positive-label-values and --negative-label-values if needed.
"""

import argparse, json, os, re, time
from typing import Any, Dict, List, Optional
import requests
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgi-url", type=str, default="http://localhost:8300", help="Base URL of TGI server")
    ap.add_argument("--dataset-name", type=str, required=True, help="HF dataset name or path (e.g., RAGTruth/ragtruth)")
    ap.add_argument("--dataset-config", type=str, default=None, help="HF dataset config name (optional)")
    ap.add_argument("--split", type=str, default="validation", help="Split name (e.g., validation, test)")
    ap.add_argument("--output-file", type=str, default="outputs/ragtruth_preds.jsonl")
    ap.add_argument("--system-prompt", type=str, default="", help="Optional system prompt prefix")

    # Column hints; if omitted, we try to auto-detect.
    ap.add_argument("--input-column", type=str, default=None, help="Column containing the main user question/prompt")
    ap.add_argument("--context-column", type=str, default=None, help="Column containing retrieval/context (str or list[str])")
    ap.add_argument("--label-column", type=str, default=None, help="Column containing ground-truth yes/no or 0/1")

    # Label mapping controls
    ap.add_argument("--positive-label-values", type=str, default="yes,true,1", help="Comma-separated values considered positive")
    ap.add_argument("--negative-label-values", type=str, default="no,false,0", help="Comma-separated values considered negative")

    # Generation params
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    return ap.parse_args()

def normalize_label(x: Any, pos_vals: set, neg_vals: set) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in pos_vals:
        return "yes"
    if s in neg_vals:
        return "no"
    # Try to coerce ints/bools
    if s.isdigit():
        return "yes" if int(s) != 0 else "no"
    if s in {"true"}:
        return "yes"
    if s in {"false"}:
        return "no"
    return None

def extract_yesno(text: str) -> Optional[str]:
    if not text:
        return None
    m = YESNO_RE.search(text)
    return m.group(1).lower() if m else None

def auto_pick(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def build_prompt(question: str, context: Optional[Any], system_prompt: str) -> str:
    ctx = ""
    if context is not None:
        if isinstance(context, list):
            ctx = "\n\n".join(map(str, context))
        else:
            ctx = str(context)
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    parts.append(f"Question:\n{question}")
    if ctx:
        parts.append(f"Context:\n{ctx}")
    parts.append("Answer strictly 'yes' or 'no' to whether the response contains unsupported claims.")
    return "\n\n".join(parts)

def tgi_generate(tgi_url: str, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "stop": [],
            "do_sample": temperature > 0.0
        }
    }
    for _ in range(3):
        r = requests.post(f"{tgi_url}/generate", json=payload, timeout=120)
        if r.status_code == 200:
            data = r.json()
            return data.get("generated_text", "")
        time.sleep(1.0)
    raise RuntimeError(f"TGI generate failed after retries (url={tgi_url}).")

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    # Load HF dataset split
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)

    cols = list(ds.features.keys())
    input_col = args.input_column or auto_pick(cols, ["prompt", "question", "query", "input", "instruction"])
    context_col = args.context_column or auto_pick(cols, ["context", "contexts", "passages", "docs", "evidence", "source_documents"])
    label_col = args.label_column or auto_pick(cols, ["label", "hallucinated", "hallucination", "is_hallucination", "gold_label", "target"])

    if not input_col:
        raise ValueError(f"Could not auto-detect an input column. Available columns: {cols}. Provide --input-column.")
    # label_col is optional; if not found we'll just write predictions

    pos_vals = {s.strip().lower() for s in args.positive_label_values.split(",") if s.strip()}
    neg_vals = {s.strip().lower() for s in args.negative_label_values.split(",") if s.strip()}

    preds_bin: List[int] = []
    gts_bin: List[int] = []

    # Keep records in-memory so we can compute BERTScore afterwards (if available)
    all_recs = []

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for ex in tqdm(ds, desc="Evaluating", unit="ex"):
            question = ex.get(input_col)
            context = ex.get(context_col) if context_col else None
            label_raw = ex.get(label_col) if label_col else None
            gt = normalize_label(label_raw, pos_vals, neg_vals)

            prompt = build_prompt(str(question), context, args.system_prompt)
            gen = tgi_generate(args.tgi_url, prompt, args.temperature, args.top_p, args.max_new_tokens)
            pred = extract_yesno(gen) or extract_yesno(gen.splitlines()[0] if gen else "") or "no"

            rec = {
                "prompt": question,
                "context": context,
                "prediction": pred,
                "label": gt,
                "raw_output": gen
            }
            # write to file (original behavior)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # store in-memory for BERT scoring later
            all_recs.append(rec)

            if gt is not None:
                preds_bin.append(1 if pred == "yes" else 0)
                gts_bin.append(1 if gt == "yes" else 0)

    # Original metrics calculation (unchanged behavior)
    if gts_bin:
        y_true = np.array(gts_bin)
        y_pred = np.array(preds_bin)
        acc = (y_true == y_pred).mean().item()
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics = {
            "accuracy": round(float(acc), 6),
            "precision_yes": round(float(prec), 6),
            "recall_yes": round(float(rec), 6),
            "f1_yes": round(float(f1), 6),
            "support": int(len(y_true))
        }
        print("\nMetrics:", json.dumps(metrics, indent=2))
        metrics_path = os.path.splitext(args.output_file)[0] + ".metrics.json"

        # --- BERTScore addition: compute BERT between raw_output and the normalized label when label exists ---
        try:
            # Import BERT scorer (deferred import so script still works if bert_score is absent)
            try:
                from bert_score.scorer import BERTScorer
                berter = "scorer"
            except Exception:
                # fallback to score function
                from bert_score import score as bert_score_func
                berter = "func"
        except Exception as e:
            print("Warning: bert_score not available; skipping BERT scoring. Install with `pip install bert-score` to enable. Error:", e)
            # write original metrics and exit
            with open(metrics_path, "w", encoding="utf-8") as mf:
                json.dump(metrics, mf, indent=2)
            return

        # prepare pairs where label exists
        cand_texts = []
        ref_texts = []
        for r in all_recs:
            if r.get("label") is None:
                continue
            # Use raw_output as candidate and the normalized label text as reference
            cand_texts.append(r.get("raw_output", ""))
            ref_texts.append(r.get("label", ""))

        if cand_texts and ref_texts:
            try:
                if berter == "scorer":
                    # Force roberta-large
                    device = "cuda" if (hasattr(torch, 'cuda') and torch.cuda.is_available()) else "cpu"
                    scorer = BERTScorer(model_type="roberta-large", lang="en", idf=False, batch_size=64, use_fast_tokenizer=True, device=device)
                    P, R, F = scorer.score(cand_texts, ref_texts, verbose=False, batch_size=64)
                else:
                    # functional API fallback: try to request roberta-large; if it doesn't accept model_type, call without it
                    try:
                        P, R, F = bert_score_func(cand_texts, ref_texts, model_type="roberta-large", verbose=False, batch_size=64)
                    except TypeError:
                        P, R, F = bert_score_func(cand_texts, ref_texts, lang="en", verbose=False, batch_size=64)
                # convert to floats; BERTScore outputs are in [0,1]
                bert_p_mean = float(P.mean().item())
                bert_r_mean = float(R.mean().item())
                bert_f_mean = float(F.mean().item())
                # Add to metrics (report as standard fraction 0..1)
                metrics["bert_precision"] = round(bert_p_mean, 6)
                metrics["bert_recall"] = round(bert_r_mean, 6)
                metrics["bert_f1"] = round(bert_f_mean, 6)
                print("BERTScore (mean) P/R/F:", metrics["bert_precision"], metrics["bert_recall"], metrics["bert_f1"])
            except Exception as e:
                print("Warning: BERT scoring failed:", e)
        else:
            print("No labeled examples with raw_output found — skipping BERT scoring for RAGTruth.")

        # write metrics (original behavior + bert additions)
        with open(metrics_path, "w", encoding="utf-8") as mf:
            json.dump(metrics, mf, indent=2)
    else:
        print("\nNo ground-truth labels found. Wrote predictions only to:", args.output_file)

if __name__ == "__main__":
    main()
