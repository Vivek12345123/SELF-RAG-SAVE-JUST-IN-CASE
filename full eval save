#!/usr/bin/env python3
"""
run_all_evals.py

Unified runner that:
 - Loads Self-RAG 7B via vLLM
 - Starts a tiny TGI-compatible adapter for RAGTruth
 - Generates predictions for several datasets and calls the official
   evaluation scripts (unchanged) as subprocesses (invoked with same
   Python interpreter used to run this script).
 - Aggregates all metrics to outputs/all_eval_scores.json.

Usage example (512 tokens, 200 samples, batch size 4):
  python run_all_evals.py --max-tokens 512 --max-examples 200 --batch-size 4
"""
import argparse
import json
import os
import time
import shutil
import subprocess
import re
import sys
import importlib
import tempfile
from pathlib import Path
from typing import List, Optional
import threading
import logging

# --- Configurable globals ---
MAX_SAMPLES = 200
KEEP_INTERMEDIATES = False
# Add this to your run_all_evals.py near the top with other globals


# UPDATED: Use Self-RAG 7B model
MODEL_NAME = "selfrag/selfrag_llama2_7b"
MODEL_CACHE_DIR = "/gscratch/h2lab/akari/model_cache"
DEFAULT_TGI_HOST = "127.0.0.1"
DEFAULT_TGI_PORT = 8300
DATASET_CACHE_ROOT = Path("./datasets_cache")
OUT_ROOT = Path("./outputs")
MIN_FREE_BYTES_WARN = 30 * 1024 ** 3

EVALS_DIR = Path("evals")

# Possible filenames for each eval (we try variants to be robust to typos)
EVAL_SCRIPT_CANDIDATES = {
    "ragtruth": ["ragtruth_hf_eval.py"],
    "squad": ["squad_v2_official_eval.py"],
    "hotpot": ["hotpot_eval.py"],
    "msmarco": ["ms_marco_eval.py"],
    "nq": ["natural_questions_official_eval.py", "natural_questions_offical_eval.py"],
    "trivia": ["triviaqa_eval.py"],
}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Global tokenizer reference used by truncater
GLOBAL_TOKENIZER = None
# Conservative model max tokens (adjust if your model supports a different context)
DEFAULT_MODEL_MAX_TOKENS = 4096

# --- Helper functions ---
def get_task_max_samples(task_name: str, args) -> int:
    """
    Get max samples for a specific task, using args.max_examples if provided,
    otherwise DEFAULT MAX_SAMPLES
    """
    return args.max_examples if args.max_examples is not None else MAX_SAMPLES

def get_task_max_tokens(task_name: str, args) -> int:
    """
    Get max tokens for a specific task, using args.max_tokens if provided,
    otherwise default to 100
    """
    return args.max_tokens if args.max_tokens is not None else 100

def get_task_batch_size(task_name: str, args) -> int:
    """
    Get batch size for a specific task, using args.batch_size if provided,
    otherwise default to 8
    """
    return args.batch_size if args.batch_size is not None else 8

# --- Helpers -----------------------------------------------------------------
def bytes_to_gb(b): return b / (1024 ** 3)
def check_disk_space(path="."):
    usage = shutil.disk_usage(path)
    free = usage.free
    logging.info(f"[disk] total={bytes_to_gb(usage.total):.1f}GB used={bytes_to_gb(usage.used):.1f}GB free={bytes_to_gb(free):.1f}GB")
    if free < MIN_FREE_BYTES_WARN:
        logging.warning(f"Free disk < {bytes_to_gb(MIN_FREE_BYTES_WARN):.1f}GB. Full downloads may fail. Use --max-examples for testing.")

def resolve_script(task_key: str) -> Path:
    """
    Return first existing script Path for a task. If none exist, return
    the path for the first candidate (so errors are obvious).
    """
    candidates = EVAL_SCRIPT_CANDIDATES.get(task_key, [])
    for name in candidates:
        p = EVALS_DIR / name
        if p.exists():
            return p
    # fallback: return first candidate path (may not exist)
    if candidates:
        return EVALS_DIR / candidates[0]
    raise ValueError(f"No script candidates configured for task {task_key}")

def limit_gold_file(original_gold_path: str, max_samples: Optional[int] = None) -> str:
    """
    Create a temp limited gold file (JSON or JSONL) with up to max_samples entries.
    Returns path to the temporary file.
    """
    max_samples = int(max_samples or MAX_SAMPLES)
    p = Path(original_gold_path)
    if not p.exists():
        raise FileNotFoundError(f"Gold file not found: {original_gold_path}")
    # Try JSON
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            data["data"] = data["data"][:max_samples]
        elif isinstance(data, list):
            data = data[:max_samples]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(tmp.name, "w", encoding="utf-8") as oh:
            json.dump(data, oh)
        return tmp.name
    except Exception:
        # fallback to jsonl style
        lines = []
        with p.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    obj = line
                lines.append(obj)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        with open(tmp.name, "w", encoding="utf-8") as oh:
            for item in lines:
                # if it's already a raw str (not JSON), write as that line
                if isinstance(item, str):
                    oh.write(item + "\n")
                else:
                    oh.write(json.dumps(item, ensure_ascii=False) + "\n")
        return tmp.name

def run_and_capture_metrics(cmd: List[str], out_dir: Optional[Path] = None, fallback_names: Optional[List[Path]] = None, metrics_basename: str = "metrics.json", task_name: Optional[str] = None) -> Optional[str]:
    """
    Run subprocess using sys.executable so the same Python is used.
    Saves stdout/stderr to out_dir/<task>_stdout.txt and *_stderr.txt.
    Attempts to extract a JSON object from stdout, otherwise looks for fallback files or any *metrics*.json.
    Returns path to metrics JSON or None.
    """
    out_dir = Path(out_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_task = (task_name or "command").replace("/", "_").replace(" ", "_")
    stdout_path = out_dir / f"{safe_task}_stdout.txt"
    stderr_path = out_dir / f"{safe_task}_stderr.txt"

    if not cmd:
        logging.error("[run_and_capture] Empty command")
        return None
    # Ensure script invocation via same Python interpreter
    if cmd[0] != sys.executable:
        cmd = [sys.executable] + cmd

    logging.info(f"[run_and_capture] Running: {' '.join(cmd)} (cwd={Path('.').resolve()})")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(".").resolve()))
    except Exception as e:
        logging.error(f"[run_and_capture] Failed to start subprocess {cmd}: {e}")
        return None

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    try:
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
    except Exception as e:
        logging.debug(f"[run_and_capture] Could not write stdout/stderr: {e}")

    # Try to parse first JSON object from stdout
    m = re.search(r"\{.*?\}", stdout, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            metrics_path = out_dir / metrics_basename
            with open(metrics_path, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, indent=2, ensure_ascii=False)
            logging.info(f"[run_and_capture] Parsed metrics from stdout -> {metrics_path}")
            return str(metrics_path)
        except Exception as e:
            logging.debug(f"[run_and_capture] JSON parse from stdout failed: {e}")

    # Check fallback names
    if fallback_names:
        for p in fallback_names:
            try:
                cand = Path(p)
                if not cand.is_absolute():
                    cand = out_dir / cand.name
                if cand.exists():
                    logging.info(f"[run_and_capture] Found fallback metrics: {cand}")
                    return str(cand)
            except Exception:
                continue

    # Scan out_dir for metrics-looking files
    for cand in sorted(out_dir.glob("*metrics*.json")) + sorted(out_dir.glob("eval*.json")) + sorted(out_dir.glob("*.json")):
        if cand.name in (stdout_path.name, stderr_path.name):
            continue
        logging.info(f"[run_and_capture] Found metrics candidate in out_dir: {cand}")
        return str(cand)

    logging.warning(f"[run_and_capture] No metrics file found (exit={proc.returncode}). See {stdout_path} and {stderr_path}")
    return None

# --- Prompt truncation helper --------------------------------------
def truncate_prompt(prompt: str, max_input_tokens: int, tokenizer=None) -> str:
    """
    Truncate prompt so that tokenized length is <= max_input_tokens.
    Uses GLOBAL_TOKENIZER if tokenizer is None. If no tokenizer is available,
    falls back to an approximate character-based truncation (assumes ~4 chars/token).
    Keeps the *end* of the prompt (so instruction/last context is preserved).
    """
    global GLOBAL_TOKENIZER
    tok = tokenizer or GLOBAL_TOKENIZER
    if max_input_tokens is None or max_input_tokens <= 0:
        return prompt
    try:
        if tok is None:
            # approximate with characters
            max_chars = int(max_input_tokens * 4)
            if len(prompt) <= max_chars:
                return prompt
            return prompt[-max_chars:]
        # use tokenizer to truncate tokens
        ids = tok.encode(prompt, add_special_tokens=False)
        if len(ids) <= max_input_tokens:
            return prompt
        truncated = ids[-max_input_tokens:]
        # decode truncated tokens back to string
        # tokenizer decode sometimes requires skip_special_tokens argument
        try:
            return tok.decode(truncated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception:
            return tok.decode(truncated, skip_special_tokens=True)
    except Exception:
        # last-resort fallback
        max_chars = int(max_input_tokens * 4)
        if len(prompt) <= max_chars:
            return prompt
        return prompt[-max_chars:]

# --- TGI adapter ------------------------------------------------------------
def start_tgi_adapter_in_thread(tgi_host: str, tgi_port: int, model_obj, sampling_params):
    try:
        from flask import Flask, request, jsonify
    except Exception as e:
        raise RuntimeError("Flask required for TGI adapter") from e

    app = Flask("tgi_adapter_for_ragtruth")

    @app.route("/generate", methods=["POST"])
    def generate_route():
        payload = request.get_json(force=True)
        prompt = payload.get("inputs", "")
        params = payload.get("parameters", {})
        try:
            from vllm import SamplingParams
            temp = params.get("temperature", sampling_params.temperature)
            top_p = params.get("top_p", sampling_params.top_p)
            max_tokens = params.get("max_new_tokens", getattr(sampling_params, "max_tokens", 128))
            sp = SamplingParams(temperature=float(temp), top_p=float(top_p), max_tokens=int(max_tokens), skip_special_tokens=False)
        except Exception:
            sp = sampling_params
        try:
            preds = model_obj.generate([prompt], sp)
            outtext = preds[0].outputs[0].text
            return jsonify({"generated_text": outtext})
        except Exception as e:
            return jsonify({"error": str(e), "generated_text": ""}), 500

    def run_app():
        logging.info(f"[tgi_adapter] Starting adapter at http://{tgi_host}:{tgi_port}")
        app.run(host=tgi_host, port=tgi_port, threaded=True)

    thr = threading.Thread(target=run_app, daemon=True)
    thr.start()
    time.sleep(1.5)
    return thr

# --- Model loader -----------------------------------------------------------
def load_model_and_tokenizer(model_name: str = MODEL_NAME, cache_dir: str = MODEL_CACHE_DIR):
    logging.info(f"[model] Loading model {model_name} (vLLM, dtype=half)")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as e:
        logging.warning(f"[tokenizer] AutoTokenizer load failed: {e}")
        tokenizer = None
    try:
        from vllm import LLM, SamplingParams
        model = LLM(model_name, download_dir=cache_dir, dtype="half")
        # UPDATED: Use Self-RAG specific sampling parameters
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)
    except Exception as e:
        logging.error(f"[model] vLLM LLM load failed: {e}")
        raise
    
    # UPDATED: Use Self-RAG specific prompt format
    def format_prompt(input_text: str, paragraph: Optional[str] = None):
        prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input_text)
        if paragraph is not None:
            prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
        return prompt
    
    return model, tokenizer, sampling_params, format_prompt

# --- Self-RAG Output Cleaning Helper ---
def clean_selfrag_output(text: str) -> str:
    """
    Clean Self-RAG output by removing special tokens and extracting the core answer.
    Self-RAG outputs contain special tokens like [Retrieval], [Relevant], [Fully supported], etc.
    """
    if not text:
        return ""
    
    # Remove Self-RAG special tokens
    import re
    
    # Common Self-RAG special tokens to remove
    selfrag_tokens = [
        r'\[Retrieval\]',
        r'\[No Retrieval\]',
        r'\[Relevant\]',
        r'\[Irrelevant\]', 
        r'\[Partially supported\]',
        r'\[Fully supported\]',
        r'\[No support / Contradictory\]',
        r'\[Utility:\d+\]',
        r'<paragraph>.*?</paragraph>',
        r'</s>',
        r'<s>',
    ]
    
    cleaned = text
    for token_pattern in selfrag_tokens:
        cleaned = re.sub(token_pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

# --- Metrics extraction helper ---
def extract_metrics_from_text(text: str) -> dict:
    """Extract metrics from plain text output (last resort)."""
    metrics = {}
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for common metric patterns
        patterns = [
            r'([a-zA-Z][a-zA-Z0-9_\-]*)\s*[:\=]\s*([0-9]+\.?[0-9]*%?)',
            r'([a-zA-Z][a-zA-Z0-9_\-]*)\s*[:\=]\s*([0-9]+\.?[0-9]*)',
            r'([a-zA-Z][a-zA-Z0-9_\-]*)\s+([0-9]+\.?[0-9]*%?)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            for key, value in matches:
                key = key.lower().replace('-', '_')
                try:
                    # Convert percentage to decimal
                    if value.endswith('%'):
                        metrics[key] = float(value[:-1])
                    else:
                        metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
                    
    return metrics

# --- Task runners -----------------------------------------------------------

def run_squad_v2(model, sampling_params, format_prompt, dataset_key="rajpurkar/squad_v2", split="validation", out_dir=OUT_ROOT / "squad_v2", batch_size=8, max_tokens=128, temp=0.0, top_p=1.0, keep_intermediates: bool = False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    logging.info("[squad] Loading dataset %s split=%s", dataset_key, split)
    ds = load_dataset(dataset_key, split=split)
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    articles = {}
    qlist = []
    for ex in ds:
        qid = ex.get("id") or ex.get("question_id") or ex.get("qid") or str(len(qlist))
        question = ex.get("question") or ex.get("queries") or ex.get("query") or ""
        context = ex.get("context") or ex.get("paragraphs") or ""
        answers = ex.get("answers") or {"text": [], "answer_start": []}
        key = context
        if key not in articles:
            articles[key] = {"title": ex.get("title") or "", "paragraphs": [{"context": context, "qas": []}]}
        articles[key]["paragraphs"][0]["qas"].append({
            "id": qid, "question": question,
            "answers": [{"text": t, "answer_start": s} for t, s in zip(answers.get("text", []), answers.get("answer_start", []))]
        })
        qlist.append((qid, question, context))
    data_json = {"version": "v2.0", "data": [v for v in articles.values()]}
    data_file = out_dir / "squad_v2_data.json"
    with open(data_file, "w", encoding="utf-8") as fh:
        json.dump(data_json, fh)
    pred_file = out_dir / "preds.json"
    from vllm import SamplingParams as VSamplingParams
    sp = VSamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=False)
    preds = {}
    logging.info(f"[squad] Generating answers for {len(qlist)} examples (batch {batch_size})")
    for i in range(0, len(qlist), batch_size):
        batch = qlist[i:i+batch_size]
        max_input_tokens = DEFAULT_MODEL_MAX_TOKENS - max_tokens - 16
        prompts = [truncate_prompt(format_prompt(f"Answer the question using the context. Question: {q}\nContext: {c}"), max_input_tokens) for (_id,q,c) in batch]
        outs = model.generate(prompts, sp)
        for j, out in enumerate(outs):
            text = (out.outputs[0].text or "").strip()
            text = clean_selfrag_output(text)
            qid = batch[j][0]
            preds[qid] = text
    with open(pred_file, "w", encoding="utf-8") as fh:
        json.dump(preds, fh)
    logging.info("[squad] Wrote preds -> %s", pred_file)

    script = resolve_script("squad")
    limited_gold_file = limit_gold_file(str(data_file))
    metrics_path = out_dir / "eval.json"
    cmd = [str(script), str(limited_gold_file), str(pred_file), "--out-file", str(metrics_path)]
    metrics_found = run_and_capture_metrics(cmd, out_dir=out_dir, fallback_names=[metrics_path], metrics_basename=metrics_path.name, task_name="squad_v2")

    if not keep_intermediates:
        for p in (pred_file, data_file, limited_gold_file):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    else:
        logging.info("[squad] Kept intermediates: %s", out_dir)

    return metrics_found

def run_hotpot(model, sampling_params, format_prompt, dataset_key="hotpotqa/hotpot_qa", split="validation", out_dir=OUT_ROOT / "hotpot", batch_size=8, max_tokens=128, temp=0.0, top_p=1.0, keep_intermediates: bool = False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    logging.info("[hotpot] Loading dataset %s split=%s", dataset_key, split)
    ds = load_dataset(dataset_key, "distractor", split=split)
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    preds = {"answer": {}, "sp": {}}
    from vllm import SamplingParams as VSamplingParams
    sp = VSamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=False)
    qlist = []
    for ex in ds:
        qid = ex.get("_id") or ex.get("id") or str(len(qlist))
        question = ex.get("question") or ""
        context = ""
        if "context" in ex and ex["context"]:
            parts = []
            for p in ex["context"]:
                title = p[0] if len(p) > 0 else ""
                sents = " ".join(p[1]) if len(p) > 1 else ""
                parts.append(f"{title}: {sents}")
            context = "\n".join(parts)
        qlist.append((qid, question, context))
    logging.info(f"[hotpot] Generating answers for {len(qlist)} items (batch {batch_size})")
    for i in range(0, len(qlist), batch_size):
        batch = qlist[i:i+batch_size]
        max_input_tokens = DEFAULT_MODEL_MAX_TOKENS - max_tokens - 16
        prompts = [truncate_prompt(format_prompt(f"Answer the question using the context. Question: {q}\nContext: {c}"), max_input_tokens) for (_id,q,c) in batch]
        outs = model.generate(prompts, sp)
        for j, out in enumerate(outs):
            ans = (out.outputs[0].text or "").strip()
            ans = clean_selfrag_output(ans)
            qid = batch[j][0]
            preds["answer"][qid] = ans
            preds["sp"][qid] = []
    pred_file = out_dir / "preds_hotpot.json"
    gold_file = out_dir / "gold_hotpot.json"
    with open(pred_file, "w", encoding="utf-8") as fh:
        json.dump(preds, fh)
    with open(gold_file, "w", encoding="utf-8") as fh:
        json.dump([ex for ex in ds], fh)
    logging.info("[hotpot] Wrote preds and gold files")

    script = resolve_script("hotpot")
    limited_gold = limit_gold_file(str(gold_file))
    metrics_path = out_dir / "hotpot_metrics.json"
    script_text = ""
    try:
        script_text = open(script, "r", encoding="utf-8").read()
    except Exception:
        script_text = ""
    if "--out-file" in script_text or "--out_file" in script_text:
        cmd = [str(script), str(pred_file), str(limited_gold), "--out-file", str(metrics_path)]
    else:
        cmd = [str(script), str(pred_file), str(limited_gold)]
    metrics_found = run_and_capture_metrics(cmd, out_dir=out_dir, fallback_names=[metrics_path], metrics_basename=metrics_path.name, task_name="hotpot")

    if not keep_intermediates:
        for p in (pred_file, gold_file, limited_gold):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    return metrics_found

def run_ragtruth_adapter_and_eval(tgi_host: str, tgi_port: int, dataset_name: str, dataset_config: Optional[str], split: str, output_file: str, system_prompt: str, model_obj, sampling_params, keep_intermediates: bool = False):
    out_dir = Path(output_file).parent
    thr = start_tgi_adapter_in_thread(tgi_host, tgi_port, model_obj, sampling_params)
    script = resolve_script("ragtruth")
    cmd = [str(script), "--tgi-url", f"http://{tgi_host}:{tgi_port}", "--dataset-name", dataset_name, "--split", split, "--output-file", output_file]
    if dataset_config:
        cmd += ["--dataset-config", dataset_config]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]
    metrics_found = run_and_capture_metrics(cmd, out_dir=out_dir, fallback_names=[out_dir/"ragtruth_metrics.json"], metrics_basename="ragtruth_metrics.json", task_name="ragtruth")
    if not keep_intermediates:
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
        except Exception:
            pass
    return metrics_found

def run_ms_marco(model, sampling_params, format_prompt, dataset_key="microsoft/ms_marco", dataset_config="v2.1", split="test", out_dir=OUT_ROOT / "ms_marco", batch_size=8, max_tokens=64, temp=0.0, top_p=1.0, keep_intermediates: bool = False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    safe_max_tokens = min(max_tokens, DEFAULT_MODEL_MAX_TOKENS - 100)
    if safe_max_tokens != max_tokens:
        logging.warning(f"[msmarco] Reduced max_tokens from {max_tokens} to {safe_max_tokens} for safety")
        max_tokens = safe_max_tokens
    
    from datasets import load_dataset
    logging.info("[msmarco] Loading dataset %s config=%s split=%s", dataset_key, dataset_config, split)
    ds = load_dataset(dataset_key, dataset_config, split=split)
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    references = []
    queries = []
    for ex in ds:
        qid = ex.get("query_id") or ex.get("id") or str(len(queries))
        query = ex.get("query") or ex.get("question") or ex.get("query_text") or ""
        answers = ex.get("answers") or []
        if not isinstance(answers, list):
            answers = [str(answers)]
        references.append({"query_id": qid, "answers": answers if answers else [""]})
        queries.append((qid, query))
    ref_path = out_dir / "msmarco_references.jsonl"
    cand_path = out_dir / "msmarco_candidates.jsonl"
    with open(ref_path, "w", encoding="utf-8") as fh:
        for rec in references:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    from vllm import SamplingParams as VSamplingParams
    sp = VSamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=False)
    candidates = []
    logging.info(f"[msmarco] Generating answers for {len(queries)} queries (batch {batch_size})")
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        max_input_tokens = DEFAULT_MODEL_MAX_TOKENS - max_tokens - 100
        prompts = [truncate_prompt(format_prompt(f"Answer the query: {q}", paragraph=None), max_input_tokens) for (_id, q) in batch]
        outs = model.generate(prompts, sp)
        for j, out in enumerate(outs):
            text = (out.outputs[0].text or "").strip()
            text = clean_selfrag_output(text)
            qid = batch[j][0]
            candidates.append({"query_id": qid, "answers": [text if text else ""]})
    with open(cand_path, "w", encoding="utf-8") as fh:
        for rec in candidates:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    script = resolve_script("msmarco")
    metrics_path = out_dir / "msmarco_metrics.json"
    try:
        script_text = open(script, "r", encoding="utf-8").read()
    except Exception:
        script_text = ""
    if "--out-file" in script_text or "--out_file" in script_text:
        cmd = [str(script), str(ref_path), str(cand_path), "--out-file", str(metrics_path)]
    else:
        cmd = [str(script), str(ref_path), str(cand_path)]
    metrics_found = run_and_capture_metrics(cmd, out_dir=out_dir, fallback_names=[metrics_path], metrics_basename=metrics_path.name, task_name="msmarco")

    if not keep_intermediates:
        for p in (ref_path, cand_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    return metrics_found

def run_triviaqa(model, sampling_params, format_prompt, dataset_key="mandarjoshi/trivia_qa", split="test", out_dir=OUT_ROOT / "trivia", batch_size=8, max_tokens=64, temp=0.0, top_p=1.0, trivia_gold_path: Optional[str]=None, keep_intermediates: bool = False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    safe_max_tokens = min(max_tokens, DEFAULT_MODEL_MAX_TOKENS - 100)
    if safe_max_tokens != max_tokens:
        logging.warning(f"[trivia] Reduced max_tokens from {max_tokens} to {safe_max_tokens} for safety")
        max_tokens = safe_max_tokens
    
    from datasets import load_dataset
    logging.info("[trivia] Loading dataset %s split=%s (config rc)", dataset_key, split)
    ds = load_dataset(dataset_key, "rc", split=split)
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    qlist = []
    for ex in ds:
        qid = ex.get("question_id") or ex.get("id") or ex.get("q_id") or ex.get("example_id") or str(len(qlist))
        question = ex.get("question") or ex.get("query") or ex.get("text") or ""
        context = ex.get("search_results") or ex.get("context") or None
        prompt_text = f"Answer the trivia question: {question}"
        if context:
            prompt_text += f"\n\nContext: {context}"
        qlist.append((str(qid), prompt_text))
    from vllm import SamplingParams as VSamplingParams
    sp = VSamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=False)
    preds = {}
    logging.info(f"[trivia] Generating {len(qlist)} predictions (batch {batch_size})")
    for i in range(0, len(qlist), batch_size):
        batch = qlist[i:i+batch_size]
        max_input_tokens = DEFAULT_MODEL_MAX_TOKENS - max_tokens - 100
        prompts = [truncate_prompt(format_prompt(t, paragraph=None), max_input_tokens) for (_id,t) in batch]
        outs = model.generate(prompts, sp)
        for j, out in enumerate(outs):
            ans = (out.outputs[0].text or "").strip()
            ans = clean_selfrag_output(ans)
            qid = batch[j][0]
            preds[qid] = ans
    pred_file = out_dir / "trivia_preds.json"
    with open(pred_file, "w", encoding="utf-8") as fh:
        json.dump(preds, fh)
    logging.info("[trivia] Wrote predictions")

    if trivia_gold_path:
        script = resolve_script("trivia")
        limited_gold_file = limit_gold_file(trivia_gold_path)
        metrics_path = out_dir / "trivia_metrics.json"
        try:
            script_text = open(script, "r", encoding="utf-8").read()
        except Exception:
            script_text = ""
        if "--out-file" in script_text or "--out_file" in script_text:
            cmd = [str(script), "--dataset_file", limited_gold_file, "--prediction_file", str(pred_file), "--out-file", str(metrics_path)]
        else:
            cmd = [str(script), "--dataset_file", limited_gold_file, "--prediction_file", str(pred_file)]
        metrics_found = run_and_capture_metrics(cmd, out_dir=out_dir, fallback_names=[metrics_path], metrics_basename=metrics_path.name, task_name="trivia")

        if not keep_intermediates:
            for p in (pred_file, limited_gold_file):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
        return metrics_found
    else:
        logging.info("[trivia] No trivia gold path provided; predictions saved.")
        return None

# Replace your run_natural_questions function with this corrected version:

def run_natural_questions(out_dir=OUT_ROOT / "nq", dataset_key: Optional[str]=None, split: Optional[str]=None, predictions_path: Optional[str]=None, make_empty=False, keep_intermediates: bool = False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    
    # Use corrected dataset configuration
    dsid = dataset_key or "google-research-datasets/natural_questions"
    cfg = "default"  # Always use default config for this dataset
    split_to_use = split or "validation"
    
    logging.info("[nq] Loading dataset %s config=%s split=%s", dsid, cfg, split_to_use)
    
    try:
        # Load with explicit config
        ds = load_dataset(dsid, cfg, split=split_to_use)
    except Exception as e:
        logging.error("[nq] Failed to load dataset %s with config %s: %s", dsid, cfg, e)
        # Try without config as fallback
        try:
            logging.info("[nq] Trying without config...")
            ds = load_dataset(dsid, split=split_to_use)
        except Exception as e2:
            logging.error("[nq] Also failed without config: %s", e2)
            raise e2
    
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    gold_path = out_dir / "nq_gold.json"
    
    with open(gold_path, "w", encoding="utf-8") as fh:
        json.dump([ex for ex in ds], fh)
    
    pred_path = predictions_path
    if pred_path is None and make_empty:
        pred_path = out_dir / "nq_empty_preds.json"
        preds = {}
        for ex in ds:
            ex_id = ex.get("example_id") or ex.get("id") or ex.get("query_id") or str(len(preds))
            preds[str(ex_id)] = {}
        with open(pred_path, "w", encoding="utf-8") as fh:
            json.dump(preds, fh)
    
    if pred_path is None:
        logging.info("[nq] No predictions provided. Skipping official NQ eval. Provide --nq-predictions to run official script.")
        return None

    script = resolve_script("nq")
    metrics_path = out_dir / "nq_metrics.json"
    
    try:
        script_text = open(script, "r", encoding="utf-8").read()
    except Exception:
        script_text = ""
    
    if "--out_file" in script_text or "--out-file" in script_text:
        cmd = [str(script), f"--gold_path={gold_path}", f"--predictions_path={pred_path}", f"--out_file={metrics_path}"]
    else:
        cmd = [str(script), f"--gold_path={gold_path}", f"--predictions_path={pred_path}"]
    
    metrics_found = run_and_capture_metrics(cmd, out_dir=out_dir, fallback_names=[metrics_path], metrics_basename=metrics_path.name, task_name="nq")

    if not keep_intermediates:
        for p in (gold_path, pred_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    return metrics_found

# --- BERT Score calculation ---
def calculate_bert_scores(predictions: dict, references: dict, task_name: str, out_dir: Path) -> Optional[str]:
    """
    Calculate BERT scores for predictions vs references.
    Returns path to BERT scores JSON file or None if failed.
    """
    try:
        from bert_score import score
        logging.info(f"[bert_score] Calculating BERT scores for {task_name}")
        
        # Align predictions and references
        pred_list = []
        ref_list = []
        
        for key in predictions:
            if key in references:
                pred_list.append(str(predictions[key]))
                ref_list.append(str(references[key]))
        
        if not pred_list:
            logging.warning(f"[bert_score] No aligned predictions/references for {task_name}")
            return None
            
        # Calculate BERT scores
        P, R, F1 = score(pred_list, ref_list, lang="en", verbose=False)
        
        bert_metrics = {
            "bert_precision": float(P.mean()),
            "bert_recall": float(R.mean()), 
            "bert_f1": float(F1.mean()),
            "bert_precision_std": float(P.std()),
            "bert_recall_std": float(R.std()),
            "bert_f1_std": float(F1.std()),
            "num_samples": len(pred_list)
        }
        
        bert_path = out_dir / f"{task_name}_bert_scores.json"
        with open(bert_path, "w", encoding="utf-8") as fh:
            json.dump(bert_metrics, fh, indent=2)
            
        logging.info(f"[bert_score] BERT F1: {bert_metrics['bert_f1']:.4f} for {task_name}")
        return str(bert_path)
        
    except Exception as e:
        logging.warning(f"[bert_score] Failed to calculate BERT scores for {task_name}: {e}")
        return None

# --- Enhanced task runners with BERT scores ---
def run_squad_v2_with_bert(model, sampling_params, format_prompt, dataset_key="rajpurkar/squad_v2", split="validation", out_dir=OUT_ROOT / "squad_v2", batch_size=8, max_tokens=128, temp=0.0, top_p=1.0, keep_intermediates: bool = False):
    """Run SQuAD v2 evaluation with both regular and BERT scores."""
    # Run regular evaluation
    metrics_path = run_squad_v2(model, sampling_params, format_prompt, dataset_key, split, out_dir, batch_size, max_tokens, temp, top_p, keep_intermediates)
    
    # Calculate BERT scores
    try:
        from datasets import load_dataset
        out_dir = Path(out_dir)
        ds = load_dataset(dataset_key, split=split)
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
        
        # Load predictions
        pred_file = out_dir / "preds.json"
        if pred_file.exists():
            with open(pred_file, "r", encoding="utf-8") as fh:
                preds = json.load(fh)
                
            # Build references from dataset
            references = {}
            for ex in ds:
                qid = ex.get("id") or ex.get("question_id") or ex.get("qid")
                if qid:
                    answers = ex.get("answers", {}).get("text", [])
                    # Use first answer as reference for BERT score
                    references[qid] = answers[0] if answers else ""
            
            bert_path = calculate_bert_scores(preds, references, "squad_v2", out_dir)
            
            # Combine metrics if both exist
            if metrics_path and bert_path and os.path.exists(metrics_path) and os.path.exists(bert_path):
                with open(metrics_path, "r", encoding="utf-8") as fh:
                    regular_metrics = json.load(fh)
                with open(bert_path, "r", encoding="utf-8") as fh:
                    bert_metrics = json.load(fh)
                
                combined_metrics = {**regular_metrics, **bert_metrics}
                combined_path = out_dir / "combined_metrics.json"
                with open(combined_path, "w", encoding="utf-8") as fh:
                    json.dump(combined_metrics, fh, indent=2)
                return str(combined_path)
    except Exception as e:
        logging.warning(f"[squad_bert] Failed to calculate BERT scores: {e}")
    
    return metrics_path

# --- CLI --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument("--cache-dir", type=str, default=MODEL_CACHE_DIR)
    p.add_argument("--download-datasets", action="store_true")
    p.add_argument("--max-examples", type=int, default=None, help="Override MAX_SAMPLES")
    p.add_argument("--tgi-host", type=str, default=DEFAULT_TGI_HOST)
    p.add_argument("--tgi-port", type=int, default=DEFAULT_TGI_PORT)
    p.add_argument("--run", type=str, default="all", help="Comma-separated list of tasks: ragtruth,squad,hotpot,msmarco,nq,trivia or 'all'")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--nq-predictions", type=str, default=None)
    p.add_argument("--nq-empty", action="store_true")
    p.add_argument("--trivia-gold", type=str, default=None)
    p.add_argument("--msmarco-config", type=str, default="v2.1")
    p.add_argument("--squad-split", type=str, default="validation")
    p.add_argument("--hotpot-split", type=str, default="validation")
    p.add_argument("--msmarco-split", type=str, default="test")
    p.add_argument("--trivia-split", type=str, default="test")
    p.add_argument("--nq-split", type=str, default="validation")
    p.add_argument("--keep-intermediates", action="store_true")
    p.add_argument("--calculate-bert", action="store_true", help="Calculate BERT scores in addition to regular metrics")
    return p.parse_args()

# Replace the relevant sections in your main() function with these updated calls:

def main():
    global MAX_SAMPLES, KEEP_INTERMEDIATES, GLOBAL_TOKENIZER
    args = parse_args()
    KEEP_INTERMEDIATES = bool(args.keep_intermediates)
    check_disk_space(".")
    tasks_arg = args.run or "all"
    tasks = [t.strip().lower() for t in tasks_arg.split(",")]
    if "all" in tasks:
        tasks = ["ragtruth","squad","hotpot","msmarco","nq","trivia"]

    if args.max_examples is not None:
        MAX_SAMPLES = int(args.max_examples)
        logging.info(f"[main] MAX_SAMPLES set to {MAX_SAMPLES}")

    model, tokenizer, sampling_params, format_prompt = load_model_and_tokenizer(MODEL_NAME, args.cache_dir)
    GLOBAL_TOKENIZER = tokenizer

    # Optional dataset downloads with correct configurations
    if "all" in tasks and args.download_datasets:
        from datasets import load_dataset
        hf_tasks = {
            "hotpot": ("hotpotqa/hotpot_qa", "distractor", "validation"),
            "msmarco": ("microsoft/ms_marco", "v2.1", "test"),
            "nq": ("google-research-datasets/natural_questions", "default", "validation"),
            "trivia": ("mandarjoshi/trivia_qa", "rc", "test"),
            "ragtruth": ("wandbRAGTruth-processed", None, "test"),
            "squad": ("rajpurkar/squad_v2", None, "validation"),
        }
        for k, (dsid, cfg, split) in hf_tasks.items():
            try:
                logging.info(f"[dl] Downloading {dsid} (cfg={cfg}) split={split}")
                _ = load_dataset(dsid, cfg, split=split) if cfg else load_dataset(dsid, split=split)
            except Exception as e:
                logging.warning(f"[dl] Failed downloading {dsid}: {e}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_summary = {}

    def run_task_safely(name, fn, *args, **kwargs):
        try:
            mpath = fn(*args, **kwargs)
            if mpath and os.path.exists(mpath):
                with open(mpath, "r", encoding="utf-8") as fh:
                    try:
                        metrics_data = json.load(fh)
                        metrics_summary[name] = metrics_data
                        logging.info(f"[main] {name} metrics loaded successfully: {len(metrics_data)} metrics")
                    except Exception as e:
                        metrics_summary[name] = {"_raw_metrics_file": str(mpath), "_parse_error": str(e)}
                        logging.warning(f"[main] {name} metrics exist but could not be parsed as JSON: {e}")
            else:
                logging.warning(f"[main] {name} metrics not found. Check outputs/{name}/*_stderr.txt")
                stdout_file = OUT_ROOT / name / f"{name}_stdout.txt"
                if stdout_file.exists():
                    try:
                        stdout_content = stdout_file.read_text(encoding="utf-8")
                        extracted_metrics = extract_metrics_from_text(stdout_content)
                        if extracted_metrics:
                            metrics_summary[name] = extracted_metrics
                            logging.info(f"[main] {name} metrics extracted from stdout")
                        else:
                            metrics_summary[name] = {"error": "metrics file not found", "stdout_checked": True}
                    except Exception:
                        metrics_summary[name] = {"error": "metrics file not found"}
                else:
                    metrics_summary[name] = {"error": "metrics file not found"}
        except Exception as e:
            logging.exception(f"[main] Exception while running {name}: {e}")
            metrics_summary[name] = {"error": str(e)}

    # Run tasks using the corrected dataset configurations
    if "ragtruth" in tasks:
        logging.info("[main] Running RAGTruth")
        ragtruth_samples = get_task_max_samples("ragtruth", args)
        ragtruth_tokens = get_task_max_tokens("ragtruth", args)
        ragtruth_batch_size = get_task_batch_size("ragtruth", args)
        logging.info(f"[main] RAGTruth using {ragtruth_samples} samples, {ragtruth_tokens} tokens, batch_size={ragtruth_batch_size}")
        out_file = OUT_ROOT / "ragtruth_preds.jsonl"
        old_max = MAX_SAMPLES
        MAX_SAMPLES = ragtruth_samples
        run_task_safely("ragtruth", run_ragtruth_adapter_and_eval, args.tgi_host, args.tgi_port, "wandbRAGTruth-processed", None, "test", str(out_file), "", model, sampling_params, KEEP_INTERMEDIATES)
        MAX_SAMPLES = old_max

    if "squad" in tasks:
        logging.info("[main] Running SQuAD v2")
        squad_samples = get_task_max_samples("squad", args)
        squad_tokens = get_task_max_tokens("squad", args)
        squad_batch_size = get_task_batch_size("squad", args)
        logging.info(f"[main] SQuAD v2 using {squad_samples} samples, {squad_tokens} tokens, batch_size={squad_batch_size}")
        old_max = MAX_SAMPLES
        MAX_SAMPLES = squad_samples
        if args.calculate_bert:
            run_task_safely("squad_v2", run_squad_v2_with_bert, model, sampling_params, format_prompt, "rajpurkar/squad_v2", "validation", OUT_ROOT/"squad_v2", squad_batch_size, squad_tokens, args.temp, args.top_p, KEEP_INTERMEDIATES)
        else:
            run_task_safely("squad_v2", run_squad_v2, model, sampling_params, format_prompt, "rajpurkar/squad_v2", "validation", OUT_ROOT/"squad_v2", squad_batch_size, squad_tokens, args.temp, args.top_p, KEEP_INTERMEDIATES)
        MAX_SAMPLES = old_max

    if "hotpot" in tasks:
        logging.info("[main] Running HotPotQA")
        hotpot_samples = get_task_max_samples("hotpot", args)
        hotpot_tokens = get_task_max_tokens("hotpot", args)
        hotpot_batch_size = get_task_batch_size("hotpot", args)
        logging.info(f"[main] HotPotQA using {hotpot_samples} samples, {hotpot_tokens} tokens, batch_size={hotpot_batch_size}")
        old_max = MAX_SAMPLES
        MAX_SAMPLES = hotpot_samples
        run_task_safely("hotpot", run_hotpot, model, sampling_params, format_prompt, "hotpotqa/hotpot_qa", "validation", OUT_ROOT/"hotpot", hotpot_batch_size, hotpot_tokens, args.temp, args.top_p, KEEP_INTERMEDIATES)
        MAX_SAMPLES = old_max

    if "msmarco" in tasks:
        logging.info("[main] Running MS MARCO")
        msmarco_samples = get_task_max_samples("msmarco", args)
        msmarco_tokens = get_task_max_tokens("msmarco", args)
        msmarco_batch_size = get_task_batch_size("msmarco", args)
        logging.info(f"[main] MS MARCO using {msmarco_samples} samples, {msmarco_tokens} tokens, batch_size={msmarco_batch_size}")
        old_max = MAX_SAMPLES
        MAX_SAMPLES = msmarco_samples
        run_task_safely("msmarco", run_ms_marco, model, sampling_params, format_prompt, "microsoft/ms_marco", "v2.1", "test", OUT_ROOT/"ms_marco", msmarco_batch_size, msmarco_tokens, args.temp, args.top_p, KEEP_INTERMEDIATES)
        MAX_SAMPLES = old_max

    if "nq" in tasks:
        logging.info("[main] Running Natural Questions")
        nq_samples = get_task_max_samples("nq", args)
        logging.info(f"[main] Natural Questions using {nq_samples} samples")
        old_max = MAX_SAMPLES
        MAX_SAMPLES = nq_samples
        run_task_safely("nq", run_natural_questions, OUT_ROOT/"nq", "google-research-datasets/natural_questions", "validation", args.nq_predictions, args.nq_empty, KEEP_INTERMEDIATES)
        MAX_SAMPLES = old_max

    if "trivia" in tasks:
        logging.info("[main] Running TriviaQA")
        trivia_samples = get_task_max_samples("trivia", args)
        trivia_tokens = get_task_max_tokens("trivia", args)
        trivia_batch_size = get_task_batch_size("trivia", args)
        logging.info(f"[main] TriviaQA using {trivia_samples} samples, {trivia_tokens} tokens, batch_size={trivia_batch_size}")
        old_max = MAX_SAMPLES
        MAX_SAMPLES = trivia_samples
        run_task_safely("trivia", run_triviaqa, model, sampling_params, format_prompt, "mandarjoshi/trivia_qa", "test", OUT_ROOT/"trivia", trivia_batch_size, trivia_tokens, args.temp, args.top_p, args.trivia_gold, KEEP_INTERMEDIATES)
        MAX_SAMPLES = old_max

    # aggregate summary
    summary_path = OUT_ROOT / "all_eval_scores.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_summary, fh, indent=2, ensure_ascii=False)
    logging.info("[main] Aggregated metrics written to %s", summary_path)
    logging.info("Outputs saved under: %s", OUT_ROOT.resolve())
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for task, metrics in metrics_summary.items():
        print(f"\n{task.upper()}:")
        if "error" in metrics:
            print(f"  ERROR: {metrics['error']}")
        else:
            # Print key metrics
            key_metrics = ["exact_match", "f1", "accuracy", "em", "bert_f1", "bert_precision", "bert_recall"]
            for metric in key_metrics:
                if metric in metrics:
                    print(f"  {metric}: {metrics[metric]:.4f}")

if __name__ == "__main__":
    main()
