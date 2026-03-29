#!/usr/bin/env python3
"""
ui_inference.py — Single-question KG-TRACES inference for the UI.

Key design decisions:

1. PATH GENERATION (Phase 1) — unchanged from gen_predict_path.py.
   Relation mode: model outputs <PATH>rel1<SEP>rel2</PATH>
   Triple mode:   model outputs <PATH>s -> r -> o -> r2 -> o2</PATH>
   The model does NOT produce entity names for relation paths — that requires
   a live KG lookup which we do not have. The relation labels ARE the path.

2. REASONING STEPS — synthesized from paths, NOT from model output.
   The model was fine-tuned to answer concisely and will not produce
   structured reasoning text reliably. Instead we build human-readable
   reasoning steps programmatically from the parsed paths + final answer.
   This is reliable and always populated.

3. ANSWER GENERATION (Phase 2) — standard PromptBuilder, no custom instruction.
   The model outputs its standard answer format (e.g. "**Answer**:\nMemphis").
   We parse the answer text and strip formatting.

4. UI_EVENT routing:
   - UI_EVENT step          → Live Events tab (pipeline status lines)
   - UI_EVENT reasoning_step → Answer tab timeline (KG reasoning narrative)
   - UI_EVENT path           → Graph + Path cards
   - UI_EVENT token          → Answer streaming
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

current_dir  = Path(__file__).parent.resolve()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

import utils
from qa_prediction.build_qa_input import PromptBuilder
from tools.logger_factory import setup_logger

logger = setup_logger("ui_inference")

# ── Path generation instructions (identical to gen_predict_path.py) ───────────
PATH_RE = re.compile(r"<PATH>(.*?)</PATH>", re.IGNORECASE | re.DOTALL)

INSTRUCTION_RELATION = (
    "Please generate a valid reasoning relation path that can be helpful "
    "for answering the following question:"
)

INSTRUCTION_TRIPLE = """Please generate a valid reasoning triple path that can be helpful for answering the following question
- The reasoning triples path should follow the format:
  <PATH>subject -> relation -> object</PATH>
- If multiple triples are needed, output them in a logical sequence like:
  <PATH>s1 -> r1 -> o1 -> r2 -> o2</PATH>
- If no meaningful path can be generated, return <PATH>NONE</PATH>.
"""


# ── UI event emitters ─────────────────────────────────────────────────────────
def ui_step(icon: str, html: str):
    """Pipeline status line — appears in Live Events tab."""
    print(f"UI_EVENT step icon={icon} html={html}", flush=True)

def ui_reasoning_step(index: int, label: str, detail: str, path_label: str = ""):
    """KG reasoning step — appears in Answer tab timeline."""
    print(f"UI_EVENT reasoning_step {json.dumps({'index': index, 'label': label, 'detail': detail, 'path_label': path_label})}", flush=True)

def ui_path(index: int, path_data: dict):
    print(f"UI_EVENT path {json.dumps({'index': index, 'path_data': path_data})}", flush=True)

def ui_token(ch: str):
    print(f"UI_EVENT token {json.dumps(ch)}", flush=True)

def ui_stats(paths: int, conf, tokens: int):
    print(f"UI_EVENT stats {json.dumps({'paths': paths, 'conf': conf, 'tokens': tokens})}", flush=True)

def ui_done():
    print("UI_EVENT done", flush=True)

def ui_error(msg: str):
    print(f"UI_EVENT error {msg}", flush=True)


# ── StoppingCriteria ──────────────────────────────────────────────────────────
class _AnswerEndCriteria(StoppingCriteria):
    def __init__(self, stop_id_seqs: list[list[int]]):
        self.stop_id_seqs = stop_id_seqs
        self._max_len = max(len(s) for s in stop_id_seqs)

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        tail = input_ids[0, -self._max_len:].tolist()
        for seq in self.stop_id_seqs:
            n = len(seq)
            if tail[-n:] == seq:
                return True
        return False


def build_stopping_criteria(tokenizer, explain: bool, early_exit: bool):
    if not early_exit or explain:
        return None
    stop_strings = ["]\n", "]\n\n"]
    stop_id_seqs = [
        tokenizer.encode(s, add_special_tokens=False)
        for s in stop_strings
    ]
    stop_id_seqs = [s for s in stop_id_seqs if s]
    if not stop_id_seqs:
        return None
    return StoppingCriteriaList([_AnswerEndCriteria(stop_id_seqs)])


# ── Path parsers ──────────────────────────────────────────────────────────────
def parse_relation_prediction(predictions: list[str]) -> list[list[str]]:
    """
    Parse beam outputs → list[list[str]].
    Each inner list is one path of relation labels.
    Input format: <PATH>rel1<SEP>rel2<SEP>rel3</PATH>
    """
    results = []
    seen = set()
    for p in predictions:
        m = PATH_RE.search(p)
        if not m:
            continue
        rules = [r.strip() for r in m.group(1).split("<SEP>") if r.strip()]
        if rules:
            key = tuple(rules)
            if key not in seen:
                seen.add(key)
                results.append(rules)
    return results


def parse_triple_prediction(predictions: list[str]) -> list[list[tuple]]:
    """
    Parse beam outputs → list[list[tuple(s,r,o)]].

    Handles all observed model output formats:
      1. Fully tagged:        <PATH>s -> r -> o</PATH>
      2. Missing opening tag: s -> r -> o</PATH>       ← model often outputs this
      3. No tags at all:      s -> r -> o              ← model skips tags sometimes
      4. Unicode arrows:      s → r → o
      5. Multi-hop:           s -> r -> o -> r2 -> o2
      6. Multi-line:          one triple per line, under **Reasoning Triple**:
    """
    results = []

    for prediction in predictions:
        # Strip any **Reasoning Triple**: header the model prepends
        text = re.sub(r"^\*+Reasoning\s*Triple\*+:\s*", "", prediction,
                      flags=re.IGNORECASE).strip()

        path_lines: list[str] = []

        # Strategy 1: fully-tagged <PATH>...</PATH> blocks (correct format)
        full_matches = re.findall(r"<PATH>(.*?)</PATH>", text,
                                  re.IGNORECASE | re.DOTALL)
        if full_matches:
            path_lines = full_matches
        else:
            # Strategy 2 + 3: lines that end with </PATH> or have -> / → but no tags
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Remove trailing </PATH> (possibly duplicated: </PATH></PATH>)
                line = re.sub(r"(</PATH>)+\s*$", "", line,
                               flags=re.IGNORECASE).strip()
                # Remove leading <PATH> if present
                line = re.sub(r"^<PATH>\s*", "", line,
                               flags=re.IGNORECASE).strip()
                # Only keep lines that look like triples
                if "->" in line or "→" in line:
                    path_lines.append(line)

        # Parse each candidate line into a list of (s, r, o) triples
        for content in path_lines:
            content = content.strip()
            if not content or content.upper() == "NONE":
                continue
            if "<SEP>" in content:
                continue  # relation-format path — skip

            if "->" in content:
                parts = [p.strip() for p in content.split("->")]
            elif "→" in content:
                parts = [p.strip() for p in content.split("→")]
            else:
                continue

            parts = [p for p in parts if p]
            if len(parts) < 3:
                continue
            # Even count means malformed — drop last part and retry
            if len(parts) % 2 == 0:
                parts = parts[:-1]
            if len(parts) < 3:
                continue

            triples = []
            for i in range(0, len(parts) - 2, 2):
                s, r, o = parts[i], parts[i + 1], parts[i + 2]
                if s and r and o:
                    triples.append((s, r, o))
            if triples:
                results.append(triples)

    # Deduplicate paths while preserving order
    seen: set = set()
    deduped: list = []
    for path in results:
        key = tuple(tuple(t) for t in path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


# ── Relation label → English ──────────────────────────────────────────────────
def rel_to_english(rel: str) -> str:
    """
    Convert a Freebase-style relation like 'people.deceased_person.place_of_death'
    into readable English like 'place of death'.
    """
    parts = rel.split(".")
    # Last part is most descriptive
    last = parts[-1].replace("_", " ")
    # Add domain context if ambiguous (single word)
    if len(last.split()) == 1 and len(parts) >= 2:
        domain = parts[0].replace("_", " ")
        return f"{last} ({domain})"
    return last


# ── Synthesize reasoning steps from paths ────────────────────────────────────
def synthesize_reasoning_steps(
    question: str,
    parsed_paths: list,
    path_type: str,
    answer: str,
) -> list[dict]:
    """
    Build human-readable reasoning steps from the parsed paths + final answer.

    This is synthesized programmatically because:
    - The model was fine-tuned to answer concisely, not produce structured reasoning
    - Relation paths have no entity names (no KG available) so we describe hops
    - Triple paths have full (s, r, o) so we show each entity transition

    Returns list of {label, detail, path_label} dicts for the UI timeline.
    """
    steps = []

    if not parsed_paths:
        steps.append({
            "label": "No paths found",
            "detail": "Model answered from parametric knowledge (no KG path available)",
            "path_label": ""
        })
        return steps

    if path_type == "relation":
        # ── Step 1: question ────────────────────────────────────────────────
        steps.append({
            "label": "Question received",
            "detail": question.strip().rstrip("?"),
            "path_label": ""
        })

        # ── Steps 2+: each unique path as a reasoning chain ─────────────────
        for pi, path in enumerate(parsed_paths):
            # Build a readable chain string: rel1 → rel2 → ... → answer
            chain_parts = []
            for rel in path:
                chain_parts.append(rel_to_english(rel))
            chain_str = " → ".join(chain_parts)

            # Detail: the raw relation labels
            raw_chain = " → ".join(path)

            steps.append({
                "label": f"Path {pi + 1}: {chain_str}",
                "detail": raw_chain,
                "path_label": f"P{pi + 1}"
            })

        # ── Consensus note if multiple paths ────────────────────────────────
        if len(parsed_paths) > 1:
            unique_ends = list(dict.fromkeys(
                p[-1] for p in parsed_paths if p
            ))
            steps.append({
                "label": f"{len(parsed_paths)} paths evaluated",
                "detail": f"All paths suggest the same relation chain towards the answer",
                "path_label": ""
            })

        # ── Final: answer ───────────────────────────────────────────────────
        if answer:
            steps.append({
                "label": f"Answer: {answer}",
                "detail": f"Retrieved via knowledge graph traversal",
                "path_label": ""
            })

    elif path_type == "triple":
        steps.append({
            "label": "Question received",
            "detail": question.strip().rstrip("?"),
            "path_label": ""
        })

        for pi, path in enumerate(parsed_paths):
            steps.append({
                "label": f"Triple path {pi + 1}",
                "detail": f"{len(path)} hop{'s' if len(path) > 1 else ''}",
                "path_label": f"P{pi + 1}"
            })
            for j, (s, r, o) in enumerate(path):
                is_last = (j == len(path) - 1)
                steps.append({
                    "label": f"{s}  →  {o}",
                    "detail": f"via {r}  ({rel_to_english(r)})",
                    "path_label": f"P{pi + 1} · hop {j + 1}"
                })

        if answer:
            steps.append({
                "label": f"Answer: {answer}",
                "detail": "Retrieved via KG triple path",
                "path_label": ""
            })

    return steps


def paths_to_ui_chain(parsed_paths: list, path_type: str) -> list[dict]:
    """Convert parsed paths to the chain format for graph + path cards."""
    ui_paths = []
    for path in parsed_paths:
        if path_type == "relation":
            chain = [{"type": "relation", "label": r} for r in path]
            ui_paths.append({"type": "kg", "chain": chain, "steps": []})
        elif path_type == "triple":
            chain, steps = [], []
            for i, (s, r, o) in enumerate(path):
                is_last = (i == len(path) - 1)
                if i == 0:
                    chain.append({"type": "entity", "label": s, "isAnswer": False})
                chain.append({"type": "relation", "label": r})
                chain.append({"type": "entity", "label": o, "isAnswer": is_last})
                steps.append({"relation": r, "object": o, "source": "kg"})
            ui_paths.append({"type": "kg", "chain": chain, "steps": steps})
    return ui_paths


def parse_answer_text(raw: str) -> str:
    """
    Extract just the answer from the model's raw output.
    Handles:
      **Answer**:\nMemphis
      **Answer**: Memphis
      Memphis  (model already brief)
    """
    m = re.search(r'\*{0,2}Answer\*{0,2}:\s*', raw, re.IGNORECASE)
    if m:
        return raw[m.end():].strip()
    return raw.strip()


def extract_path_terminal_answers(parsed_paths: list, path_type: str) -> list[str]:
    """
    Extract the KG-supported answer candidates from path terminal entities.

    For triple paths: the last object in each path is the entity the KG
    evidence directly points to — this is more reliable than the model's
    parametric answer when the two disagree.

    For relation paths: no entity names are available (no KG was walked),
    so returns an empty list.

    Deduplicates while preserving path order.
    """
    if path_type != "triple" or not parsed_paths:
        return []

    seen: set[str] = set()
    terminals: list[str] = []
    for path in parsed_paths:
        if not path:
            continue
        # path = list[tuple(s, r, o)] — terminal is last tuple's object
        last_triple = path[-1]
        terminal = last_triple[2].strip() if len(last_triple) >= 3 else None
        if terminal and terminal not in seen:
            seen.add(terminal)
            terminals.append(terminal)
    return terminals


# ── Phase 1: path generation ──────────────────────────────────────────────────
def generate_paths(question: str, args, tokenizer, model) -> tuple[list, list[str]]:
    prompter = utils.InstructFormater(args.prompt_path)
    q = question if question.endswith("?") else question + "?"

    if args.path_type == "relation":
        prompt = prompter.format(system=INSTRUCTION_RELATION, query="**Question**:\n" + q)
    else:
        prompt = prompter.format(system=INSTRUCTION_TRIPLE, query="**Question**:\n" + q)

    ui_step("▸", f"Tokenising input ({len(prompt.split())} words)")

    inputs = tokenizer(
        [prompt], padding=True, truncation=True,
        add_special_tokens=False, return_tensors="pt",
    )
    input_ids      = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    ui_step("▸", f"Beam search: {args.n_beam} beams × {args.max_new_tokens} max tokens")

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=args.n_beam,
            num_return_sequences=args.n_beam,
            early_stopping=False,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=args.max_new_tokens,
        )

    raw_preds = tokenizer.batch_decode(
        output.sequences[:, input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    raw_preds = [p.strip() for p in raw_preds]

    # Log raw beams for Live Events tab
    for i, raw in enumerate(raw_preds):
        ui_step("▸", f"Beam {i + 1}: {raw[:100].replace(chr(10), ' ')}")

    ui_step("✓", f"Parsing {len(raw_preds)} beams")

    if args.path_type == "relation":
        parsed = parse_relation_prediction(raw_preds)
    else:
        parsed = parse_triple_prediction(raw_preds)

    if len(parsed) == 0:
        ui_step("⚠", "0 paths parsed — check Live Events for raw beam output")

    return parsed, raw_preds


# ── Phase 2: answer generation ────────────────────────────────────────────────
def generate_answer(
    question: str,
    parsed_paths: list,
    args,
    tokenizer,
    model,
    stopping_criteria,
) -> tuple[str, str, float | None]:
    """
    Returns: (answer_text, raw_output, confidence)
    """
    sample = {
        "question":                 question,
        "choices":                  [],
        "graph":                    [],
        "q_entity":                 [],
        "ground_paths":             [],
        "predicted_relation_paths": parsed_paths if args.path_type == "relation" else [],
        "predicted_triple_paths":   parsed_paths if args.path_type == "triple"   else [],
        "faithfulness_note":        "",
    }

    input_builder = PromptBuilder(
        add_path       = True,
        use_pred_path  = True,
        pred_path_type = args.path_type,
        maximun_token  = args.max_input_tokens,
        tokenize       = lambda t: len(tokenizer.tokenize(t)),
    )

    prompt, _ = input_builder.process_input(sample)

    if args.chat_model:
        messages = [{"role": "user", "content": prompt}]
        prompt   = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    ui_step("◉", "Generating answer from reasoning paths")

    inputs    = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_ids = inputs.input_ids

    with torch.inference_mode():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=args.max_output_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=stopping_criteria,
        )

    new_ids    = gen_out.sequences[0][input_ids.shape[1]:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    answer     = parse_answer_text(raw_output)

    conf = None
    if gen_out.scores:
        stacked   = torch.stack(gen_out.scores, dim=0)
        log_probs = F.log_softmax(stacked[:, 0, :], dim=-1)
        n         = min(len(new_ids), log_probs.shape[0])
        token_lp  = log_probs[torch.arange(n), new_ids[:n]]
        conf      = round(token_lp.mean().item(), 4)

    return answer, raw_output, conf


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",          required=True)
    parser.add_argument("--path_type",         default="triple", choices=["relation", "triple"])
    parser.add_argument("--n_beam",            type=int, default=3)
    parser.add_argument("--model_path",        default="models/KG-TRACES")
    parser.add_argument("--model_name",        default="KG-TRACES")
    parser.add_argument("--model_type",        default="webqsp_cwq_tuned")
    parser.add_argument("--prompt_path",       default="prompts/qwen2.5.txt")
    parser.add_argument("--out_dir",           required=True)
    parser.add_argument("--max_new_tokens",    type=int, default=128)
    parser.add_argument("--max_output_tokens", type=int, default=512)
    parser.add_argument("--max_input_tokens",  type=int, default=4096)
    parser.add_argument("--chat_model",        default=True,
                        type=lambda x: str(x).lower() == "true")
    parser.add_argument("--early_exit",        default=True,
                        type=lambda x: str(x).lower() == "true")
    parser.add_argument("--explain",           action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    ui_step("◈", f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"

    use_cuda    = torch.cuda.is_available()
    torch_dtype = torch.float16 if use_cuda else torch.float32
    device_map  = "auto"        if use_cuda else "cpu"
    attn_impl   = "sdpa"        if use_cuda else "eager"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    model.eval()
    ui_step("✓", f"Model loaded ({'GPU' if use_cuda else 'CPU'})")

    stopping_criteria = build_stopping_criteria(
        tokenizer, explain=args.explain, early_exit=args.early_exit
    )

    # ── Phase 1: path generation ───────────────────────────────────────────────
    ui_step("◈", f"Predicting {args.path_type} paths (beam={args.n_beam})")
    parsed_paths, raw_beam_outputs = generate_paths(args.question, args, tokenizer, model)
    ui_step("✓", f"Found {len(parsed_paths)} valid path(s)")

    # Emit path events for graph + path cards + triples tab
    ui_chains = paths_to_ui_chain(parsed_paths, args.path_type)
    for i, chain in enumerate(ui_chains):
        ui_path(i, chain)
        time.sleep(0.04)

    # ── Phase 2: answer generation ─────────────────────────────────────────────
    answer, raw_output, conf = generate_answer(
        args.question, parsed_paths, args, tokenizer, model, stopping_criteria
    )
    ui_step("✓", "Answer generated — streaming")

    for ch in answer:
        ui_token(ch)
        time.sleep(0.008)

    n_tokens = len(tokenizer.tokenize(raw_output))
    ui_stats(paths=len(parsed_paths), conf=conf, tokens=n_tokens)
    if conf is not None:
        ui_step("▸", f"Confidence (mean log-prob): {conf}")

    # ── Synthesize reasoning steps from paths + answer ─────────────────────────
    # Done AFTER answer generation so we have the final answer text to include.
    reasoning_steps = synthesize_reasoning_steps(
        args.question, parsed_paths, args.path_type, answer
    )

    # Emit each reasoning step as a structured UI event for the Answer tab
    for i, step in enumerate(reasoning_steps):
        ui_reasoning_step(i, step["label"], step["detail"], step.get("path_label", ""))
        time.sleep(0.03)

    # ── Serialise ──────────────────────────────────────────────────────────────
    if args.path_type == "triple":
        serialized_paths = [
            [list(t) if isinstance(t, tuple) else list(t) for t in path]
            for path in parsed_paths
        ]
    else:
        serialized_paths = parsed_paths  # list[list[str]] — already JSON-safe

    # KG-derived answer candidates — terminal entities from the paths.
    # These are more trustworthy than the model's parametric answer
    # when the two disagree (the model may hallucinate; the path terminals
    # come directly from the structured prediction).
    path_derived_answers = extract_path_terminal_answers(parsed_paths, args.path_type)

    # Emit path-derived answers as a UI event so the HTML can show them live
    if path_derived_answers:
        ui_step("▸", f"Path evidence: {', '.join(path_derived_answers)}")

    # Flag if model answer conflicts with path evidence
    model_answer_items = [a.strip() for a in answer.split('\n') if a.strip()]
    conflicts = path_derived_answers and not any(
        any(p.lower() in m.lower() or m.lower() in p.lower()
            for m in model_answer_items)
        for p in path_derived_answers
    )
    if conflicts:
        ui_step("⚠", f"Model answer may conflict with path evidence — path says: {path_derived_answers[0]}")

    output = {
        "id":                    "ui_query",
        "question":              args.question,
        "prediction":            [answer],
        "path_derived_answers":  path_derived_answers,
        "raw_model_output":      raw_output,
        "reasoning_steps":       reasoning_steps,
        "confidence":            conf,
        "num_of_paths":          len(parsed_paths),
        "output_token_num":      n_tokens,
        "parsed_paths":          serialized_paths,
        "raw_beam_outputs":      raw_beam_outputs,
        "path_type":             args.path_type,
        "model_name":            args.model_name,
        "model_type":            args.model_type,
        "n_beam":                args.n_beam,
        "early_exit_used":       stopping_criteria is not None,
        "answer_conflict":       conflicts,
    }

    out_file = out_dir / "output.json"
    out_file.write_text(json.dumps(output, indent=2))
    (out_dir / "predictions.jsonl").write_text(json.dumps(output) + "\n")

    ui_step("✓", "Results saved")
    ui_done()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        ui_error(str(e))
        traceback.print_exc()
        sys.exit(1)
