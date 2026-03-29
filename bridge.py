#!/usr/bin/env python3
"""
bridge.py — Slurm job submitter + log poller for the KG-TRACES UI.

Run once on coe-hpc3 (login node):
    python bridge.py --port 8765

SSH tunnel from your laptop:
    ssh -L 8765:localhost:8765 -t 018228028@coe-hpc1.sjsu.edu "ssh -L 8765:localhost:8765 coe-hpc3"

Endpoints:
    POST /submit  → writes job.sh matching predict.sh flags, runs sbatch
    GET  /poll    → tails slurm-<id>.out for the UI (called every 2s)
    GET  /result  → returns final predictions.jsonl content
    GET  /jobs    → lists recent jobs
    GET  /cancel  → runs scancel <slurm_id>
    GET  /health  → sanity check
"""

import argparse
import http.server
import json
import os
import re
import subprocess
import sys
import urllib.parse
from datetime import datetime
from pathlib import Path

# ── Repo layout ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.resolve()
JOBS_DIR  = REPO_ROOT / "results" / "ui_jobs"
SRC_DIR   = REPO_ROOT / "src"
PYTHON    = sys.executable

# ── Slurm defaults — match kg-traces.sbatch exactly ───────────────────────────
DEFAULT_PARTITION = os.environ.get("SLURM_PARTITION", "nsfqs")
DEFAULT_GRES      = os.environ.get("SLURM_GRES",      "")        # no --gres in your sbatch
DEFAULT_TIME      = os.environ.get("SLURM_TIME",       "48:00:00")
DEFAULT_MEM       = os.environ.get("SLURM_MEM",        "64G")
DEFAULT_CPUS      = os.environ.get("SLURM_CPUS",       "8")
DEFAULT_NODES     = "1"
DEFAULT_NTASKS    = "1"
CONDA_ENV         = os.environ.get("CONDA_ENV",         "kg_traces")

# ── predict.sh hardcoded values ────────────────────────────────────────────────
DEFAULT_MODEL_NAME  = "KG-TRACES"
DEFAULT_MODEL_PATH  = "models/KG-TRACES"
DEFAULT_MODEL_TYPE  = "webqsp_cwq_tuned"   # --model_type
DEFAULT_DATASET     = "webqsp"
DEFAULT_PATH_TYPE   = "triple"             # PRED_PATH_TYPE_LIST in predict.sh
DEFAULT_BATCH_SIZE  = 2                    # --batch_size=2 in predict.sh
DEFAULT_BEAM        = 3

JOBS_DIR.mkdir(parents=True, exist_ok=True)


# ── Job script builder ─────────────────────────────────────────────────────────
def build_job_script(job_dir: Path, cfg: dict) -> Path:
    """
    Generates an sbatch script that calls ui_inference.py — a single-question
    pipeline that loads the model once, generates paths, then generates the
    answer, all for the one question typed in the UI.

    This replaces the old approach of running predict_answer.py on the full
    test set every time.
    """

    job_id = cfg["job_id"]

    # Slurm resources
    partition  = cfg.get("partition", DEFAULT_PARTITION) or DEFAULT_PARTITION
    gres       = cfg.get("gres",      DEFAULT_GRES)      or DEFAULT_GRES
    slurm_time = cfg.get("time",      DEFAULT_TIME)      or DEFAULT_TIME
    mem        = cfg.get("mem",       DEFAULT_MEM)       or DEFAULT_MEM
    cpus       = cfg.get("cpus",      DEFAULT_CPUS)      or DEFAULT_CPUS

    # Inference config
    question   = cfg.get("question",  "")
    path_type  = cfg.get("pathType",  DEFAULT_PATH_TYPE)
    n_beam     = int(cfg.get("beams", DEFAULT_BEAM))
    model_name = cfg.get("model",     DEFAULT_MODEL_NAME)
    model_path = cfg.get("modelPath", DEFAULT_MODEL_PATH)
    model_type = cfg.get("modelType", DEFAULT_MODEL_TYPE)
    mode       = cfg.get("mode",      "reasoning")

    out_dir    = job_dir / "output"

    # --gres line only if non-empty (your sbatch has no --gres)
    gres_line  = f"#SBATCH --gres={gres}" if gres else ""

    # chat_model flag — true for instruction-tuned models like KG-TRACES/Qwen
    chat_model = "true"

    # include_reasoning maps to mode
    reasoning_flag = "--include_reasoning" if mode == "reasoning" else ""

    script = f"""#!/bin/bash
#SBATCH --job-name=kgt-ui-{job_id[:8]}
#SBATCH --output={job_dir}/slurm-%j.out
#SBATCH --error={job_dir}/slurm-%j.out
#SBATCH --partition={partition}
{gres_line}
#SBATCH --nodes={DEFAULT_NODES}
#SBATCH --ntasks={DEFAULT_NTASKS}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={slurm_time}

echo "=== Activating environment ==="
source ~/anaconda3/etc/profile.d/conda.sh
conda activate {CONDA_ENV}

echo "Running on host: $(hostname)"
echo "CUDA devices visible: $CUDA_VISIBLE_DEVICES"

echo "=== GPU CHECK ==="
nvidia-smi || echo "nvidia-smi not found or no GPU visible"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd {REPO_ROOT}
export PYTHONPATH={SRC_DIR}:$PYTHONPATH

echo "=== Single-question KG-TRACES inference ==="
echo "  question:   {question}"
echo "  path_type:  {path_type}"
echo "  n_beam:     {n_beam}"
echo "  model:      {model_name} ({model_path})"
echo "  out_dir:    {out_dir}"

{PYTHON} {SRC_DIR}/qa_prediction/ui_inference.py \\
    --question {json.dumps(question)} \\
    --path_type {path_type} \\
    --n_beam {n_beam} \\
    --model_path {model_path} \\
    --model_name {model_name} \\
    --model_type {model_type} \\
    --prompt_path prompts/qwen2.5.txt \\
    --out_dir {out_dir} \\
    --chat_model {chat_model} \\
    --max_new_tokens 128 \\
    --max_output_tokens 512

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "UI_EVENT error ui_inference.py exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi
"""

    script_path = job_dir / "job.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


# ── HTTP handler ───────────────────────────────────────────────────────────────
class BridgeHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        route  = parsed.path

        if route == "/health":
            self._json({
                "status":     "ok",
                "repo":       str(REPO_ROOT),
                "partition":  DEFAULT_PARTITION,
                "conda_env":  CONDA_ENV,
                "model_type": DEFAULT_MODEL_TYPE,
                "path_type":  DEFAULT_PATH_TYPE,
            })

        elif route == "/jobs":
            jobs = []
            if JOBS_DIR.exists():
                for jdir in sorted(JOBS_DIR.iterdir(), reverse=True)[:20]:
                    meta_f = jdir / "meta.json"
                    if meta_f.exists():
                        try:
                            m = json.loads(meta_f.read_text())
                            m["status"] = _job_status(jdir)
                            jobs.append(m)
                        except Exception:
                            pass
            self._json({"jobs": jobs})

        elif route == "/poll":
            job_id = params.get("job_id", "")
            after  = int(params.get("after", 0))
            jdir   = JOBS_DIR / job_id

            out_files = sorted(jdir.glob("slurm-*.out")) if jdir.exists() else []
            if not out_files:
                self._json({"lines": [], "offset": 0, "status": "pending"})
                return

            content = out_files[-1].read_bytes()
            chunk   = content[after:]
            lines   = chunk.decode("utf-8", errors="replace").splitlines()
            self._json({
                "lines":  lines,
                "offset": len(content),
                "status": _job_status(jdir),
            })

        elif route == "/result":
            job_id      = params.get("job_id", "")
            jdir        = JOBS_DIR / job_id
            out         = jdir / "output"
            output_json = out / "output.json"
            if output_json.exists():
                self._json(json.loads(output_json.read_text()))
                return
            preds = sorted(out.glob("**/predictions.jsonl")) if out.exists() else []
            if not preds:
                self._json({"error": "not ready"}, 404)
                return
            lines = [l for l in preds[-1].read_text().strip().splitlines() if l.strip()]
            self._json(json.loads(lines[-1]) if lines else {})

        elif route == "/cancel":
            slurm_id = params.get("slurm_id", "")
            if not slurm_id:
                self._json({"error": "no slurm_id"}, 400)
                return
            r = subprocess.run(["scancel", slurm_id], capture_output=True, text=True)
            self._json({"cancelled": r.returncode == 0, "msg": r.stdout + r.stderr})

        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            cfg = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return

        route = urllib.parse.urlparse(self.path).path

        if route == "/submit":
            ts     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            job_id = f"{ts}_{os.urandom(3).hex()}"
            jdir   = JOBS_DIR / job_id
            jdir.mkdir(parents=True)

            cfg["job_id"] = job_id
            (jdir / "input.json").write_text(json.dumps(cfg, indent=2))

            try:
                script_path = build_job_script(jdir, cfg)
            except Exception as e:
                self._json({"error": f"script build failed: {e}"}, 500)
                return

            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True, text=True, cwd=str(REPO_ROOT),
            )
            if result.returncode != 0:
                self._json({"error": result.stderr.strip() or result.stdout.strip()}, 500)
                return

            m        = re.search(r"(\d+)", result.stdout)
            slurm_id = m.group(1) if m else "unknown"

            meta = {
                "job_id":    job_id,
                "slurm_id":  slurm_id,
                "question":  cfg.get("question",  ""),
                "mode":      cfg.get("mode",       "reasoning"),
                "dataset":   cfg.get("dataset",    DEFAULT_DATASET),
                "path_type": cfg.get("pathType",   DEFAULT_PATH_TYPE),
                "n_beam":    cfg.get("beams",       DEFAULT_BEAM),
                "submitted": datetime.utcnow().isoformat(),
            }
            (jdir / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"  Submitted Slurm job {slurm_id}  (ui_job={job_id})")

            self._json({"job_id": job_id, "slurm_id": slurm_id})

        else:
            self._json({"error": "not found"}, 404)


# ── Job status from filesystem only — no squeue call needed ───────────────────
def _job_status(jdir: Path) -> str:
    out_files = sorted(jdir.glob("slurm-*.out"))
    if not out_files:
        return "pending"

    content = out_files[-1].read_text(errors="replace")

    if "UI_EVENT done" in content:
        return "done"
    if "UI_EVENT error" in content or "exited with code" in content:
        return "failed"
    if "CANCELLED" in content or "DUE TO TIME LIMIT" in content:
        return "failed"
    if "UI_EVENT step" in content or "Running KG-TRACES" in content:
        return "running"

    return "pending"


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    server = http.server.HTTPServer((args.host, args.port), BridgeHandler)
    print(f"\n  KG-TRACES Bridge")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Repo:       {REPO_ROOT}")
    print(f"  Partition:  {DEFAULT_PARTITION}  (no --gres)")
    print(f"  Conda env:  {CONDA_ENV}")
    print(f"  Model type: {DEFAULT_MODEL_TYPE}")
    print(f"  Path type:  {DEFAULT_PATH_TYPE}")
    print(f"  Jobs dir:   {JOBS_DIR}")
    print(f"\n  Tunnel from laptop:")
    print(f"  ssh -L {args.port}:localhost:{args.port} -t 018228028@coe-hpc1.sjsu.edu \"ssh -L {args.port}:localhost:{args.port} coe-hpc3\"\n")
    server.serve_forever()
