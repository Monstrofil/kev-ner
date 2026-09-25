"""Run every experiment in parallel on rented Modal GPUs; reports land in ``results/<run>/``.

Prep (host):  fixtures, see the README (run 21 needs ``python -m scripts.pull 21 --ground <store-url>``)
Run:          modal run modal_app.py::main                 # every run in RUNS
              modal run modal_app.py::main --only ner-     # runs whose name starts with it
"""

import json
import os
import shlex
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
FIXTURE = "fixtures/run-21.json"
REVIEW = "adjudications/run-21.json"
RULES = "archive/rules/run-21.json"
NER = "fixtures/uner-en.json"
ZS = "fixtures/zs.json"   # ``python -m scripts.zs_data data/zs fixtures/zs.json``
LONG = "fixtures/uner-long.json"   # ``python -m kev.long_docs fixtures/uner-en.json fixtures/uner-long.json``
Q3B = ["-m", "kev.ner", NER, "--base", "Qwen/Qwen3-4B-Base", "--bidir-doc", "--grad-ckpt", "--cpu-latency", "20"]
COMMON = ["--languages", "uk", "--dev", "tsrada.gov.ua", "uzmr.gov.ua", "--cpu-latency", "10", "--review", REVIEW]
RUNS = {
    "gliner-ft": ["-m", "baselines.gliner_span", FIXTURE, *COMMON, "--steps", "1500", "--batch", "8"],
    "kev-0.5b": ["-m", "kev.span", FIXTURE, *COMMON, "--epochs", "12", "--batch", "8"],
    "kev-1.5b": ["-m", "kev.span", FIXTURE, *COMMON, "--base", "Qwen/Qwen2.5-1.5B", "--epochs", "12", "--batch", "4"],
    # Long documents (512-4096 tokens): kev reads each in one pass, RoBERTa in 512-token windows.
    "long-kev-keep24": ["-m", "scripts.long_eval", LONG, "/out/ner-q3b-keep24"],
    "ner-q3b-keep24-longtrain": [*Q3B, "--keep-layers", "24", "--long-train", "4096"],
    "long-kev-keep24-longtrain": ["-m", "scripts.long_eval", LONG, "/out/ner-q3b-keep24-longtrain"],   # after the one above
    "long-roberta-large": ["-m", "baselines.encoder_ner", NER, "--long", LONG],
    # Zero-/few-shot: Pile-NER open types, then CrossNER + MIT types never trained on.
    "zs-q3b-keep24": ["-m", "kev.zero_shot", ZS, "--base", "Qwen/Qwen3-4B-Base", "--keep-layers", "24", "--bidir-doc",
                      "--grad-ckpt"],
    # Archived: date/number fields assembled from user-declared parts (archive/rules/run-21.json).
    "compose-1.5b": ["-m", "archive.compose", FIXTURE, RULES, "--languages", "uk",
                     "--dev", "tsrada.gov.ua", "uzmr.gov.ua", "--review", REVIEW, "--grad-ckpt"],
    # Universal NER English: flat PER/ORG/LOC, many entities per type (``python -m kev.uner data/uner fixtures/uner-en.json``).
    "ner-kev-0.5b": ["-m", "kev.ner", NER, "--cpu-latency", "50"],
    "ner-kev-0.5b-bidir": ["-m", "kev.ner", NER, "--bidir-doc", "--cpu-latency", "50"],
    "ner-kev-1.5b": ["-m", "kev.ner", NER, "--base", "Qwen/Qwen2.5-1.5B", "--cpu-latency", "50"],
    "ner-kev-1.5b-bidir": ["-m", "kev.ner", NER, "--base", "Qwen/Qwen2.5-1.5B", "--bidir-doc", "--cpu-latency", "50"],
    "ner-kev-q3-4b": ["-m", "kev.ner", NER, "--base", "Qwen/Qwen3-4B-Base", "--grad-ckpt", "--cpu-latency", "20"],
    "ner-kev-q3-4b-bidir": Q3B,
    "ner-roberta-large": ["-m", "baselines.encoder_ner", NER, "--cpu-latency", "50"],
    # Hacks on the best NER run (Qwen3-4B bidir, 36 layers): keep only the first N layers; few-shot descriptions.
    "ner-q3b-keep30": [*Q3B, "--keep-layers", "30"],
    "ner-q3b-keep24": [*Q3B, "--keep-layers", "24"],
    "ner-q3b-keep18": [*Q3B, "--keep-layers", "18"],
    "ner-q3b-keep12": [*Q3B, "--keep-layers", "12"],
    "ner-q3b-ex8": [*Q3B, "--examples", "8"],
    # 500 sentences: batch 8 x 10 epochs (~630 steps), so the small set is trained, not starved.
    "ner-q3b-small": [*Q3B, "--train-limit", "500", "--batch", "8", "--epochs", "10"],
    "ner-q3b-small-ex8": [*Q3B, "--train-limit", "500", "--batch", "8", "--epochs", "10", "--examples", "8"],
}

CODE_IGNORE = ["**/__pycache__"]
image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime", add_python=None)
    .pip_install(
        "transformers==4.51.3", "peft==0.15.2", "accelerate==1.6.0", "sentencepiece==0.2.0",
        "gliner==0.2.29", "rapidfuzz", "dateparser", "roman", "pydantic", "scikit-learn", "numpy<2",
    )
    .add_local_dir(HERE / "kev", "/work/kev", ignore=CODE_IGNORE)
    .add_local_dir(HERE / "baselines", "/work/baselines", ignore=CODE_IGNORE)
    .add_local_dir(HERE / "scripts", "/work/scripts", ignore=CODE_IGNORE)
    .add_local_dir(HERE / "archive", "/work/archive", ignore=CODE_IGNORE)
    .add_local_file(HERE / NER, f"/work/{NER}")
    .add_local_file(HERE / ZS, f"/work/{ZS}")
    .add_local_file(HERE / LONG, f"/work/{LONG}")
    .add_local_file(HERE / FIXTURE, f"/work/{FIXTURE}")
    .add_local_file(HERE / REVIEW, f"/work/{REVIEW}")
)

app = modal.App("kev-ner", image=image)
hf_cache = modal.Volume.from_name("kev-ner-hf-cache", create_if_missing=True)
# Trained weights + reports persist here: `modal volume get kev-ner <run>/ results/`.
outputs = modal.Volume.from_name("kev-ner", create_if_missing=True)
# GPU is fixed on the decorator at import time; override per invocation via BAKEOFF_GPU.
GPU = os.environ.get("BAKEOFF_GPU", "H100")


@app.function(gpu=GPU, cpu=8, memory=32768, volumes={"/hf": hf_cache, "/out": outputs}, timeout=6 * 3600)
def experiment(name: str, argv: list[str]) -> dict:
    subprocess.run(["nvidia-smi", "-L"], check=False)   # logs the GPU actually provisioned
    env = {**os.environ, "HF_HOME": "/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false",
           "OMP_NUM_THREADS": "8",
           "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    out = f"/out/{name}"
    print(f"### {name}: {' '.join(argv)}  (cpu-latency on {os.cpu_count()} visible cores, 8 reserved)", flush=True)
    subprocess.run(["python", *argv, "--out", out], cwd="/work", env=env, check=True)
    hf_cache.commit()
    outputs.commit()
    return json.loads(Path(out, "report.json").read_text())


@app.function(gpu=GPU, cpu=8, memory=32768, volumes={"/hf": hf_cache, "/out": outputs}, timeout=1800)
def run_tool(argv: list[str]) -> str:
    env = {**os.environ, "HF_HOME": "/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUTF8": "1"}
    return subprocess.run(["python", *argv], cwd="/work", env=env, check=True, capture_output=True, text=True).stdout


@app.local_entrypoint()
def tool(cmd: str, save: str):
    """One-off script against the trained runs on the volume (mounted at /out), stdout saved locally:
    modal run modal_app.py::tool --cmd "-m scripts.show_io /out/ner-kev-q3-4b-bidir ..." --save results/ner-io.json"""
    Path(save).write_text(run_tool.remote(shlex.split(cmd)), encoding="utf-8")


def summary(rep: dict) -> str:
    """One line per report, by what produced it."""
    if rep["contender"] == "kev-ner-long":
        return " ".join(f"{s} F1 {r['micro']['f1']:.3f}" for s, r in rep["results"].items())
    if rep["contender"] == "kev-zs":
        return "average F1 " + ", ".join(f"{k} {v:.3f}" for k, v in rep["average_f1"].items())
    if rep["contender"] == "kev-compose":
        return "; ".join(f"{s}: " + ", ".join(f"{n} fixed {r['fixed']}/{r['n']} composed {r['composed']}/{r['n']}"
                                              for n, r in rows.items()) for s, rows in rep["comparison"].items())
    if rep["contender"] in ("kev-ner", "encoder-bio"):
        line = (" ".join(f"{s} F1 {r['micro']['f1']:.4f}" for s, r in rep["results"].items())
                + f" | {rep['gpu_sent_per_sec_b64']:.0f} sent/s b64, {rep['gpu_sent_per_sec_b1']:.0f} b1")
        if rep["contender"] == "encoder-bio":   # it also scores the long-document fixture, in windows
            line += "".join(f" | {s} {r['micro']['f1']:.3f}" for s, r in rep["long_results"].items())
        return line
    adj = rep["adjudicated"]
    return (f"mean-field {rep['span_model']['mean_field']:.3f} (baseline {rep['baseline']['mean_field']:.3f}); "
            f"adjudicated {adj['span_model']['mean_field']:.3f} (baseline {adj['baseline']['mean_field']:.3f})"
            f" unit-exact {rep['span_model']['unit_exact']:.3f} ece {rep['ece']:.3f}"
            f" gpu {rep['gpu_ms_per_unit']:.0f}ms cpu {rep['cpu_ms_per_unit']:.0f}ms")


@app.local_entrypoint()
def main(only: str = ""):
    names = [n for n in RUNS if n.startswith(only)] if only else list(RUNS)
    for name, rep in zip(names, experiment.starmap([(n, RUNS[n]) for n in names], return_exceptions=True)):
        if isinstance(rep, Exception):
            print(f"{name}: FAILED {rep!r}")
            continue
        path = HERE / "results" / name / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        print(f"{name}: {summary(rep)} → {path}")
