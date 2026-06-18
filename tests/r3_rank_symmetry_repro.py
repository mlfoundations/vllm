"""#237 — R3 rank-symmetry repro / op-counter harness (Stages 0/2/3).

Small-MoE (OLMoE-1B-7B), TP=2, enforce_eager, R3 capture ON, forced mixed
chunked-prefill+decode step. Uses the env-gated (VLLM_R3_OPCOUNT) per-rank
op-issuance counter added to routed_experts_capturer.py to detect whether
rank-0 and the non-rank-0 TP workers issue an asymmetric op sequence in the
per-step D2H epilogue (the TP-rank-desync wedge mechanism, #232 root cause).

This is NOT a debugger (ptrace blocked in the SIF, NCCL FR doesn't dump the
lagging rank). The op-counter is the observable localization/gate signal.

Modes (env R3_REPRO_MODE):
  - opcount  (default): run, then dump each rank's op-issuance log; compare.
               GATE (Stage 0): rank sequences MISMATCH (bug reproduced).
               GATE (Stage 2): rank sequences IDENTICAL every step + no hang.
  - capture: run a fixed deterministic decode trace with R3 ON and dump the
             captured routed_experts arrays per request to .npz (Stage 3
             G-CAPTURE: np.array_equal pre vs post patch).

Args via env:
  R3_MODEL       model path (default OLMoE snapshot)
  R3_TP          tensor_parallel_size (default 2)
  R3_OUT         output dir for per-rank op logs / captures
  R3_FLAG        "on"|"off" -> enable_return_routed_experts (default on)
  R3_STEPS       max generated tokens to drive decode (default 64)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np


def main() -> int:
    from vllm import LLM, SamplingParams

    model = os.environ.get(
        "R3_MODEL",
        "/e/scratch/jureap59/feuer1/hf_cache/hub/"
        "models--allenai--OLMoE-1B-7B-0924-Instruct/snapshots/"
        "7f1c97f440f06ce36705e4f2b843edb5925f4498",
    )
    tp = int(os.environ.get("R3_TP", "2"))
    out = os.environ.get("R3_OUT", "/tmp/r3_opcount")
    flag_on = os.environ.get("R3_FLAG", "on").lower() == "on"
    steps = int(os.environ.get("R3_STEPS", "64"))
    mode = os.environ.get("R3_REPRO_MODE", "opcount")
    os.makedirs(out, exist_ok=True)
    # Workers (spawned by LLM()) inherit this; capturer self-dumps op logs here.
    os.environ.setdefault("VLLM_R3_OPCOUNT_DIR", out)

    print(
        f"[repro] model={model} tp={tp} R3={'ON' if flag_on else 'OFF'} "
        f"steps={steps} mode={mode} out={out} "
        f"VLLM_R3_OPCOUNT={os.environ.get('VLLM_R3_OPCOUNT')}",
        flush=True,
    )

    # Chunked prefill with a small token budget so a long prompt prefills over
    # multiple chunks while short prompts decode -> forces the MIXED step.
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        enforce_eager=True,
        enable_return_routed_experts=flag_on,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=1234,
        # R3 capture is not validated with async scheduling on this build
        # (engine rejects the combo). Async scheduling is orthogonal to the
        # TP-rank op-issuance asymmetry we are localizing, so disable it.
        async_scheduling=False,
    )

    # Mixed batch: a couple of long prompts (prefill over many 256-token chunks)
    # co-scheduled with several short prompts that immediately start decoding.
    long_prompt = "The history of computing. " * 400  # ~> several prefill chunks
    short_prompts = [
        "Q: What is 2+2? A:",
        "The capital of France is",
        "Once upon a time,",
        "def add(a, b): return",
    ]
    prompts = short_prompts + [long_prompt, long_prompt[: len(long_prompt) // 2]]

    sp = SamplingParams(temperature=0.0, max_tokens=steps, seed=1234)
    outs = llm.generate(prompts, sp)
    print(f"[repro] generated {len(outs)} sequences", flush=True)
    for i, o in enumerate(outs):
        rex = getattr(o.outputs[0], "routed_experts", None)
        shp = None if rex is None else np.asarray(rex).shape
        print(
            f"[repro]  seq{i} out_len={len(o.outputs[0].token_ids)} "
            f"routed_experts_shape={shp}",
            flush=True,
        )

    if mode == "capture":
        # Stage 3 G-CAPTURE: dump captured arrays (rank-0 only has them).
        capdir = os.path.join(out, "captures")
        os.makedirs(capdir, exist_ok=True)
        arrs = {}
        for i, o in enumerate(outs):
            rex = getattr(o.outputs[0], "routed_experts", None)
            if rex is not None:
                arrs[f"seq{i}"] = np.asarray(rex)
        np.savez(os.path.join(capdir, "captured.npz"), **arrs)
        print(f"[repro] capture: wrote {len(arrs)} arrays to {capdir}", flush=True)

    # Per-rank op logs are SELF-DUMPED by each TP worker's capturer to
    # VLLM_R3_OPCOUNT_DIR (== R3_OUT) at each step (no serializable RPC needed,
    # which V1's collective_rpc cannot do for a raw function). Give the last
    # in-flight async dump a moment to land, then read them.
    import time

    time.sleep(2)

    # Local-side comparison summary (also re-done by the analyzer).
    logs = {}
    for r in range(tp):
        p = os.path.join(out, f"oplog_rank{r}.json")
        if os.path.exists(p):
            with open(p) as f:
                logs[r] = json.load(f)
    # The GATE compares the GPU/stream/collective-affecting op subsequence.
    # "scatter" is pure-host numpy (rank-0-only, collective-NEUTRAL): it cannot
    # reorder the main-stream collective launches, so it is excluded from the
    # symmetry gate. Every other token is a GPU/stream op that MUST be issued
    # identically on every rank (the FIX B invariant).
    HOST_ONLY = {"scatter"}

    def _coll(seq):
        return [t for t in seq if t not in HOST_ONLY]

    if len(logs) == tp and flag_on:
        seqs = {r: logs[r]["op_log"] for r in logs}
        n_steps = min(len(s) for s in seqs.values())
        mismatch_steps = []
        for s in range(n_steps):
            base = _coll(seqs[0][s])
            for r in range(1, tp):
                if _coll(seqs[r][s]) != base:
                    mismatch_steps.append((s, r, base, _coll(seqs[r][s])))
        # Also report the raw (incl. host-only) rank-0 vs rank-1 example.
        example = {r: seqs[r][0] if seqs[r] else [] for r in seqs}
        print(
            f"[repro] OPCOUNT SUMMARY: steps_logged_per_rank="
            f"{ {r: len(seqs[r]) for r in seqs} } "
            f"collective_mismatch_steps={len(mismatch_steps)} "
            f"first_3_mismatches={mismatch_steps[:3]} "
            f"step0_raw_per_rank={example}",
            flush=True,
        )
        if mismatch_steps:
            print("[repro] RESULT=MISMATCH (collective op-sequences DIVERGE)", flush=True)
        else:
            print(
                "[repro] RESULT=SYMMETRIC (collective op-sequences IDENTICAL "
                "across all ranks every step)",
                flush=True,
            )
    print("[repro] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
