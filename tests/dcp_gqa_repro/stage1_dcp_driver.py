"""Stage-1 DCP root-cause driver (standalone, lives in the vLLM fork).

Builds a dcp=1 reference engine and a dcp=2 engine on Qwen2.5-1.5B-Instruct
(heads=12, kv=2, q_per_kv=6 -> dcp=2 VALID at tp=4) on the SAME prompts/seed and
drives greedy decode. Its job is NOT to assert parity (that is the SkyRL Stage-3
gate) but to EXERCISE the real `_forward_with_dcp` path so the env-gated
SKYRL_DCP_DEBUG instrumentation dumps the construction-step tensors + the fp32
REF combine check. It still prints the greedy token-id / logprob divergence so we
can correlate the dump with WHERE the rollout diverges (decode-side, grows-with-S).

Run (4-GPU single node):
    SKYRL_DCP_DEBUG=1 SKYRL_DCP_DEBUG_REF=1 \
    apptainer exec --nv <sif> python tests/dcp_gqa_repro/stage1_dcp_driver.py
"""

import os
import sys

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")

MODEL_NAME = os.environ.get("DCP_PARITY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TP = int(os.environ.get("DCP_PARITY_TP", "4"))
DCP = int(os.environ.get("DCP_PARITY_DCP", "2"))
MAX_TOKENS = int(os.environ.get("DCP_PARITY_MAX_TOKENS", "64"))
SEED = 1234
LOGPROB_ATOL = 1e-2

PROMPT_TEXTS = [
    "The capital of France is",
    "In a galaxy far, far away, there lived a",
    "To compute the factorial of a number in Python, you can write",
    "The three primary colors are red, blue, and",
]


def _extract(outputs):
    response_ids, response_logprobs = [], []
    for output in outputs:
        resp = output.outputs[0]
        response_ids.append(list(resp.token_ids))
        lp = None
        if resp.logprobs:
            lp = []
            for i, token_logprobs in enumerate(resp.logprobs):
                tid = resp.token_ids[i]
                lp.append(token_logprobs[tid].logprob)
        response_logprobs.append(lp)
    return response_ids, response_logprobs


def _build(dcp):
    import vllm

    kwargs = dict(
        model=MODEL_NAME,
        tensor_parallel_size=TP,
        enforce_eager=True,
        seed=SEED,
        dtype=os.environ.get("DCP_PARITY_DTYPE", "bfloat16"),
        gpu_memory_utilization=0.45,
        max_model_len=2048,
        disable_log_stats=True,
        enable_prefix_caching=False,
    )
    if dcp > 1:
        kwargs["decode_context_parallel_size"] = dcp
    return vllm.LLM(**kwargs)


def _run(llm, prompt_token_ids):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, logprobs=0, seed=SEED
    )
    outputs = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids],
        sampling_params=sp,
    )
    outputs = sorted(outputs, key=lambda o: int(o.request_id))
    return _extract(outputs)


def main():
    import subprocess

    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        ngpu = len([ln for ln in out.splitlines() if ln.strip().startswith("GPU ")])
    except Exception:
        ngpu = 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        ngpu = len([x for x in cvd.split(",") if x.strip()])
    assert ngpu >= TP, f"need >= {TP} GPUs; got {ngpu}"

    import vllm
    from transformers import AutoTokenizer

    print(
        f"[Stage1-DCP] vllm={vllm.__version__} model={MODEL_NAME} tp={TP} dcp={DCP} "
        f"max_tokens={MAX_TOKENS} seed={SEED} gpus={ngpu} "
        f"SKYRL_DCP_DEBUG={os.environ.get('SKYRL_DCP_DEBUG')} "
        f"REF={os.environ.get('SKYRL_DCP_DEBUG_REF')}",
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    pti = [tok(t, add_special_tokens=True)["input_ids"] for t in PROMPT_TEXTS]

    # dcp=1 reference first (no DCP path; for the divergence reference only).
    print("\n[Stage1-DCP] === building dcp=1 reference ===", flush=True)
    a = _build(1)
    ids_a, lp_a = _run(a, pti)
    del a
    import gc
    import time

    gc.collect()
    time.sleep(5)

    # dcp=2 — THIS run produces the SKYRL_DCP_DEBUG dumps from _forward_with_dcp.
    print(f"\n[Stage1-DCP] === building dcp={DCP} (instrumented path) ===", flush=True)
    b = _build(DCP)
    ids_b, lp_b = _run(b, pti)
    del b

    n = len(pti)
    first_div = None
    max_dlp = 0.0
    for i in range(n):
        m = min(len(ids_a[i]), len(ids_b[i]))
        for j in range(m):
            if ids_a[i][j] != ids_b[i][j]:
                if first_div is None:
                    first_div = (i, j, ids_a[i][j], ids_b[i][j])
                break
            if lp_a[i] and lp_b[i]:
                max_dlp = max(max_dlp, abs(lp_a[i][j] - lp_b[i][j]))
    print(
        f"\n[Stage1-DCP] divergence: first_div={first_div} "
        f"max|Δlogprob(pre-div)|={max_dlp:.4e} (atol={LOGPROB_ATOL})",
        flush=True,
    )
    print(
        "[Stage1-DCP] (first_div pos correlates the SKYRL_DCP_DEBUG dump with "
        "the decode step where the combine drift flips the greedy argmax)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
