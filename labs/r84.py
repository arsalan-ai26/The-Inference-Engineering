# Reproduce Sarathi-Serve's core claim at small scale with mini_engine:
# a long prompt arriving while other requests decode stalls them, unless its prefill is chunked.
import sys, time, torch
from transformers import AutoTokenizer
from mini_engine import Model, Engine, Request
torch.set_num_threads(2)
name = "Qwen/Qwen2.5-0.5B-Instruct"; tok = AutoTokenizer.from_pretrained(name); m = Model(name)
def ids(p): return tok(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"]
short = [ids(f"Count from {i} upward, one number per line.") for i in range(8)]
long_prompt = ids("Summarize this log:\n" + "\n".join(f"line {i}: service ok, latency {i % 97} ms" for i in range(110)))

def run(budget, n_decoders=4, inject_at=6, out=24):
    e = Engine(m, num_blocks=1024, token_budget=budget, prefix_caching=False)
    reqs = [Request(i, short[i], out) for i in range(n_decoders)]
    for r in reqs: e.add(r)
    stamps = {r.rid: [] for r in reqs}; L = None; step = 0
    if not n_decoders: inject_at = 0
    while e.waiting or e.running or L is None:
        if step == inject_at:
            L = Request(99, long_prompt, 4); e.add(L); stamps[99] = []
        for rid, tok_ in e.step(): stamps[rid].append(time.perf_counter())
        step += 1
    gaps = [b - a for rid in stamps if rid != 99 for a, b in zip(stamps[rid], stamps[rid][1:])]
    gaps.sort()
    return dict(max_itl=gaps[-1] if gaps else 0, p50_itl=gaps[len(gaps) // 2] if gaps else 0, ttft_long=L.t_first - L.t_arrive, steps=e.steps)

if __name__ == "__main__":
    print("long prompt", len(long_prompt), "tokens")
    print("budget   decoders  max ITL  median ITL  TTFT of long prompt  steps")
    for budget, nd in [(4096, 4), (512, 4), (256, 4), (128, 4), (64, 4), (4096, 0), (128, 0), (128, 8), (4096, 8)]:
        r = run(budget, nd)
        print(f"{budget:6d}   {nd:8d}  {r['max_itl']:6.2f} s  {r['p50_itl']:8.2f} s  {r['ttft_long']:12.2f} s  {r['steps']:6d}" if nd else
              f"{budget:6d}   {nd:8d}       -           -  {r['ttft_long']:12.2f} s  {r['steps']:6d}")
