import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from mini_engine import Model, Engine, Request
torch.set_num_threads(2)
name = "Qwen/Qwen2.5-0.5B-Instruct"; tok = AutoTokenizer.from_pretrained(name); m = Model(name)
hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
def ids(p): return tok(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"]
P = [ids(f"Explain {t} in three sentences.") for t in ["hash tables", "TCP", "photosynthesis"]]
N = 48; ref = {}
for i in range(3):
    with torch.no_grad(): ref[i] = hf.generate(torch.tensor([P[i]]), max_new_tokens=N, min_new_tokens=N, do_sample=False, repetition_penalty=1.0)[0, len(P[i]):].tolist()
for label, kw in [("budget 512", dict(token_budget=512)), ("chunked prefill, budget 16", dict(token_budget=16)), ("tiny KV pool (12 blocks)", dict(num_blocks=12))]:
    t = time.perf_counter(); e = Engine(m, **kw); rs = e.run([Request(i, P[i], N) for i in range(3)])
    print(f"{label:28s} outputs match HF: {all(r.out == ref[r.rid] for r in rs)}  steps {e.steps}  preemptions {e.preemptions}  ({time.perf_counter()-t:.0f} s)", flush=True)
