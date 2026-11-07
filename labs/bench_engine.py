import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from mini_engine import Model, Engine, Request
torch.set_num_threads(2)
name = "Qwen/Qwen2.5-0.5B-Instruct"; tok = AutoTokenizer.from_pretrained(name); m = Model(name)
hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
eos = tok.convert_tokens_to_ids("<|im_end|>")
def ids(p, sys=None):
    msgs = ([{"role": "system", "content": sys}] if sys else []) + [{"role": "user", "content": p}]
    return tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"]
topics = ["hash tables", "TCP", "photosynthesis", "black holes", "the French Revolution", "vaccines", "GPUs", "compilers",
          "inflation", "plate tectonics", "DNS", "neural networks", "the Roman Empire", "solar panels", "databases", "jazz"]
P = [ids(f"Explain {t} in three sentences.") for t in topics]
N_OUT = 48
# 2. throughput: one request at a time (HF generate) vs continuous batching (mini engine)
t = time.perf_counter()
for x in P:
    with torch.no_grad(): hf.generate(torch.tensor([x]), max_new_tokens=N_OUT, min_new_tokens=N_OUT, do_sample=False, repetition_penalty=1.0)
t_hf = time.perf_counter() - t
e = Engine(m, num_blocks=512); t = time.perf_counter(); rs = e.run([Request(i, x, N_OUT) for i, x in enumerate(P)]); t_me = time.perf_counter() - t
ttft = sorted(r.t_first - r.t_arrive for r in rs)
print(flush=True); print(f"16 requests x {N_OUT} tokens: HF one at a time {16*N_OUT/t_hf:.1f} tok/s ({t_hf:.0f} s); mini engine {16*N_OUT/t_me:.1f} tok/s ({t_me:.0f} s), "
      f"{e.steps} steps, median TTFT {ttft[8]:.1f} s")
# 3. prefix caching: 8 requests that share a long system prompt, after one warm-up request
SYS = "You are the support assistant for Acme Bank. " + " ".join(f"Policy {i}: customers may request statement copies, card replacements and limit changes through the app or a branch." for i in range(12))
Q = [ids(f"Question {i}: how do I order a new card?", SYS) for i in range(9)]
for pc in [False, True]:
    e = Engine(m, num_blocks=512, prefix_caching=pc)
    e.run([Request(0, Q[0], 8)])                                  # warm-up request fills the cache
    c0 = e.computed_tokens; t = time.perf_counter()
    rs = e.run([Request(i, Q[i], 8) for i in range(1, 9)])
    dt = time.perf_counter() - t
    print(f"prefix caching {'on ' if pc else 'off'}: prompt {len(Q[1])} tokens, tokens computed for 8 requests {e.computed_tokens - c0}, "
          f"time {dt:.1f} s, mean TTFT {sum(r.t_first - r.t_arrive for r in rs)/8:.2f} s, block hits {e.pool.hits}", flush=True)
