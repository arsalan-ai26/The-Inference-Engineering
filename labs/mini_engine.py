"""mini_engine.py: a small inference engine for Qwen2-family models, built step by step.

Step 1 model runner   Step 2 KV cache   Step 3 batched forward over a flat token list
Step 4 continuous batching scheduler with a token budget (chunked prefill)
Step 5 paged KV cache: block pool, block tables, reference counts
Step 6 streaming per request          Step 7 prefix caching and metrics
Runs on CPU in FP32; the structure is the same as vLLM's (chapter 8.2).
"""
import hashlib, time, collections, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- Step 1: the model, written out from its weights ----------
class Model:
    def __init__(self, name):
        hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        c = hf.config; sd = {k: v.detach() for k, v in hf.state_dict().items()}
        self.L, self.H, self.KV = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads
        self.D = c.hidden_size // self.H; self.eps = c.rms_norm_eps
        theta = (getattr(c, "rope_parameters", None) or {}).get("rope_theta", getattr(c, "rope_theta", 1e6))
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.D, 2).float() / self.D))
        g = lambda k: sd[k]
        self.emb = g("model.embed_tokens.weight"); self.norm = g("model.norm.weight")
        self.head = sd.get("lm_head.weight", self.emb)            # tied embeddings in small Qwen models
        self.layers = [{k: g(f"model.layers.{i}.{k}") for k in [
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight", "self_attn.k_proj.bias",
            "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]} for i in range(self.L)]
        self.vocab = self.emb.shape[0]; del hf

    def rms(self, x, w): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x, pos):                                        # x: [N, heads, D], rotate-half convention
        f = pos[:, None].float() * self.inv_freq[None]; cos, sin = f.cos().repeat(1, 2)[:, None], f.sin().repeat(1, 2)[:, None]
        x1, x2 = x[..., : self.D // 2], x[..., self.D // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    # ---------- Step 3 + 5: one forward pass over a flat list of tokens from many sequences ----------
    def forward(self, tokens, positions, seqs, kv, block_size):
        """tokens, positions: [N]. seqs: list of (start, length, block_table, context_len) per sequence.
        kv: [layers, 2, blocks * block_size, KV, D]. Writes new K/V into the paged cache, returns logits of each seq's last token."""
        x = self.emb[tokens]
        slots = torch.cat([torch.tensor([bt[p // block_size] * block_size + p % block_size for p in range(ctx - n, ctx)])
                           for (_, n, bt, ctx) in seqs])          # where each new token's K/V goes
        for li, w in enumerate(self.layers):
            h = self.rms(x, w["input_layernorm.weight"])
            q = (h @ w["self_attn.q_proj.weight"].T + w["self_attn.q_proj.bias"]).view(-1, self.H, self.D)
            k = (h @ w["self_attn.k_proj.weight"].T + w["self_attn.k_proj.bias"]).view(-1, self.KV, self.D)
            v = (h @ w["self_attn.v_proj.weight"].T + w["self_attn.v_proj.bias"]).view(-1, self.KV, self.D)
            q, k = self.rope(q, positions), self.rope(k, positions)
            kflat, vflat = kv[li, 0], kv[li, 1]                    # [blocks * block_size, KV, D] views
            kflat[slots] = k; vflat[slots] = v                     # paged write
            outs = []
            for (s, n, bt, ctx) in seqs:                           # paged attention, one sequence at a time
                idx = torch.cat([torch.arange(b * block_size, b * block_size + block_size) for b in bt])[:ctx]
                K = kflat[idx].repeat_interleave(self.H // self.KV, 1); V = vflat[idx].repeat_interleave(self.H // self.KV, 1)
                qs = q[s:s + n]                                    # [n, H, D]
                att = torch.einsum("nhd,chd->hnc", qs, K) / self.D ** 0.5
                qpos = torch.arange(ctx - n, ctx)[:, None]; mask = torch.arange(ctx)[None] > qpos
                att = att.masked_fill(mask[None], float("-inf")).softmax(-1)
                outs.append(torch.einsum("hnc,chd->nhd", att, V).reshape(n, -1))
            x = x + torch.cat(outs) @ w["self_attn.o_proj.weight"].T
            h = self.rms(x, w["post_attention_layernorm.weight"])
            x = x + (F.silu(h @ w["mlp.gate_proj.weight"].T) * (h @ w["mlp.up_proj.weight"].T)) @ w["mlp.down_proj.weight"].T
        last = torch.tensor([s + n - 1 for (s, n, _, _) in seqs])
        return self.rms(x[last], self.norm) @ self.head.T

# ---------- Step 5 + 7: block pool with reference counts and prefix-cache hashes ----------
class BlockPool:
    def __init__(self, n):
        self.free = collections.deque(range(n)); self.ref = [0] * n
        self.hash_of = {}; self.by_hash = {}; self.hits = self.lookups = 0
    def alloc(self):
        if not self.free: return None
        b = self.free.popleft()
        if b in self.hash_of: self.by_hash.pop(self.hash_of.pop(b), None)   # evict its cached content
        self.ref[b] = 1; return b
    def release(self, b):
        self.ref[b] -= 1
        assert self.ref[b] >= 0, "double free"
        if self.ref[b] == 0: self.free.append(b)                 # cached blocks stay findable until reused
    def lookup(self, h):
        self.lookups += 1; b = self.by_hash.get(h)
        if b is None: return None
        if self.ref[b] == 0: self.free.remove(b)                 # revive a cached free block
        self.ref[b] += 1; self.hits += 1; return b
    def register(self, b, h): self.hash_of[b] = h; self.by_hash[h] = b

def block_hash(parent, toks): return hashlib.sha256((parent + ",".join(map(str, toks))).encode()).hexdigest()

class Request:
    def __init__(self, rid, prompt_ids, max_new, temperature=0.0):
        self.rid, self.prompt, self.out = rid, list(prompt_ids), []
        self.max_new, self.temp = max_new, temperature
        self.computed = 0; self.blocks = []; self.done = False
        self.t_arrive = time.perf_counter(); self.t_first = None; self.t_done = None
    @property
    def tokens(self): return self.prompt + self.out

# ---------- Step 4: the engine loop with continuous batching and a token budget ----------
class Engine:
    def __init__(self, model, num_blocks=256, block_size=16, token_budget=256, max_running=32, prefix_caching=True, eos=None):
        self.m, self.bs, self.budget, self.max_running = model, block_size, token_budget, max_running
        self.kv = torch.zeros(model.L, 2, num_blocks * block_size, model.KV, model.D)
        self.pool = BlockPool(num_blocks); self.waiting = collections.deque(); self.running = []
        self.prefix = prefix_caching; self.eos = eos; self.steps = 0; self.preemptions = 0; self.computed_tokens = 0

    def add(self, req):
        if self.prefix:                                          # Step 7: reuse full cached blocks of the prompt
            parent = ""
            for i in range(0, len(req.prompt) // self.bs * self.bs, self.bs):
                h = block_hash(parent, req.prompt[i:i + self.bs])
                b = self.pool.lookup(h)
                if b is None: break
                req.blocks.append(b); req.computed += self.bs; parent = h
            req.computed = min(req.computed, len(req.prompt) - 1)    # always recompute at least the last prompt token
        self.waiting.append(req)

    def _ensure_blocks(self, req, upto):
        while len(req.blocks) * self.bs < upto:
            b = self.pool.alloc()
            if b is None: return False
            req.blocks.append(b)
        return True

    def _preempt(self, scheduled=()):                            # evict the newest running request not yet scheduled
        cands = [r for r in self.running if r not in scheduled]
        victim = cands[-1] if cands else self.running[-1]
        self.running.remove(victim)
        for b in victim.blocks: self.pool.release(b)
        victim.blocks, victim.computed = [], 0; self.preemptions += 1; self.waiting.appendleft(victim)

    def step(self):
        budget, batch = self.budget, []                           # batch: (req, n_new_tokens)
        for r in list(self.running):                              # running first: decodes (1 token) or unfinished prefills
            if r not in self.running or budget <= 0: continue
            n = min(len(r.tokens) - r.computed, budget)
            while not self._ensure_blocks(r, r.computed + n):
                self._preempt([b[0] for b in batch] + [r])
                if r not in self.running: break
            if r in self.running: batch.append((r, n)); budget -= n
        while self.waiting and budget > 0 and len(self.running) < self.max_running:
            r = self.waiting[0]; n = min(len(r.tokens) - r.computed, budget)   # chunked prefill
            had = len(r.blocks)
            if not self._ensure_blocks(r, r.computed + n):          # no room: give back what we took, wait
                for b in r.blocks[had:]: self.pool.release(b)
                del r.blocks[had:]; break
            self.waiting.popleft(); self.running.append(r); batch.append((r, n)); budget -= n
        if not batch: return []
        toks, pos, seqs, s = [], [], [], 0
        for r, n in batch:
            toks += r.tokens[r.computed:r.computed + n]; pos += list(range(r.computed, r.computed + n))
            seqs.append((s, n, r.blocks, r.computed + n)); s += n
        self.computed_tokens += s
        with torch.no_grad():
            logits = self.m.forward(torch.tensor(toks), torch.tensor(pos), seqs, self.kv, self.bs)
        self.steps += 1; events = []
        for (r, n), lg in zip(batch, logits):
            r.computed += n
            if r.computed < len(r.tokens): continue                 # still prefilling (chunked)
            if self.prefix: self._register_full_blocks(r)
            tok = int(lg.argmax()) if r.temp == 0 else int(torch.multinomial((lg / r.temp).softmax(-1), 1))
            r.out.append(tok); now = time.perf_counter()
            if r.t_first is None: r.t_first = now
            events.append((r.rid, tok))                             # Step 6: stream each new token
            if len(r.out) >= r.max_new or tok == self.eos:
                r.done, r.t_done = True, now; self.running.remove(r)
                for b in r.blocks: self.pool.release(b)
        return events

    def _register_full_blocks(self, r):
        parent = ""
        for i, b in enumerate(r.blocks):
            if (i + 1) * self.bs > r.computed: break
            h = block_hash(parent, r.tokens[i * self.bs:(i + 1) * self.bs])
            if b not in self.pool.hash_of and h not in self.pool.by_hash: self.pool.register(b, h)
            parent = h

    def run(self, reqs, on_token=None):
        for r in reqs: self.add(r)
        while self.waiting or self.running:
            for rid, tok in self.step():
                if on_token: on_token(rid, tok)
        return reqs
