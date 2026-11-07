# Randomized property test for the block pool: no double allocation, no double free, no stale cache hits.
import random
from mini_engine import BlockPool
random.seed(0)
for trial in range(2000):
    N = random.randint(1, 16); p = BlockPool(N); held = []; content = {}
    for _ in range(200):
        op = random.random()
        if op < 0.4:
            b = p.alloc()
            if b is not None: assert b not in held, "double allocation"; held.append(b); content[b] = None
        elif op < 0.7 and held:
            b = held.pop(random.randrange(len(held))); p.release(b)
        elif op < 0.85 and held:
            b = random.choice(held); h = f"h{random.randint(0, 5)}"
            if h not in p.by_hash and b not in p.hash_of: p.register(b, h); content[b] = h
        else:
            h = f"h{random.randint(0, 5)}"; b = p.lookup(h)
            if b is not None: assert p.hash_of.get(b) == h, "stale cache hit"; held.append(b)
        assert len(set(p.free)) == len(p.free), "block twice in free list"
        assert all(p.ref[b] == 0 for b in p.free), "free block still referenced"
        assert all(p.ref[b] == held.count(b) for b in range(N)), "reference count mismatch"
print("2,000 random trials x 200 operations: all invariants held")
