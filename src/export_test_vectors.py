# export_test_vectors.py
import numpy as np

EMB_DIM = 8

np.random.seed(42)

def normalize(v):
    return v / np.linalg.norm(v)

def to_q2_6(v):
    return np.clip(np.round(v * 64), -128, 127).astype(np.int8)

task_emb  = to_q2_6(normalize(np.random.randn(EMB_DIM)))
obj_emb_0 = to_q2_6(normalize(np.random.randn(EMB_DIM)))
obj_emb_1 = to_q2_6(normalize(np.random.randn(EMB_DIM)))

def write_hex(filename, vec):
    with open(filename, 'w') as f:
        for v in vec:
            f.write(f"{int(v) & 0xFF:02x}\n")

write_hex("task_emb.hex",  task_emb)
write_hex("obj_emb_0.hex", obj_emb_0)
write_hex("obj_emb_1.hex", obj_emb_1)

print("task_emb :", list(task_emb))
print("obj_emb_0:", list(obj_emb_0))
print("obj_emb_1:", list(obj_emb_1))

# Compute golden reference
dot0 = int(np.sum(task_emb.astype(np.int32) * obj_emb_0.astype(np.int32)))
dot1 = int(np.sum(task_emb.astype(np.int32) * obj_emb_1.astype(np.int32)))

def score(dot, conf, boost, penalty):
    clip  = (dot + 16384) >> 1
    rel   = clip + boost
    if penalty: rel = rel >> 1
    rel   = max(0, min(16384, rel))
    return (663 * rel + 358 * conf) >> 10

conf0  = int(0.80 * 16384)
boost0 = int(0.25 * 16384)
conf1  = int(0.72 * 16384)
boost1 = int(0.10 * 16384)

fs0 = score(dot0, conf0, boost0, 0)
fs1 = score(dot1, conf1, boost1, 0)

winner = 0 if fs0 >= fs1 else 1
print(f"\nExpected best_object = {winner}")
print(f"Expected best_score  = {max(fs0, fs1)}")