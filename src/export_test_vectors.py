# export_test_vectors.py
import numpy as np

# ==========================================================
# Configuration
# ==========================================================
EMB_DIM   = 8
N_OBJECTS = 8
SEED      = 13

np.random.seed(SEED)

print("=" * 60)
print("DVCon India 2026 - Test Vector Generator")
print("=" * 60)
print(f"Seed        : {SEED}")
print(f"Embedding   : {EMB_DIM}")
print(f"Objects     : {N_OBJECTS}")
print("=" * 60)


# ==========================================================
# Helper Functions
# ==========================================================
def normalize(v):
    return v / np.linalg.norm(v)


def to_q2_6(v):
    return np.clip(np.round(v * 64), -128, 127).astype(np.int8)


def write_hex(filename, vec):
    with open(filename, "w") as f:
        for x in vec:
            f.write(f"{int(x) & 0xFF:02x}\n")


def score(dot, conf, boost, penalty):
    clip = (dot + 16384) >> 1

    rel = clip + boost

    if penalty:
        rel >>= 1

    rel = max(0, min(16384, rel))

    return (663 * rel + 358 * conf) >> 10


# ==========================================================
# Generate Embeddings
# ==========================================================
task_emb = to_q2_6(normalize(np.random.randn(EMB_DIM)))

obj_embs = [
    to_q2_6(normalize(np.random.randn(EMB_DIM)))
    for _ in range(N_OBJECTS)
]


# ==========================================================
# Write HEX Files
# ==========================================================
write_hex("task_emb.hex", task_emb)
def write_hex32(filename, vec):
    with open(filename, "w") as f:
        for x in vec:
            f.write(f"{int(x) & 0xFFFFFFFF:08X}\n")

def write_golden_hex(filename, winner2, score2, winner8, score8):
    with open(filename, "w") as f:
        f.write(f"{winner2 & 0xFFFFFFFF:08X}\n")
        f.write(f"{score2  & 0xFFFFFFFF:08X}\n")
        f.write(f"{winner8 & 0xFFFFFFFF:08X}\n")
        f.write(f"{score8  & 0xFFFFFFFF:08X}\n")
for i, emb in enumerate(obj_embs):
    write_hex(f"obj_emb_{i}.hex", emb)


# ==========================================================
# Metadata
# (You can modify these if required.)
# ==========================================================
conf = [
    int(0.80 * 16384),
    int(0.72 * 16384),
    int(0.76 * 16384),
    int(0.61 * 16384),
    int(0.68 * 16384),
    int(0.91 * 16384),
    int(0.58 * 16384),
    int(0.74 * 16384),
]

boost = [
    int(0.25 * 16384),
    int(0.10 * 16384),
    int(0.18 * 16384),
    int(0.05 * 16384),
    int(0.12 * 16384),
    int(0.30 * 16384),
    int(0.08 * 16384),
    int(0.15 * 16384),
]

penalty = [0] * N_OBJECTS
# ==========================================================
# Write Metadata HEX Files
# ==========================================================

write_hex32("conf.hex", conf)
write_hex32("boost.hex", boost)
write_hex32("penalty.hex", penalty)

# ==========================================================
# Print Embeddings
# ==========================================================
print("\nTask Embedding")
print(list(task_emb))

print("\nObject Embeddings")

for i, emb in enumerate(obj_embs):
    print(f"obj_emb_{i}: {list(emb)}")


# ==========================================================
# Golden Reference
# ==========================================================
scores = []

print("\n" + "=" * 60)
print("Golden Reference")
print("=" * 60)

for i in range(N_OBJECTS):

    dot = int(
        np.sum(
            task_emb.astype(np.int32)
            * obj_embs[i].astype(np.int32)
        )
    )

    fs = score(
        dot,
        conf[i],
        boost[i],
        penalty[i]
    )

    scores.append(fs)

    print(
        f"Obj {i:2d} | "
        f"Dot = {dot:6d} | "
        f"Conf = {conf[i]:5d} | "
        f"Boost = {boost[i]:5d} | "
        f"Score = {fs:5d}"
    )

# ==========================================================
# Golden References
# ==========================================================

# 2-object test
scores2 = scores[:2]
winner2 = int(np.argmax(scores2))
score2 = scores2[winner2]

# Full test (N_OBJECTS)
winner8 = int(np.argmax(scores))
score8 = scores[winner8]

# Generate Golden Reference HEX File
write_golden_hex(
    "golden_reference.hex",
    winner2,
    score2,
    winner8,
    score8
)
print("\n" + "=" * 60)
print(f"Expected best_object (2 Obj) = {winner2}")
print(f"Expected best_score  (2 Obj) = {score2}")
print()

print(f"Expected best_object ({N_OBJECTS} Obj) = {winner8}")
print(f"Expected best_score  ({N_OBJECTS} Obj) = {score8}")
print("=" * 60)
