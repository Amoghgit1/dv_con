import numpy as np
from encode import get_boost, TASKS

ALPHA = 0.65   # task relevance weight (matches uploaded file)
BETA  = 0.35   # detection confidence weight (matches uploaded file)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def compute_scores(detections, task_id, task_embeddings, label_embeddings):
    """
    detections      : list of dicts from detect.py
    task_id         : int 0-13
    task_embeddings : dict {task_name -> embedding}
    label_embeddings: dict {label -> embedding}

    Returns detections sorted by final score, highest first.
    """
    task_name = TASKS[task_id]["name"]

    if task_name not in task_embeddings:
        raise ValueError(f"Task '{task_name}' not found in embeddings.")

    task_emb = task_embeddings[task_name]
    scored = []

    for det in detections:
        label = det['label']
        if label not in label_embeddings:
            continue

        label_emb = label_embeddings[label]

        # CLIP cosine similarity (normalized to 0-1)
        raw_similarity = cosine_similarity(task_emb, label_emb)
        clip_score = (raw_similarity + 1) / 2

        # Domain knowledge boost from preferred/keywords
        boost = get_boost(label, task_id)

        # Combined task relevance
        task_relevance = min(clip_score + boost, 1.0)

        # Final score
        final_score = ALPHA * task_relevance + BETA * det['confidence']

        # Is this object relevant at all?
        relevant = task_relevance > 0.45

        scored.append({
            **det,
            'task_relevance': round(task_relevance, 3),
            'clip_score':     round(clip_score, 3),
            'boost':          round(boost, 3),
            'final_score':    round(final_score, 3),
            'relevant':       relevant
        })

    scored.sort(key=lambda x: x['final_score'], reverse=True)
    return scored

if __name__ == "__main__":
    from encode import load_embeddings
    import json

    dummy_detections = [
        {'label': 'wine glass', 'confidence': 0.85, 'box': []},
        {'label': 'cup',        'confidence': 0.72, 'box': []},
        {'label': 'bottle',     'confidence': 0.60, 'box': []},
        {'label': 'chair',      'confidence': 0.90, 'box': []},
        {'label': 'knife',      'confidence': 0.75, 'box': []},
    ]

    task_embs, label_embs = load_embeddings()

    for task_id in [0, 1, 2]:  # Serve wine, Cut food, Sit down
        print(f"\nTask: {TASKS[task_id]['name']}")
        results = compute_scores(dummy_detections, task_id, task_embs, label_embs)
        for r in results:
            tag = "<-- BEST" if r is results[0] else ""
            print(f"  {r['label']:<20} clip={r['clip_score']} boost={r['boost']} final={r['final_score']} {tag}")