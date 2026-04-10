from sentence_transformers import SentenceTransformer
import numpy as np
import os

model = None

# Correct 14 tasks from arXiv:1904.03000
TASKS = {
    0:  {"name": "Serve wine",        "preferred": ["wine glass"],                   "keywords": ["cup","bottle","bowl"],                          "fallback": ["cup","mug"]},
    1:  {"name": "Cut food",          "preferred": ["knife"],                        "keywords": ["scissors","fork"],                              "fallback": ["scissors"]},
    2:  {"name": "Sit down",          "preferred": ["chair","couch"],                "keywords": ["bench","bed"],                                  "fallback": ["bench"]},
    3:  {"name": "Look at the time",  "preferred": ["clock"],                        "keywords": ["cell phone","laptop"],                          "fallback": ["cell phone"]},
    4:  {"name": "Drink something",   "preferred": ["cup","wine glass"],             "keywords": ["bottle","bowl"],                                "fallback": ["bottle"]},
    5:  {"name": "Eat something",     "preferred": ["fork","spoon"],                 "keywords": ["bowl","sandwich","pizza","banana","apple"],     "fallback": ["sandwich"]},
    6:  {"name": "Make a phone call", "preferred": ["cell phone"],                   "keywords": ["remote"],                                       "fallback": ["laptop"]},
    7:  {"name": "Read",              "preferred": ["book"],                         "keywords": ["laptop","tv","cell phone"],                     "fallback": ["laptop"]},
    8:  {"name": "Travel",            "preferred": ["suitcase","backpack"],          "keywords": ["handbag","umbrella"],                           "fallback": ["handbag"]},
    9:  {"name": "Play sports",       "preferred": ["sports ball","tennis racket"],  "keywords": ["bicycle","frisbee","skateboard"],               "fallback": ["frisbee"]},
    10: {"name": "Cook food",         "preferred": ["knife","spoon"],                "keywords": ["oven","microwave","bowl"],                      "fallback": ["oven"]},
    11: {"name": "Take a photo",      "preferred": ["cell phone"],                   "keywords": ["laptop"],                                       "fallback": ["laptop"]},
    12: {"name": "Sleep",             "preferred": ["bed"],                          "keywords": ["couch","teddy bear"],                           "fallback": ["couch"]},
    13: {"name": "Work at desk",      "preferred": ["laptop","keyboard"],            "keywords": ["mouse","book","cell phone"],                    "fallback": ["mouse"]},
}

# Boost values applied on top of CLIP cosine similarity
PREFERRED_BOOST = 0.25
KEYWORD_BOOST   = 0.10

def load_encoder():
    global model
    if model is None:
        print("Loading CLIP text encoder...")
        model = SentenceTransformer('clip-ViT-B-32')
        print("Encoder loaded.")

def encode_text(text):
    load_encoder()
    return model.encode(text, normalize_embeddings=True)

def precompute_and_save():
    """Run once to save embeddings to disk."""
    load_encoder()
    from detect import COCO_LABELS

    print("Encoding task descriptions...")
    task_embeddings = {TASKS[i]["name"]: encode_text(TASKS[i]["name"]) for i in TASKS}
    np.save("task_embeddings.npy", task_embeddings)

    print("Encoding COCO labels...")
    valid_labels = [l for l in COCO_LABELS if l not in ('N/A', '__background__')]
    label_embeddings = {label: encode_text(label) for label in valid_labels}
    np.save("label_embeddings.npy", label_embeddings)

    print("Saved task_embeddings.npy and label_embeddings.npy")

def load_embeddings():
    if not os.path.exists("task_embeddings.npy") or not os.path.exists("label_embeddings.npy"):
        print("Embeddings not found. Running precompute...")
        precompute_and_save()
    task_embs = np.load("task_embeddings.npy", allow_pickle=True).item()
    label_embs = np.load("label_embeddings.npy", allow_pickle=True).item()
    return task_embs, label_embs

def get_boost(label, task_id):
    """Returns a domain-knowledge boost for a label given a task."""
    task = TASKS[task_id]
    label_lower = label.lower()
    if label_lower in [p.lower() for p in task["preferred"]]:
        return PREFERRED_BOOST
    if label_lower in [k.lower() for k in task["keywords"]]:
        return KEYWORD_BOOST
    return 0.0

if __name__ == "__main__":
    precompute_and_save()