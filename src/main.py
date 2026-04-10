from detect import detect_objects
from encode import load_embeddings, TASKS
from score import compute_scores

def run_pipeline(image_path, task_id):
    print(f"\nImage : {image_path}")
    print(f"Task  : [{task_id}] {TASKS[task_id]['name']}")
    print("-" * 40)

    detections = detect_objects(image_path)
    if not detections:
        print("No objects detected.")
        return None

    task_embs, label_embs = load_embeddings()
    ranked = compute_scores(detections, task_id, task_embs, label_embs)

    print(f"Detected {len(detections)} objects. Top results:\n")
    for i, r in enumerate(ranked[:5]):
        marker = " <-- BEST MATCH" if i == 0 else ""
        print(f"  {i+1}. {r['label']:<20} score={r['final_score']}  conf={r['confidence']}{marker}")

    return ranked[0]['label'] if ranked else None

if __name__ == "__main__":
    # Test all 14 tasks
    for task_id in range(14):
        result = run_pipeline("test.jpg", task_id)
        print(f"Answer: {result}\n")