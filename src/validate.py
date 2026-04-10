from main import run_pipeline
from encode import TASKS
import json
import os

# Ground truth using correct task IDs
GROUND_TRUTH = [
    {"image": "test.jpg", "task_id": 0, "ground_truth": "wine glass"},
    {"image": "test.jpg", "task_id": 1, "ground_truth": "knife"},
    {"image": "test.jpg", "task_id": 2, "ground_truth": "chair"},
]

def validate():
    correct = 0
    results = []

    for tc in GROUND_TRUTH:
        if not os.path.exists(tc["image"]):
            print(f"Skipping {tc['image']} — file not found")
            continue

        predicted = run_pipeline(tc["image"], tc["task_id"])
        is_correct = predicted is not None and predicted.lower() == tc["ground_truth"].lower()
        correct += int(is_correct)
        status = "PASS" if is_correct else f"FAIL (got '{predicted}')"

        print(f"  [{status}] Task='{TASKS[tc['task_id']]['name']}'  expected='{tc['ground_truth']}'")
        results.append({
            "task":    TASKS[tc["task_id"]]["name"],
            "gt":      tc["ground_truth"],
            "pred":    predicted,
            "correct": is_correct
        })

    if results:
        accuracy = correct / len(results)
        print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.1%}")
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    validate()