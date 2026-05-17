from main import run_pipeline
from encode import TASKS
import json
import os

# Ground truth using correct task IDs
GROUND_TRUTH = [
    {"image": "../test_images/kitchen2.jpg",    "task_id": 0,  "ground_truth": "wine glass"},
    {"image": "../test_images/knife2.png",       "task_id": 1,  "ground_truth": "knife"},
    {"image": "../test_images/living_room2.jpg", "task_id": 2, "ground_truth": "couch"},
    {"image": "../test_images/clock.jpg",        "task_id": 3,  "ground_truth": "clock"},
    {"image": "../test_images/kitchen2.jpg",     "task_id": 4,  "ground_truth": "wine glass"},
    {"image": "../test_images/kitchen.jpg",      "task_id": 5,  "ground_truth": "fork"},
    {"image": "../test_images/phone.jpg",        "task_id": 6,  "ground_truth": "cell phone"},#cellphone
    {"image": "../test_images/book6.jpg",         "task_id": 7,  "ground_truth": "book"},
    {"image": "../test_images/travel.jpg",       "task_id": 8,  "ground_truth": "suitcase"},
    {"image": "../test_images/sports2.jpg",      "task_id": 9,  "ground_truth": "sports ball"},
    {"image": "../test_images/kitchen.jpg",      "task_id": 10, "ground_truth": "fork"},
    {"image": "../test_images/phone.jpg",        "task_id": 11, "ground_truth": "cell phone"},
    {"image": "../test_images/bedroom.jpg",      "task_id": 12, "ground_truth": "bed"},
    {"image": "../test_images/desk2.jpg",        "task_id": 13, "ground_truth": "laptop"},
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