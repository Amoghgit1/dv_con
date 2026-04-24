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
    import os

    test_cases = [
        ("../test_images/kitchen.jpg",     0),   # Serve wine
        ("../test_images/kitchen.jpg",     1),   # Cut food
        ("../test_images/living_room.jpg", 2),   # Sit down
        ("../test_images/living_room.jpg", 3),   # Look at time
        ("../test_images/kitchen.jpg",     4),   # Drink something
        ("../test_images/kitchen.jpg",     5),   # Eat something
        ("../test_images/living_room.jpg", 6),   # Make a phone call
        ("../test_images/desk.jpg",        7),   # Read
        ("../test_images/bedroom.jpg",     8),   # Travel
        ("../test_images/outdoor.jpg",     9),   # Play sports
        ("../test_images/kitchen.jpg",    10),   # Cook food
        ("../test_images/living_room.jpg",11),   # Take a photo
        ("../test_images/bedroom.jpg",    12),   # Sleep
        ("../test_images/desk.jpg",       13),   # Work at desk
    ]

    for image_path, task_id in test_cases:
        if os.path.exists(image_path):
            result = run_pipeline(image_path, task_id)
            print(f"Answer: {result}\n")
        else:
            print(f"Skipping — image not found: {image_path}\n")