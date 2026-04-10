import subprocess
import sys

def run(script):
    print(f"\n{'='*40}")
    print(f"Running {script}...")
    print('='*40)
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        print(f"\nERROR: {script} failed. Fix the error above before continuing.")
        sys.exit(1)

run("encode.py")    # Step 1: download CLIP, precompute embeddings
run("detect.py")    # Step 2: test MobileNet-SSD
run("score.py")     # Step 3: test affinity scoring
run("main.py")      # Step 4: full pipeline
run("validate.py")  # Step 5: accuracy check