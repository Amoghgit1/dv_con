import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run(script):
    print(f"\n{'='*40}")
    print(f"Running {script}...")
    print('='*40)
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        print(f"\nERROR: {script} failed. Fix the error above before continuing.")
        sys.exit(1)

run("encode.py")
run("detect.py")
run("score.py")
run("main.py")
run("validate.py")