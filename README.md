# Task-Aware Object Detection Framework
### DVCon India 2026 — Stage 2 Submission

> A lightweight, edge-compatible pipeline that selects the most task-relevant object in an image given a natural language task description. Built for simulation and eventual deployment on the VEGA RISC-V processor with Genesys-2 FPGA acceleration.

---

## Overview

Conventional object detectors identify *everything* in an image equally. This project goes further — given a task like **"serve wine"**, the system analyses a scene and selects the single most appropriate object (e.g. a wine glass, not a cup or bottle) by combining visual detection with semantic language understanding.

The pipeline runs fully in Python for simulation and is designed for edge deployment on the VEGA processor + Genesys-2 FPGA platform.

---

## Pipeline Architecture

```
Input Image
     │
     ▼
┌─────────────────────────┐
│   MobileNet-SSD         │  ← Object Detection (VEGA CPU)
│   (COCO pretrained)     │    Outputs: label, confidence, bbox
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│   CLIP Text Encoder     │  ← Task Encoding (VEGA CPU)
│   (clip-ViT-B-32)       │    Outputs: task embedding vector
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│   Affinity Scorer       │  ← Scoring (FPGA Accelerator)
│   Score = 0.65×task     │    Cosine similarity + domain boost
│         + 0.35×conf     │
└─────────────────────────┘
     │
     ▼
  Best matching object for the task
```

---

## Supported Tasks

14 tasks from the reference paper (arXiv:1904.03000):

| ID | Task | Preferred Objects |
|----|------|-------------------|
| 0 | Serve wine | wine glass |
| 1 | Cut food | knife |
| 2 | Sit down | chair, couch |
| 3 | Look at the time | clock |
| 4 | Drink something | cup, wine glass |
| 5 | Eat something | fork, spoon |
| 6 | Make a phone call | cell phone |
| 7 | Read | book |
| 8 | Travel | suitcase, backpack |
| 9 | Play sports | sports ball, tennis racket |
| 10 | Cook food | knife, spoon |
| 11 | Take a photo | cell phone |
| 12 | Sleep | bed |
| 13 | Work at desk | laptop, keyboard |

---

## Project Structure

```
task_aware_detection/
├── src/
│   ├── detect.py            # MobileNet-SSD object detection
│   ├── encode.py            # CLIP text encoder + embedding precomputation
│   ├── score.py             # Affinity scoring logic
│   ├── main.py              # Full pipeline entry point
│   ├── validate.py          # Accuracy evaluation
│   └── runn.py              # Runs all stages in order
├── test_images/
│   ├── kitchen.jpg
│   ├── living_room.jpg
│   ├── desk.jpg
│   ├── bedroom.jpg
│   └── outdoor.jpg
├── venv/
├── .gitignore
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.9+, Windows/Linux/Mac

```bash
# Clone the repository
git clone https://github.com/your-username/task_aware_detection.git
cd task_aware_detection

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
cd src

# Run all stages in order
python runn.py
```

Or run individual stages:

```bash
# Step 1: Precompute CLIP embeddings (run once)
python encode.py

# Step 2: Test object detection
python detect.py

# Step 3: Test affinity scoring
python score.py

# Step 4: Run full pipeline
python main.py

# Step 5: Evaluate accuracy
python validate.py
```

---

## How Scoring Works

Each detected object is scored against the task using two signals:

**1. CLIP Semantic Similarity**
The task description and object label are both encoded into CLIP embedding space. Cosine similarity measures how semantically related they are.

**2. Domain Knowledge Boost**
On top of CLIP similarity, objects explicitly listed as preferred or keywords for a task receive a score boost:

```
preferred object  → +0.25 boost
keyword object    → +0.10 boost
other objects     → no boost
```

**Final Score Formula:**
```
task_relevance = clip_similarity + domain_boost   (capped at 1.0)
final_score    = 0.65 × task_relevance + 0.35 × detection_confidence
```

The 0.65/0.35 weighting means task relevance is prioritised over raw detection confidence — a less confidently detected wine glass will still beat a highly confident cup when the task is "serve wine".

---

## Example Output

```
Image : test_images/kitchen.jpg
Task  : [0] Serve wine
----------------------------------------
Detected 5 objects. Top results:

  1. wine glass          score=0.948  conf=0.887  <-- BEST MATCH
  2. cup                 score=0.902  conf=0.743
  3. bottle              score=0.860  conf=0.612
  4. bowl                score=0.734  conf=0.531
  5. knife               score=0.623  conf=0.445
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| torch + torchvision | MobileNet-SSD inference |
| sentence-transformers | CLIP text encoding |
| opencv-python | Image loading and processing |
| numpy | Embedding computation |
| Pillow | Image preprocessing |

Install all with:
```bash
pip install -r requirements.txt
```

---

## Hardware Target

This simulation is designed for eventual deployment on:

- **Processor:** VEGA RISC-V (CDAC Trivandrum)
- **FPGA Board:** Genesys-2 (Xilinx Kintex-7)
- **Accelerated Stage:** Affinity scoring (dot product + ranking) on FPGA fabric
- **CPU Stages:** Object detection + CLIP encoding on VEGA ARM/RISC-V cores

---

## Dataset

- **COCO 2017** — 80 object categories, used for MobileNet-SSD pretraining
- **14 Task Definitions** — from arXiv:1904.03000 (Fang et al.)

---


## References

1. Fang et al. — "Learning Task-Oriented Grasping for Tool Manipulation from Simulated Self-Supervision" — arXiv:1904.03000
2. Radford et al. — "Learning Transferable Visual Models From Natural Language Supervision (CLIP)" — arXiv:2103.00020
3. Howard et al. — "Searching for MobileNetV3" — arXiv:1905.02244
4. Lin et al. — "Microsoft COCO: Common Objects in Context" — arXiv:1405.0312
