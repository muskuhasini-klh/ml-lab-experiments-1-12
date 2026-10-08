# ML Lab Experiments 1–12

This repository brings together the supplied Placement Predict lab code, input data, experiment outputs, and the Experiment 11–12 lab handout.

## Contents

- `src/data/`: data ingestion, exploratory analysis, placement data, and the image used by Experiment 11.
- `src/models/`: gradient descent and least-squares regression, decision tree, random forest, boosted-tree models, image segmentation, and placement clustering.
- `reports/figures/`: generated plots and result tables. `sampled_3000/` and `sampled_5000/` contain sample-based clustering and classification results.
- `docs/`: the supplied Experiment 11–12 instructions and questions.

## Setup

Use Python 3.12 or later, then install the packages:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run commands from the repository root. For example, Experiment 11:

```powershell
.\.venv\Scripts\python.exe .\src\models\kmeans_image.py
```

Experiment 12's full-data Agglomerative Clustering step can require tens of gigabytes of memory on the supplied 100,000-row dataset. The checked-in `reports/figures/sampled_3000/` results were produced with a deterministic 3,000-row sample.
