# AMR Lab 3: Place Recognition with LiDAR

This project implements a global descriptor-based place recognition system for LiDAR point clouds using the NCLT dataset. It enables a robot to recognize previously visited locations (Loop Closure Detection) by comparing structural "fingerprints" of the environment.

## 🚀 Overview
The system transforms raw 3D LiDAR scans into a **Global Polar Descriptor** (Scan Context) that captures the vertical distribution of the environment. This allows for rotation-invariant matching, crucial for identifying a location even when approached from different headings.



## 🛠️ Tasks & Implementation

### 1. Descriptor Implementation
* **Method**: Implemented a Polar Height Map (Scan Context style).
* **Logic**: Bins point clouds into a 20x60 grid (radial and angular).
* **Feature**: Extracts the **Maximum Height** in each bin to represent structural landmarks like buildings and pillars.

### 2. Matching Algorithm
* **Similarity Measure**: Employs **Cosine Similarity** to compare descriptors.
* **Rotational Invariance**: Implemented a **Circular Shift** search. The algorithm "slides" one descriptor horizontally against the other to find the best alignment, ensuring the system is heading-agnostic.
* **Retrieval**: A multi-session search mechanism that can query a scan from one day and find its match in a database containing multiple sessions (e.g., matching a query from the 22nd against the 8th and 15th).

### 3. Evaluation
* **Ground Truth**: Matches are defined by a spatial threshold (< 5.0m) and temporal separation (> 60s) using NCLT ground truth CSVs.
* **Metrics**: Comprehensive evaluation using Precision-Recall (PR) curves, ROC curves, and F1-score distribution.



## 📊 Results Summary
* **Successful Loop Closure**: Identified a revisit at **0.04m** distance with a similarity score of **0.4742**.
* **Discriminative Power**: Successfully distinguished between different locations, with scores dropping significantly for negative matches (e.g., **0.3477** for locations 195m+ apart).
* **Performance Insights**: The PR and ROC curves demonstrate that while "perceptual aliasing" exists in repetitive campus corridors, the system maintains high precision for distinctive structural landmarks.

## 📂 Project Structure
```text
├── data/
│   └── raw/                # NCLT Sessions (2012-01-08, 2012-01-15, 2012-01-22)
├── notebooks/              # Development and Visualization notebooks
├── src/
│   ├── descriptor.py       # ScanContext generation logic
│   ├── matching.py         # Similarity and circular shift implementation
│   └── evaluation.py       # Metrics calculation and plotting
└── requirements.txt        # Project dependencies
