# Benchmarking Image-to-3D Reconstruction Models

## A Comparative Study of TripoSR and Shap-E in a GPU-Accelerated FastAPI Architecture

This repository provides a benchmarking framework for evaluating **Image-to-3D reconstruction models**, focusing on a comparative study between **TripoSR** and **Shap-E**.

The system is deployed using a **GPU-accelerated FastAPI architecture**, enabling efficient inference and automated benchmarking of generated 3D meshes.

The benchmarking pipeline evaluates reconstruction quality, mesh complexity, and model performance using multiple geometric metrics and inference time measurements.

---

# Project Overview

Recent advances in generative AI have enabled systems capable of converting images into detailed 3D models. These models are useful for many real-world applications including:

* Augmented Reality (AR)
* Virtual Reality (VR)
* Game development
* Robotics perception
* Digital asset generation
* Metaverse environments

However, comparing different reconstruction models requires quantitative evaluation metrics.

This project implements a **benchmarking framework** that compares **TripoSR** and **Shap-E** outputs using:

* Mesh statistics
* Chamfer Distance
* File size analysis
* GPU inference latency

---

# Repository Structure

```
benchmark-image-to-3d/

├── triposr_outputs/
│   └── *.glb
│
├── shape_outputs/
│   └── *.obj
│
├── benchmark.py
│
├── benchmark_results.csv
│
├── plots/
│   ├── vertices_comparison.png
│   ├── faces_comparison.png
│   ├── surface_area_comparison.png
│   ├── file_size_comparison.png
│   └── chamfer_distance.png
│
└── README.md
```

---

# Evaluation Metrics

The benchmarking system evaluates several geometric and computational metrics.

## 1. Vertices Count

Represents the number of vertices in the generated mesh.

Higher vertex counts typically indicate more detailed geometry but increase rendering cost.
### Vertices Comparison
<img src="Plots/vertices_comparison.png" width="600">

---

## 2. Faces Count

Represents the number of triangular faces in the mesh.

More faces often improve geometric fidelity but increase computational complexity.
### Faces Comparison
<img src="Plots/faces_comparison.png" width="600">

---

## 3. Surface Area

Measures the total surface area of the generated 3D object.

This helps compare how much geometric space the reconstructed object occupies.
### Surface Area Comparison
<img src="Plots/surface_area_comparison.png" width="600">

---

## 4. File Size

Represents the storage requirement of the generated mesh file.

Smaller file sizes are beneficial for real-time rendering and web applications.

### File Size Comparison
<img src="Plots/file_size_comparison.png" width="600">

---

## 5. Chamfer Distance

Chamfer Distance measures the similarity between two point clouds sampled from two meshes.

Lower Chamfer Distance indicates higher geometric similarity.

The Chamfer Distance is defined as:

```
CD(P,Q) = (1/|P|) Σp∈P minq∈Q ||p−q||² + (1/|Q|) Σq∈Q minp∈P ||q−p||²
```

Where:

P = sampled points from mesh 1
Q = sampled points from mesh 2

### Chamfer Distance per Object
<img src="Plots/chamfer_distance.png" width="600">

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/benchmark-image-to-3d.git

cd benchmark-image-to-3d
```

Install dependencies:

```
pip install trimesh
pip install pandas
pip install matplotlib
pip install torch
```

Optional dependency for faster mesh processing:

```
pip install pyembree
```

---

# Running the Benchmark

Place generated meshes in the following folders:

```
triposr_outputs/
shape_outputs/
```

Supported formats:

```
TripoSR  → .glb
Shap-E   → .obj
```

Run the benchmarking script:

```
python benchmark.py
```

---

# Experimental Setup

The experiments were conducted using a **GPU-accelerated environment** with a FastAPI backend for model inference.

Pipeline workflow:

1. Input image is processed by the reconstruction model.
2. The model generates a 3D mesh.
3. Meshes are saved in GLB or OBJ format.
4. The benchmarking script loads meshes and samples surface points.
5. Chamfer Distance is computed.
6. Mesh statistics are extracted.
7. Visualization plots are generated.

---

# Inference Time Results

The inference latency was measured for several generated objects.

| Object   | Inference Time (seconds) | Inference Time (ms) |
| -------- | ------------------------ | ------------------- |
| Iron Man | 34.6787                  | 34678.76            |
| Human    | 35.1744                  | 35174.41            |
| Penguin  | 35.3512                  | 35351.25            |
| Star     | 34.3891                  | 34389.11            |
| Robot    | 35.5813                  | 35581.37            |

Average inference time:

* **~35 seconds per object**
* **~35,000 ms per object**

This indicates stable GPU performance across different input objects.

---

# Output Files

The benchmarking script generates:

### Results Table

```
benchmark_results.csv
```

Example structure:

| Object | TripoSR Vertices | ShapE Vertices | Chamfer Distance |
| ------ | ---------------- | -------------- | ---------------- |
| robot  | 7421             | 5230           | 0.014            |

---

### Visualization Graphs

The system automatically generates comparison plots:

* Vertices comparison
* Faces comparison
* Surface area comparison
* File size comparison
* Chamfer distance per object

Generated files:

```
vertices_comparison.png
faces_comparison.png
surface_area_comparison.png
file_size_comparison.png
chamfer_distance.png
```

---

# Technologies Used

* Python
* PyTorch
* Trimesh
* Pandas
* Matplotlib
* FastAPI
* CUDA GPU acceleration

---

# Applications

This benchmarking framework can be used for:

* Evaluating generative 3D models
* Comparing reconstruction algorithms
* Research in text-to-3D and image-to-3D generation
* Testing geometry quality of neural reconstruction systems
* Performance analysis of GPU-based inference pipelines

---
