import os
import trimesh
import pandas as pd
import matplotlib.pyplot as plt
import torch


triposr_dir = "triposr_outputs"
shape_dir = "shape_outputs"


# ---------- LOAD MESH FUNCTION ----------
def load_mesh(path):

    mesh = trimesh.load(path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))

    return mesh


# ---------- SAMPLE POINTS ----------
def sample_points(mesh, n_points=5000):

    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return torch.tensor(points).float()


# ---------- CHAMFER DISTANCE ----------
def chamfer_distance(mesh1, mesh2):

    p1 = sample_points(mesh1)
    p2 = sample_points(mesh2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(0)

    dist = torch.cdist(p1, p2)

    cd = dist.min(1)[0].mean() + dist.min(0)[0].mean()

    return cd.item()


# ---------- BENCHMARK LOOP ----------
results = []

files = os.listdir(triposr_dir)

for f in files:

    if f.endswith(".glb"):

        name = f.replace(".glb","")

        triposr_path = os.path.join(triposr_dir, f)
        shape_path = os.path.join(shape_dir, name + ".obj")

        triposr_mesh = load_mesh(triposr_path)
        shape_mesh = load_mesh(shape_path)

        cd = chamfer_distance(triposr_mesh, shape_mesh)

        results.append({

            "Object": name,

            "TripoSR Vertices": len(triposr_mesh.vertices),
            "TripoSR Faces": len(triposr_mesh.faces),
            "TripoSR SurfaceArea": triposr_mesh.area,
            "TripoSR FileSize_KB": os.path.getsize(triposr_path)/1024,

            "ShapE Vertices": len(shape_mesh.vertices),
            "ShapE Faces": len(shape_mesh.faces),
            "ShapE SurfaceArea": shape_mesh.area,
            "ShapE FileSize_KB": os.path.getsize(shape_path)/1024,

            "Chamfer Distance": cd
        })


# ---------- CREATE DATAFRAME ----------
df = pd.DataFrame(results)

print("\nBenchmark Results\n")
print(df)

df.to_csv("benchmark_results.csv", index=False)


# ---------- AVERAGE METRICS ----------
print("\nAverage Metrics\n")
print(df.mean(numeric_only=True))


# ---------- PLOTS ----------

models = ["TripoSR", "Shap-E"]

avg_vertices = [df["TripoSR Vertices"].mean(), df["ShapE Vertices"].mean()]
avg_faces = [df["TripoSR Faces"].mean(), df["ShapE Faces"].mean()]
avg_area = [df["TripoSR SurfaceArea"].mean(), df["ShapE SurfaceArea"].mean()]
avg_size = [df["TripoSR FileSize_KB"].mean(), df["ShapE FileSize_KB"].mean()]


# Vertices Plot
plt.figure()
plt.bar(models, avg_vertices)
plt.title("Average Vertices Comparison")
plt.ylabel("Vertices")
plt.savefig("vertices_comparison.png")


# Faces Plot
plt.figure()
plt.bar(models, avg_faces)
plt.title("Average Faces Comparison")
plt.ylabel("Faces")
plt.savefig("faces_comparison.png")


# Surface Area Plot
plt.figure()
plt.bar(models, avg_area)
plt.title("Surface Area Comparison")
plt.ylabel("Surface Area")
plt.savefig("surface_area_comparison.png")


# File Size Plot
plt.figure()
plt.bar(models, avg_size)
plt.title("File Size Comparison")
plt.ylabel("KB")
plt.savefig("file_size_comparison.png")


# Chamfer Plot
plt.figure()
plt.bar(df["Object"], df["Chamfer Distance"])
plt.title("Chamfer Distance per Object")
plt.ylabel("Chamfer Distance")
plt.xlabel("Object")
plt.savefig("chamfer_distance.png")


plt.show()
