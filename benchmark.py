import os
import gc
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import torch
import trimesh
import pyrender
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
triposr_dir = "triposr_outputs"
shape_dir = "shape_outputs"
input_img_dir = "input_images"

# Memory‑saving settings
n_sample_points = 5000              # Chamfer distance (was 10000)
render_resolution = (224, 224)      # PSNR/SSIM render size (was 256)
camera_poses = ['front', 'back', 'left', 'right']

# BATCH SETTINGS
BATCH_SIZE = 6                      # Process 6 objects at a time
RESUME = True                       # If True, skip already processed objects
CHECKPOINT_FILE = "checkpoint.csv"  # Saves progress
# ====================================

def load_mesh(path):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    return mesh

def sample_points(mesh, n_points=n_sample_points):
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return torch.tensor(points).float()

def chamfer_distance(mesh1, mesh2):
    p1 = sample_points(mesh1)
    p2 = sample_points(mesh2)
    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(0)
    dist = torch.cdist(p1, p2)
    cd = dist.min(1)[0].mean() + dist.min(0)[0].mean()
    del p1, p2, dist
    return cd.item()

def render_mesh(mesh, pose='front'):
    scene = pyrender.Scene()
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_pyrender)

    # Camera placement
    if pose == 'front':
        camera_pose = np.eye(4)
        camera_pose[2, 3] = 2.0
    elif pose == 'back':
        camera_pose = np.eye(4)
        camera_pose[2, 3] = -2.0
        camera_pose[0, 0] = -1
    elif pose == 'left':
        camera_pose = np.eye(4)
        camera_pose[0, 3] = -2.0
        camera_pose[2, 2] = 0
        camera_pose[2, 0] = 1
        camera_pose[0, 2] = -1
    elif pose == 'right':
        camera_pose = np.eye(4)
        camera_pose[0, 3] = 2.0
        camera_pose[2, 2] = 0
        camera_pose[2, 0] = -1
        camera_pose[0, 2] = 1
    else:
        raise ValueError("Unknown pose")

    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1,1,1], intensity=3.0)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(render_resolution[0], render_resolution[1])
    color, _ = r.render(scene)
    r.delete()
    del scene, r
    return color

def texture_quality(mesh_pred, input_img_path):
    if not os.path.exists(input_img_path):
        return None, None
    ref_img = cv2.imread(input_img_path)
    if ref_img is None:
        return None, None
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    ref_img = cv2.resize(ref_img, render_resolution)

    psnr_list, ssim_list = [], []
    for pose in camera_poses:
        rendered = render_mesh(mesh_pred, pose)
        if rendered.shape[-1] == 4:
            rendered = rendered[:,:,:3]
        rendered = rendered.astype(np.uint8)
        psnr = peak_signal_noise_ratio(ref_img, rendered, data_range=255)
        ssim = structural_similarity(ref_img, rendered, multichannel=True,
                                     data_range=255, channel_axis=2)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        del rendered
    return np.mean(psnr_list), np.mean(ssim_list)

def find_image(object_name, img_dir):
    """Find image file that contains object_name (case‑insensitive)."""
    name_lower = object_name.lower()
    for img_file in os.listdir(img_dir):
        if name_lower in img_file.lower() and img_file.lower().endswith(('.png','.jpg','.jpeg')):
            return os.path.join(img_dir, img_file)
    return None

# ========== LOAD / RESUME CHECKPOINT ==========
all_objects = [f.replace(".glb", "") for f in os.listdir(triposr_dir) if f.endswith(".glb")]
all_objects.sort()
print(f"Found {len(all_objects)} objects in triposr_outputs/")

results = []
processed_objects = set()

if RESUME and os.path.exists(CHECKPOINT_FILE):
    checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
    if "Object" in checkpoint_df.columns:
        processed_objects = set(checkpoint_df["Object"].astype(str))
        results = checkpoint_df.to_dict('records')
        print(f"Resuming from checkpoint. Already processed: {len(processed_objects)} objects")
    else:
        print("Checkpoint file corrupt – starting fresh.")

# Filter objects not yet processed
remaining_objects = [obj for obj in all_objects if obj not in processed_objects]
print(f"Remaining to process: {len(remaining_objects)}")

# ========== PROCESS IN BATCHES ==========
total_batches = (len(remaining_objects) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(0, len(remaining_objects), BATCH_SIZE):
    batch = remaining_objects[batch_idx:batch_idx + BATCH_SIZE]
    batch_num = batch_idx // BATCH_SIZE + 1
    print(f"\n{'='*60}")
    print(f"BATCH {batch_num}/{total_batches} – Processing {len(batch)} objects")
    print(f"{'='*60}")

    for obj_name in batch:
        print(f"\n>>> Processing {obj_name}...")

        # Paths
        triposr_path = os.path.join(triposr_dir, obj_name + ".glb")
        shape_path = os.path.join(shape_dir, obj_name + ".obj")
        if not os.path.exists(shape_path):
            print(f"  ❌ No Shap-E output for {obj_name}, skipping.")
            continue

        # Load meshes
        triposr_mesh = load_mesh(triposr_path)
        shape_mesh = load_mesh(shape_path)

        # Row dictionary
        row = {
            "Object": obj_name,
            "TripoSR Vertices": len(triposr_mesh.vertices),
            "TripoSR Faces": len(triposr_mesh.faces),
            "TripoSR SurfaceArea": triposr_mesh.area,
            "TripoSR FileSize_KB": os.path.getsize(triposr_path)/1024,
            "ShapE Vertices": len(shape_mesh.vertices),
            "ShapE Faces": len(shape_mesh.faces),
            "ShapE SurfaceArea": shape_mesh.area,
            "ShapE FileSize_KB": os.path.getsize(shape_path)/1024,
            "Chamfer Distance (TripoSR vs ShapE)": chamfer_distance(triposr_mesh, shape_mesh),
        }

        # Texture metrics
        img_path = find_image(obj_name, input_img_dir)
        if img_path:
            print(f"  ✅ Found image: {os.path.basename(img_path)}")
            psnr_t, ssim_t = texture_quality(triposr_mesh, img_path)
            if psnr_t is not None:
                row["TripoSR PSNR"] = psnr_t
                row["TripoSR SSIM"] = ssim_t
            psnr_s, ssim_s = texture_quality(shape_mesh, img_path)
            if psnr_s is not None:
                row["ShapE PSNR"] = psnr_s
                row["ShapE SSIM"] = ssim_s
        else:
            print(f"  ⚠️  No image found for {obj_name} – texture metrics skipped")

        results.append(row)

        # Save checkpoint after each object
        checkpoint_df = pd.DataFrame(results)
        checkpoint_df.to_csv(CHECKPOINT_FILE, index=False)
        print(f"  ✓ Saved checkpoint after {obj_name}")

        # Cleanup
        del triposr_mesh, shape_mesh
        gc.collect()

    # After each batch, optionally sleep to let system cool down
    if batch_idx + BATCH_SIZE < len(remaining_objects):
        print(f"\nBatch {batch_num} completed. Pausing 5 seconds before next batch...")
        time.sleep(5)

# ========== FINAL DATAFRAME AND OUTPUTS ==========
df = pd.DataFrame(results)
print("\n" + "="*60)
print("Benchmark Results (all objects)")
print("="*60)
print(df.to_string())
df.to_csv("benchmark_results.csv", index=False)
print("\n✓ Saved full benchmark_results.csv")

# ========== SUMMARY STATISTICS ==========
print("\n📊 Average Metrics\n")
print(df.mean(numeric_only=True))

# ========== PLOTS ==========
def save_plot(plt, filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")

# Vertices
plt.figure()
plt.bar(['TripoSR', 'ShapE'], [df["TripoSR Vertices"].mean(), df["ShapE Vertices"].mean()],
        yerr=[df["TripoSR Vertices"].std(), df["ShapE Vertices"].std()], capsize=5)
plt.title("Average Vertices Comparison")
plt.ylabel("Vertices")
save_plot(plt, "vertices_comparison.png")

# Faces
plt.figure()
plt.bar(['TripoSR', 'ShapE'], [df["TripoSR Faces"].mean(), df["ShapE Faces"].mean()],
        yerr=[df["TripoSR Faces"].std(), df["ShapE Faces"].std()], capsize=5)
plt.title("Average Faces Comparison")
plt.ylabel("Faces")
save_plot(plt, "faces_comparison.png")

# Surface Area
plt.figure()
plt.bar(['TripoSR', 'ShapE'], [df["TripoSR SurfaceArea"].mean(), df["ShapE SurfaceArea"].mean()],
        yerr=[df["TripoSR SurfaceArea"].std(), df["ShapE SurfaceArea"].std()], capsize=5)
plt.title("Average Surface Area Comparison")
plt.ylabel("Surface Area")
save_plot(plt, "surface_area_comparison.png")

# File Size
plt.figure()
plt.bar(['TripoSR', 'ShapE'], [df["TripoSR FileSize_KB"].mean(), df["ShapE FileSize_KB"].mean()],
        yerr=[df["TripoSR FileSize_KB"].std(), df["ShapE FileSize_KB"].std()], capsize=5)
plt.title("Average File Size Comparison")
plt.ylabel("KB")
save_plot(plt, "file_size_comparison.png")

# Chamfer distance per object
plt.figure(figsize=(12, 6))
plt.bar(df["Object"], df["Chamfer Distance (TripoSR vs ShapE)"], color='green', alpha=0.7)
plt.title("Chamfer Distance per Object (TripoSR vs ShapE)")
plt.ylabel("Chamfer Distance")
plt.xticks(rotation=90)
save_plot(plt, "chamfer_distance_between_models.png")

# Texture metrics if available
if "TripoSR PSNR" in df.columns:
    x = np.arange(len(df["Object"]))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, df["TripoSR PSNR"], width, label="TripoSR")
    plt.bar(x + width/2, df["ShapE PSNR"], width, label="ShapE")
    plt.xticks(x, df["Object"], rotation=90)
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Input Image")
    plt.legend()
    save_plot(plt, "psnr.png")

if "TripoSR SSIM" in df.columns:
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, df["TripoSR SSIM"], width, label="TripoSR")
    plt.bar(x + width/2, df["ShapE SSIM"], width, label="ShapE")
    plt.xticks(x, df["Object"], rotation=90)
    plt.ylabel("SSIM")
    plt.title("SSIM vs Input Image")
    plt.legend()
    save_plot(plt, "ssim.png")

print("\n✅ All plots saved.")
print("🏁 Benchmark complete.")