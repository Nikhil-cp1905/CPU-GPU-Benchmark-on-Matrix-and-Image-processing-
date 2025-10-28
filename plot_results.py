import pandas as pd, matplotlib.pyplot as plt
# matrix
mat = pd.read_csv("matrix_results.csv")
agg = mat.groupby(["backend","size"])["total_ms"].median().reset_index()
pivot = agg.pivot(index="size", columns="backend", values="total_ms")
plt.figure(figsize=(8,5))
for col in pivot.columns:
    plt.plot(pivot.index, pivot[col], marker='o', label=col)
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel("Matrix size (N)")
plt.ylabel("Median total time (ms)")
plt.title("Matrix multiply: CPU (NumPy) vs GPU (CuPy)")
plt.legend(); plt.grid(True,which='both',ls='--')
plt.savefig("matrix_time_vs_size.png", dpi=200)
# speedup
if 'numpy' in pivot.columns and 'cupy' in pivot.columns:
    speed = pivot['numpy'] / pivot['cupy']
    plt.figure(figsize=(6,4))
    plt.plot(pivot.index, speed, marker='o')
    plt.xscale('log', base=2)
    plt.xlabel("Matrix size (N)")
    plt.ylabel("Speedup (T_cpu/T_gpu)")
    plt.title("Speedup vs size")
    plt.grid(True); plt.savefig("matrix_speedup.png", dpi=200)
# image
img = pd.read_csv("image_results.csv")
img_agg = img.groupby(["backend","size"])["compute_ms"].median().reset_index()
img_p = img_agg.pivot(index="size", columns="backend", values="compute_ms")
plt.figure(figsize=(8,5))
for col in img_p.columns:
    plt.plot(img_p.index, img_p[col], marker='o', label=col)
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel("Image size (px)")
plt.ylabel("Median compute time (ms)")
plt.title("Image processing: CPU vs GPU")
plt.legend(); plt.grid(True); plt.savefig("image_time_vs_size.png", dpi=200)
# gpu utilization plot (if exists)
try:
    g = pd.read_csv("gpu_log.csv", parse_dates=["timestamp"])
    # gpu utilization column name varies, take second column
    cols = list(g.columns)
    util_col = cols[1]
    plt.figure(figsize=(10,3))
    plt.plot(g['timestamp'], g[util_col])
    plt.xlabel("Time"); plt.ylabel("GPU util (%)"); plt.title("GPU utilization over time")
    plt.tight_layout(); plt.savefig("gpu_util_time.png", dpi=200)
except Exception as e:
    print("Skipping gpu util plot:", e)
print("Saved PNGs: matrix_time_vs_size.png, matrix_speedup.png (if computed), image_time_vs_size.png, gpu_util_time.png")

