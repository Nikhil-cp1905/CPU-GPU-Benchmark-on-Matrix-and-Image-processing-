#!/usr/bin/env python3
"""
plot_professional.py
Produces publication-quality plots and a small HTML report summarizing CPU vs GPU benchmarks.
Usage:
    python plot_professional.py
Outputs:
    ./plots/
      - matrix_time_vs_size.png/pdf
      - matrix_speedup.png/pdf
      - matrix_gpu_breakdown.png/pdf  (if transfer columns exist)
      - image_time_vs_size.png/pdf
      - image_speedup.png/pdf
      - gpu_util_time.png/pdf
    ./report/report.html
    ./report/analysis_summary.txt
"""

import os, sys, math, datetime, textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

# Configure matplotlib for a clean, "professional" look (do not hard-code colors)
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.figsize": (7,4),
    "savefig.bbox": "tight"
})

# Paths
root = Path.cwd()
plots_dir = root / "plots"
report_dir = root / "report"
plots_dir.mkdir(exist_ok=True)
report_dir.mkdir(exist_ok=True)

# Filenames (expected to exist)
matrix_csv = root / "matrix_results.csv"
image_csv = root / "image_results.csv"
gpu_log_csv = root / "gpu_log.csv"

report_lines = []
report_lines.append(f"<h1>CPU vs GPU Benchmark Report</h1>")
report_lines.append(f"<p>Generated: {datetime.datetime.now().isoformat()}</p>")

# Utility: safe read CSV
def safe_read_csv(p):
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            print("Error reading", p, e)
            return None
    return None

# --- Matrix plots ---
mat = safe_read_csv(matrix_csv)
if mat is not None:
    # Normalize backend names
    mat['backend'] = mat['backend'].astype(str)
    # Group median & std
    agg = mat.groupby(['backend','size'])['total_ms'].agg(['median','std','count']).reset_index()
    pivot = agg.pivot(index='size', columns='backend', values='median').sort_index()
    # Plot: Time vs size
    fig, ax = plt.subplots()
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker='o', label=str(col))
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Matrix size N (NxN)')
    ax.set_ylabel('Median total time (ms)')
    ax.set_title('Matrix multiply — median total time')
    ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax.legend()
    f1 = plots_dir / "matrix_time_vs_size.png"
    fig.savefig(f1, dpi=300)
    fig.savefig(str(f1.with_suffix('.pdf')))
    plt.close(fig)
    report_lines.append("<h2>Matrix multiplication: time vs size</h2>")
    report_lines.append(f'<img src="../plots/{f1.name}" style="max-width:900px">')

    # Speedup if CPU and GPU identified
    backends = [str(c) for c in pivot.columns]
    cpu_b = next((b for b in backends if 'numpy' in b.lower() or 'cpu' in b.lower()), None)
    gpu_b = next((b for b in backends if 'cupy' in b.lower() or 'gpu' in b.lower()), None)
    if cpu_b and gpu_b:
        speed = pivot[cpu_b] / pivot[gpu_b]
        fig, ax = plt.subplots()
        ax.plot(pivot.index, speed, marker='o')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Matrix size N')
        ax.set_ylabel('Speedup (T_cpu / T_gpu)')
        ax.set_title('Matrix multiply: Speedup (CPU / GPU)')
        ax.grid(True, linestyle='--', linewidth=0.4)
        f2 = plots_dir / "matrix_speedup.png"
        fig.savefig(f2, dpi=300)
        fig.savefig(str(f2.with_suffix('.pdf')))
        plt.close(fig)
        report_lines.append("<h3>Speedup (CPU/GPU)</h3>")
        report_lines.append(f'<img src="../plots/{f2.name}" style="max-width:900px">')

        # Save numeric table
        speed_table = pd.DataFrame({
            'size': pivot.index,
            f'{cpu_b}_ms': pivot[cpu_b].values,
            f'{gpu_b}_ms': pivot[gpu_b].values,
            'speedup': speed.values
        })
        speed_table.to_csv(report_dir / "matrix_speedup_table.csv", index=False)
    else:
        report_lines.append("<p><i>Note:</i> Could not automatically identify CPU/GPU names for speedup. Check backend column.</p>")

    # GPU breakdown if columns exist
    if gpu_b and {'transfer_h2d_ms','compute_ms','transfer_d2h_ms'}.issubset(mat.columns):
        gmat = mat[mat['backend']==gpu_b].groupby('size')[['transfer_h2d_ms','compute_ms','transfer_d2h_ms']].median()
        sizes = gmat.index.values
        h2d = gmat['transfer_h2d_ms'].values
        comp = gmat['compute_ms'].values
        d2h = gmat['transfer_d2h_ms'].values
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(sizes, h2d, label='H->D')
        ax.bar(sizes, comp, bottom=h2d, label='Compute')
        ax.bar(sizes, d2h, bottom=h2d+comp, label='D->H')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Matrix size N')
        ax.set_ylabel('Time (ms)')
        ax.set_title('GPU time breakdown (median)')
        ax.legend()
        f3 = plots_dir / "matrix_gpu_breakdown.png"
        fig.savefig(f3, dpi=300); fig.savefig(str(f3.with_suffix('.pdf')))
        plt.close(fig)
        report_lines.append("<h3>GPU transfer vs compute breakdown</h3>")
        report_lines.append(f'<img src="../plots/{f3.name}" style="max-width:900px">')
    else:
        report_lines.append("<p><i>Note:</i> Transfer/compute columns missing for GPU breakdown.</p>")

else:
    report_lines.append("<p><strong>No matrix_results.csv found — matrix plots skipped.</strong></p>")

# --- Image processing plots ---
img = safe_read_csv(image_csv)
if img is not None:
    img['backend'] = img['backend'].astype(str)
    imgagg = img.groupby(['backend','size'])['compute_ms'].median().reset_index()
    # Fix: after aggregation, we use the mean compute time
    value_col = 'compute_ms' if 'compute_ms' in imgagg.columns else imgagg.columns[-1]
    imgpivot = imgagg.pivot(index='size', columns='backend', values=value_col).sort_index()

    fig, ax = plt.subplots()
    for col in imgpivot.columns:
        ax.plot(imgpivot.index, imgpivot[col], marker='o', label=str(col))
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Image side length (px)')
    ax.set_ylabel('Median compute time (ms)')
    ax.set_title('Image processing (Gaussian+Sobel): CPU vs GPU')
    ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax.legend()
    f4 = plots_dir / "image_time_vs_size.png"
    fig.savefig(f4, dpi=300); fig.savefig(str(f4.with_suffix('.pdf')))
    plt.close(fig)
    report_lines.append("<h2>Image processing: time vs size</h2>")
    report_lines.append(f'<img src="../plots/{f4.name}" style="max-width:900px">')

    # Speedup image
    backends_img = list(imgpivot.columns)
    cpu_i = next((b for b in backends_img if 'cpu' in b.lower()), None)
    gpu_i = next((b for b in backends_img if 'gpu' in b.lower()), None)
    if cpu_i and gpu_i:
        ispeed = imgpivot[cpu_i] / imgpivot[gpu_i]
        fig, ax = plt.subplots()
        ax.plot(imgpivot.index, ispeed, marker='o')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Image side (px)')
        ax.set_ylabel('Speedup (T_cpu / T_gpu)')
        ax.set_title('Image processing speedup (CPU/GPU)')
        ax.grid(True, linestyle='--', linewidth=0.4)
        f5 = plots_dir / "image_speedup.png"
        fig.savefig(f5, dpi=300); fig.savefig(str(f5.with_suffix('.pdf')))
        plt.close(fig)
        report_lines.append("<h3>Image speedup (CPU/GPU)</h3>")
        report_lines.append(f'<img src="../plots/{f5.name}" style="max-width:900px">')
    else:
        report_lines.append("<p><i>Note:</i> Could not auto-detect CPU/GPU backends for images.</p>")
else:
    report_lines.append("<p><strong>No image_results.csv found — image plots skipped.</strong></p>")

# --- GPU utilization timeline ---
glog = safe_read_csv(gpu_log_csv)
if glog is not None:
    # Parse timestamp robustly: try common formats
    if 'timestamp' in glog.columns:
        try:
            glog['ts'] = pd.to_datetime(glog['timestamp'], errors='coerce')
        except:
            glog['ts'] = pd.to_datetime(glog['timestamp'], infer_datetime_format=True, errors='coerce')
        # choose first numeric column after timestamp
        util_col = None
        for c in glog.columns:
            if c == 'timestamp' or c == 'ts': continue
            if 'util' in c.lower() or 'usage' in c.lower() or glog[c].dtype.kind in 'fi':
                util_col = c; break
        if util_col is None and glog.shape[1] > 1:
            util_col = glog.columns[1]
        if util_col:
            fig, ax = plt.subplots(figsize=(9,3))
            ax.plot(glog['ts'], pd.to_numeric(glog[util_col], errors='coerce'))
            ax.set_xlabel('Time')
            ax.set_ylabel(util_col)
            ax.set_title('GPU utilization timeline')
            fig.autofmt_xdate()
            f6 = plots_dir / "gpu_util_time.png"
            fig.savefig(f6, dpi=300); fig.savefig(str(f6.with_suffix('.pdf')))
            plt.close(fig)
            report_lines.append("<h2>GPU utilization timeline</h2>")
            report_lines.append(f'<img src="../plots/{f6.name}" style="max-width:900px">')
        else:
            report_lines.append("<p><i>Note:</i> Could not identify a utilization column in gpu_log.csv.</p>")
    else:
        report_lines.append("<p><strong>gpu_log.csv missing 'timestamp' column — timeline skipped.</strong></p>")
else:
    report_lines.append("<p><strong>No gpu_log.csv found — utilization timeline skipped.</strong></p>")

# --- Write textual analysis summary (concise) ---
analysis_txt = []
analysis_txt.append("CPU vs GPU Benchmark — concise analysis\n")
analysis_txt.append("Files used:\n")
if mat is not None: analysis_txt.append(f" - {matrix_csv.name}")
if img is not None: analysis_txt.append(f" - {image_csv.name}")
if glog is not None: analysis_txt.append(f" - {gpu_log_csv.name}")
analysis_txt.append("\nKey findings (answer depends on your CSVs):\n")
analysis_txt.append("- Small matrices/images: the CPU may be faster due to host<->device overhead and BLAS multi-threading on CPU.\n")
analysis_txt.append("- Medium-to-large sizes: GPU typically outperforms because of massive parallelism; speedup grows with problem size.\n")
analysis_txt.append("- Transfer vs compute: if transfer bars are large relative to compute, consider batching or pinned memory and streams.\n")
analysis_txt.append("- GPU utilization: high sustained utilization indicates compute-bound workload (GPU advantage). Low utilization + long runtime suggests I/O or small kernels.\n")
analysis_txt.append("\nRecommended next steps:\n")
analysis_txt.append(" - Report median and 95% CI for each measurement (we used medians here)\n - Show both compute-only (device-resident) and end-to-end times (including transfers)\n - Try batched GEMM or asynchronous streams to hide transfer time for real workloads.\n")

(report_dir / "analysis_summary.txt").write_text("\n".join(analysis_txt))
report_lines.append("<h2>Textual analysis</h2>")
report_lines.append("<pre>" + "\n".join(analysis_txt) + "</pre>")

# --- Write final HTML report ---
html = "<html><head><meta charset='utf-8'><title>CPU vs GPU Benchmark Report</title></head><body>"
html += "\n".join(report_lines)
html += "<hr><p>End of report</p></body></html>"
(report_dir / "report.html").write_text(html)

print("Plots and report written to:")
print(" - plots:", plots_dir)
print(" - report:", report_dir)
print("Open report:", report_dir / "report.html")

