# Start GPU monitor
bash monitor_gpu.sh gpu_log.csv & echo $! > gpu_monitor.pid
# Start CPU monitor
bash monitor_cpu.sh cpu_snapshots & echo $! > cpu_monitor.pid || true
# Give monitors a moment
sleep 1
# Run benchmarks (example sizes — edit if OOM)
export OMP_NUM_THREADS=8
python bench_matrix.py --sizes 256 512 1024 --iters 6
python bench_image.py --sizes 512 1024 --iters 6
# Stop monitors
kill $(cat gpu_monitor.pid) && rm gpu_monitor.pid
if [ -f cpu_monitor.pid ]; then kill $(cat cpu_monitor.pid) && rm cpu_monitor.pid; fi
# Plot
python plot_results.py
echo "All done. Check PNGs and CSVs in $(pwd)"

