
OUT=${1:-gpu_log.csv}
echo "timestamp,utilization.gpu,utilization.memory,memory.used,power.draw" > "$OUT"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv,noheader,nounits -lms 200 | \
while IFS= read -r line; do
  echo "$(date --iso-8601=ns),$line" >> "$OUT"
done

