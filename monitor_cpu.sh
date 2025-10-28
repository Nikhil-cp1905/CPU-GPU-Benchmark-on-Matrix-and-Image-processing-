#!/bin/bash
OUTDIR=${1:-cpu_snapshots}
mkdir -p "$OUTDIR"
i=0
while true; do
  date --iso-8601=ns > ${OUTDIR}/time_$i.txt
  top -b -n 1 | head -n 40 > ${OUTDIR}/top_$i.txt
  sleep 0.2
  i=$((i+1))
done

