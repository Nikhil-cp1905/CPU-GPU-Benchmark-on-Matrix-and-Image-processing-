#!/usr/bin/env bash
set -e
# Activate venv if you use one; adjust path:
# source ./venv/bin/activate

python3 plot_professional.py

REPORT=report/report.html
if [ -f "$REPORT" ]; then
  # Try to open in default GUI browser (if available)
  xdg-open "$REPORT" >/dev/null 2>&1 || echo "Report saved: $REPORT"
else
  echo "Report not found - check plot_professional.py output"
fi

