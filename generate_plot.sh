#!/bin/bash
while [[ $# -gt 0 ]]; do
  case $1 in
    --task)
      task="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 --task TASK_NAME"
      exit 1
      ;;
  esac
done
if [[ -z "$task" ]]; then
  echo "Error: --task TASK_NAME is required"
  exit 1
fi
python algorithmic/convenience_scripts/generate_summary_plots.py \
    --task $task \
    --group-pattern hyb \
    --group-pattern ssm \
    --group-pattern lm \
    --group-pattern \* \
    --include-max-only \
    --title "${task^^} Maximum Results over Models" --output exports/plots/${task}_max.png