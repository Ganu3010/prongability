#!/usr/bin/env bash
# Runs collect_data.py once per (town, weather) PAIR, each in its own
# subprocess -- not once per town with all weather presets passed together.
#
# Why this changed from the original per-town version: the crash we kept
# hitting (uncatchable C++ std::terminate from an actor-cleanup race, see
# collect_data.py's settle-window comment) happens specifically when
# collect_data.py reloads the world for its NEXT (town, weather) combo
# internally. Sharding by town alone still let collect_data.py loop over
# all 4 weather presets in one process -- so if the crash hit on the very
# first combo transition (moving from weather #1 to weather #2), every
# town's subprocess would silently produce data for only ONE weather
# preset before dying, and the retry (same command, same all-weathers
# list) would likely hit the identical wall again. This is almost
# certainly why a prior run went through all 5 towns but ended up with a
# suspiciously small, single-weather-per-town dataset.
#
# Sharding by (town, weather) pair means each subprocess only ever
# collects ONE combo and then exits normally -- the crash-prone "reload
# world for the next combo" code path never executes within any given
# process, so there's nothing for that race to hit. A crash can now only
# ever cost one (town, weather) combo's in-flight episode, not an entire
# town's worth of weather presets.
#
# Output: rulebook_dataset/Town01_ClearNoon/, rulebook_dataset/Town01_WetNoon/,
# etc. -- one subdirectory per combo, each independently numbered from
# episode_0000. Use merge_towns.py afterward to consolidate into one
# dataset with globally unique episode IDs before running postprocess.py /
# probing.py across everything at once.
#
# Usage: ./run_cluster.sh
# Full output (yours and collect_data.py's) is also saved to ./logs/run_cluster_<timestamp>.log

set -uo pipefail

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/run_cluster_$(date +%Y%m%d_%H%M%S).log"
# Redirect all of this script's stdout+stderr (including everything printed
# by collect_data.py) to both the terminal and a log file, so it survives
# even if the terminal session itself doesn't (nohup/tmux/disconnect/etc).
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging this run to: $LOG_FILE"

TOWNS=("Town01" "Town02" "Town03" "Town04" "Town05")
WEATHER=("ClearNoon" "WetNoon" "HardRainNoon" "CloudySunset")
OUTPUT_BASE="./rulebook_dataset"
EPISODES_PER_COMBO=5
FRAMES_PER_EPISODE=200
MAX_RETRIES=10  # safe to set high now: retries resume from existing episodes rather than overwriting them

for TOWN in "${TOWNS[@]}"; do
  for WX in "${WEATHER[@]}"; do
    ATTEMPT=0
    COMBO_OUTPUT="${OUTPUT_BASE}/${TOWN}_${WX}"
    while [ "$ATTEMPT" -le "$MAX_RETRIES" ]; do
      echo "=== Running ${TOWN}/${WX} (attempt $((ATTEMPT + 1))/$((MAX_RETRIES + 1))) ==="
      PYTHONUNBUFFERED=1 python collect_data.py \
        --towns "$TOWN" \
        --weather "$WX" \
        --episodes-per-combo "$EPISODES_PER_COMBO" \
        --frames-per-episode "$FRAMES_PER_EPISODE" \
        --seed "$RANDOM" \
        --output "$COMBO_OUTPUT"
      STATUS=$?
      if [ "$STATUS" -eq 0 ]; then
        echo "=== ${TOWN}/${WX} completed cleanly ==="
        break
      fi
      echo "=== ${TOWN}/${WX} exited with status ${STATUS} (crash or error) ==="
      ATTEMPT=$((ATTEMPT + 1))
      if [ "$ATTEMPT" -le "$MAX_RETRIES" ]; then
        echo "=== retrying ${TOWN}/${WX} — episodes already written to ${COMBO_OUTPUT} are untouched ==="
      else
        echo "=== giving up on ${TOWN}/${WX} after ${MAX_RETRIES} retries, moving to next combo ==="
      fi
    done
  done
done

echo ""
echo "All (town, weather) combos processed. Per-combo output under ${OUTPUT_BASE}/<Town>_<Weather>/."
echo "Merge into one dataset with globally unique episode IDs, then post-process the whole thing:"
echo "  python merge_towns.py --input ${OUTPUT_BASE} --output ${OUTPUT_BASE}_merged"
echo "  python postprocess.py --dataset ${OUTPUT_BASE}_merged"
echo "  python probing.py --dataset ${OUTPUT_BASE}_merged --output ${OUTPUT_BASE}_merged/probe_table.csv"