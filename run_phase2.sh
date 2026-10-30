#!/bin/bash
# Phase 2 — 4모델 × 5seed = 20회 학습을 빈 GPU 4개(0,3,6,7)에 분배
# 사용: conda activate pedestrian && nohup bash run_phase2.sh > logs/phase2_master.txt 2>&1 &
set -u
cd /workspace/minseok_park
mkdir -p logs

# ── ego 표준화 통계 미리 생성 (병렬 race 방지) ──
if [ ! -f cache/ego_stats.npz ]; then
  echo "[stats] ego_stats.npz 생성 중..."
  python - <<'PY'
import numpy as np, sys
sys.path.insert(0, '.')
from nuscenes.nuscenes import NuScenes
from src.data.nuscenes_ttc_dataset import NuScenesTTCDataset
nusc = NuScenes(version='v1.0-trainval', dataroot='/workspace/data/nuscenes', verbose=False)
ds = NuScenesTTCDataset(nusc, split='train', cache_path='cache/frames_train.npz')
np.savez('cache/ego_stats.npz', mean=ds.ego_mean, std=ds.ego_std)
print('[stats] 저장 완료')
PY
fi

run_jobs() {
  local gpu=$1; shift
  for job in "$@"; do
    M=${job% *}; S=${job#* }
    echo "[GPU$gpu] $M seed$S 시작 $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu python -u src/train_paper.py \
        --mode "$M" --seed "$S" --workers 6 \
        > logs/train_${M}_seed${S}.txt 2>&1
    echo "[GPU$gpu] $M seed$S 완료 $(date +%H:%M:%S)"
  done
}

run_jobs 0 "image_only 42"  "dummy 42"  "velocity_only 42"  "full 42"  "image_only 999"    &
run_jobs 3 "image_only 123" "dummy 123" "velocity_only 123" "full 123" "dummy 999"          &
run_jobs 6 "image_only 7"   "dummy 7"   "velocity_only 7"   "full 7"   "velocity_only 999"  &
run_jobs 7 "image_only 555" "dummy 555" "velocity_only 555" "full 555" "full 999"           &
wait
echo "===== Phase 2 전체 완료 $(date +%H:%M:%S) ====="