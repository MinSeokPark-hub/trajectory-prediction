#!/bin/bash
# Phase 2 평가 — 20개 체크포인트로 test 추론, pred 저장
# 사용: conda activate pedestrian && nohup bash run_eval.sh > logs/eval_master.txt 2>&1 &
set -u
cd /workspace/minseok_park
mkdir -p logs preds

run_eval() {
  local gpu=$1; shift
  for job in "$@"; do
    M=${job% *}; S=${job#* }
    echo "[GPU$gpu] eval $M seed$S 시작 $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu python -u src/evaluate_paper.py \
        --mode "$M" --seed "$S" --split test --workers 6 \
        > logs/eval_${M}_seed${S}.txt 2>&1
    echo "[GPU$gpu] eval $M seed$S 완료 $(date +%H:%M:%S)"
  done
}

run_eval 0 "image_only 42"  "dummy 42"  "velocity_only 42"  "full 42"  "image_only 999"    &
run_eval 3 "image_only 123" "dummy 123" "velocity_only 123" "full 123" "dummy 999"          &
run_eval 6 "image_only 7"   "dummy 7"   "velocity_only 7"   "full 7"   "velocity_only 999"  &
run_eval 7 "image_only 555" "dummy 555" "velocity_only 555" "full 555" "full 999"           &
wait
echo "===== 평가 전체 완료 $(date +%H:%M:%S) ====="
echo "pred 파일: $(ls preds/pred_*_test.npz 2>/dev/null | wc -l)/20"