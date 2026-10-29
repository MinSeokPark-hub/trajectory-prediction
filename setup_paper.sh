#!/bin/bash
# 논문 구현 파일 배치 스크립트
# 실행: bash setup_paper.sh (프로젝트 루트에서)

set -e
REPO=/workspace/minseok_park/trajectory-prediction

echo "=== 폴더 생성 ==="
mkdir -p $REPO/src/utils
mkdir -p $REPO/src/models
mkdir -p $REPO/src/data
mkdir -p $REPO/checkpoints
mkdir -p $REPO/cache

echo "=== __init__.py 생성 ==="
touch $REPO/src/__init__.py
touch $REPO/src/utils/__init__.py
touch $REPO/src/models/__init__.py
touch $REPO/src/data/__init__.py

echo "=== 파일 복사 ==="
SCRATCH=/tmp/paper_files   # 파일들을 여기 올려놓고 복사

cp $SCRATCH/sat_ttc.py            $REPO/src/utils/sat_ttc.py
cp $SCRATCH/ego_motion.py         $REPO/src/utils/ego_motion.py
cp $SCRATCH/resnet_ttc.py         $REPO/src/models/resnet_ttc.py
cp $SCRATCH/nuscenes_ttc_dataset.py  $REPO/src/data/nuscenes_ttc_dataset.py
cp $SCRATCH/train_paper.py        $REPO/src/train_paper.py
cp $SCRATCH/evaluate_paper.py     $REPO/src/evaluate_paper.py

echo "=== pip 패키지 확인 ==="
conda run -n pedestrian pip install scikit-learn pyquaternion --quiet

echo "=== 완료 ==="
echo "학습 명령어:"
echo "  conda activate pedestrian"
echo "  cd $REPO"
echo "  # 전체 모드 학습"
echo "  python src/train_paper.py --dataroot /workspace/data/nuscenes --mode full"
echo ""
echo "  # 4개 모드 한꺼번에 (백그라운드)"
echo "  for mode in image_only dummy velocity_only full; do"
echo "    python src/train_paper.py --mode \$mode --outdir checkpoints/\$mode &"
echo "  done"
echo "  wait"
echo ""
echo "평가 명령어:"
echo "  python src/evaluate_paper.py --ckpt checkpoints/full/best_full.pt --mode full"
