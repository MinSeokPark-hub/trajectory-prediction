import os
import pandas as pd
import logging

from utils.kitti_parser import KittiParser
from utils.sgan_parser import SGANParser

logger = logging.getLogger(__name__)


def load_all_kitti_data(label_dir: str) -> pd.DataFrame:
    return _load_all(label_dir, parser=KittiParser(fps=10))


def load_all_sgan_data(label_dir: str) -> pd.DataFrame:
    return _load_all(label_dir, parser=SGANParser(fps=2.5))


def _load_all(label_dir: str, parser) -> pd.DataFrame:
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    logger.info(f"{len(label_files)}개 파일 발견: {label_dir}")

    sequences = []
    for file_name in label_files:
        file_path = os.path.join(label_dir, file_name)
        df = parser.load(file_path)

        if df.empty:
            continue

        prefix = file_name.replace('.txt', '')
        df['track_id'] = prefix + '_' + df['track_id'].astype(str)
        sequences.append(df)

    if not sequences:
        logger.error(f"로드된 데이터 없음: {label_dir}")
        return pd.DataFrame()

    final_df = pd.concat(sequences, ignore_index=True)
    logger.info(f"통합 완료: {len(final_df)}개 데이터 포인트")
    return final_df