from abc import ABC, abstractmethod
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# 모든 파서가 공통으로 반환해야 하는 컬럼 (downstream 코드 기준)
REQUIRED_COLUMNS = [
    'frame', 'track_id', 'type',
    'x', 'y', 'w', 'h',        # dataset.py / SimpleTTC 용
    'pos_x', 'pos_y', 'pos_z', # LSTM 용 3D 좌표
    'x_pix', 'y_pix', 'w_pix', 'h_pix',  # 시각화 용 픽셀 좌표
]


class BaseParser(ABC):
    """
    KittiParser / SGANParser 공통 인터페이스.
    하위 클래스는 parse_label() 과 calculate_gt_ttc() 를 반드시 구현해야 한다.
    """

    def __init__(self, fps: float):
        self.fps = fps
        self.dt = 1.0 / fps

    # ── 하위 클래스가 반드시 구현 ──────────────────────────────

    @abstractmethod
    def parse_label(self, file_path: str) -> pd.DataFrame:
        """
        파일을 읽어 공통 스키마(REQUIRED_COLUMNS)의 DataFrame을 반환한다.
        빈 파일이면 빈 DataFrame을 반환하고 예외를 발생시키지 않는다.
        """
        ...

    @abstractmethod
    def calculate_gt_ttc(self, df: pd.DataFrame) -> pd.DataFrame:
        """velocity 와 gt_ttc 컬럼을 추가해서 반환한다."""
        ...

    # ── 공통 유틸 ──────────────────────────────────────────────

    def load(self, file_path: str) -> pd.DataFrame:
        """
        parse_label -> validate -> calculate_gt_ttc 를 한 번에 처리.
        파일이 비어 있거나 파싱 실패 시 빈 DataFrame을 반환하고 로그를 남긴다.
        """
        try:
            df = self.parse_label(file_path)
            if df.empty:
                logger.warning(f"빈 파일 스킵: {file_path}")
                return pd.DataFrame()
            self._validate(df)
            df = self.calculate_gt_ttc(df)
            return df
        except Exception as e:
            logger.error(f"파싱 실패 [{file_path}]: {e}")
            return pd.DataFrame()

    def _validate(self, df: pd.DataFrame) -> None:
        """REQUIRED_COLUMNS 가 모두 존재하는지 확인한다."""
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"누락된 컬럼: {missing}")
