import os
import sys
import glob
import matplotlib.pyplot as plt

# 경로 설정
sys.path.append(os.path.abspath("/workspace/minseok_park/"))
from src.utils.sgan_parser import SGANParser
from src.inference_engine import InferenceEngine

def process_sgan_sequence():
    print("🚀 ETH/UCY (탑다운 레이더 시점) 분석 및 시각화 시작...")
    
    # 1. 파서 및 인공지능 엔진 초기화
    parser = SGANParser(fps=2.5)
    engine = InferenceEngine()
    
    # 2. 데이터 자동 탐색 (수정된 부분)
    train_dir = "/workspace/minseok_park/data/sgan/datasets/eth/train/"
    txt_files = glob.glob(os.path.join(train_dir, "*.txt"))
    
    if not txt_files:
        print(f"❌ 에러: {train_dir} 경로에 텍스트 데이터 파일이 없습니다!")
        return
        
    file_path = txt_files[0] # 첫 번째 발견된 텍스트 파일 사용
    print(f"📂 발견된 데이터 파일 로드: {file_path}")
    
    df = parser.parse_label(file_path)
    df = parser.calculate_gt_ttc(df)
    
    # 결과물 저장 폴더 생성
    output_dir = "/workspace/minseok_park/output/sgan_eth"
    os.makedirs(output_dir, exist_ok=True)
    
    # 시간순(프레임순) 정렬
    frames = sorted(df['frame'].unique())
    
    # 결과를 빠르게 확인하기 위해 처음 100프레임만 렌더링
    for frame_idx in frames[:100]:
        current_objects = df[df['frame'] == frame_idx]
        if len(current_objects) == 0: continue
        
        # 2D 레이더 도화지 세팅
        plt.figure(figsize=(10, 10))
        plt.xlim(-10, 15) 
        plt.ylim(-5, 15)  
        plt.title(f"SR-LSTM Bird's Eye View Radar (Frame {int(frame_idx)})", fontsize=15)
        
        for _, obj in current_objects.iterrows():
            tid = obj['track_id']
            # 과거 5프레임 궤적 확보
            history = df[(df['track_id'] == tid) & (df['frame'] <= frame_idx)].tail(5)
            
            if len(history) < 5:
                # 정보가 부족한 객체는 회색 점으로 표시
                plt.scatter(obj['pos_x'], obj['pos_z'], color='gray', s=30, alpha=0.5)
                continue
            
            # 모델 입력 데이터 세팅 
            pix_input = history[['x_pix', 'y_pix', 'w_pix', 'h_pix', 'pos_z']].values
            real_3d = history[['pos_x', 'pos_y', 'pos_z']].values
            
            # 추론
            ttc, status = engine.predict(pix_input, real_3d)
            
            # 위험도에 따른 색상 지정
            color = 'green'
            if status == "Danger": color = 'red'
            elif status == "Warning": color = 'orange'
            
            # 레이더망에 그리기
            plt.plot(history['pos_x'], history['pos_z'], color='blue', alpha=0.4, linewidth=2)
            plt.scatter(obj['pos_x'], obj['pos_z'], color=color, s=150, edgecolors='black')
            plt.text(obj['pos_x'] + 0.3, obj['pos_z'] + 0.3, f"ID:{int(tid)} TTC:{ttc:.1f}s", fontsize=9, fontweight='bold')

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel("X Position (meters)")
        plt.ylabel("Z Position (Depth meters)")
        
        # 이미지 저장
        save_path = f"{output_dir}/frame_{int(frame_idx):06d}.png"
        plt.savefig(save_path)
        plt.close()
        
    print(f"\n✅ 렌더링 완료! 레이더망 이미지들이 저장되었습니다: {output_dir}")

if __name__ == "__main__":
    process_sgan_sequence()