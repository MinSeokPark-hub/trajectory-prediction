import cv2
import os
import shutil
from utils.kitti_parser import KittiParser
from inference_engine import InferenceEngine

def process_sequence(seq_id, engine, parser, base_output_path):
    print(f"🚀 시퀀스 [{seq_id}] 분석 중...")
    label_path = f"/workspace/minseok_park/data/kitti/labels/{seq_id}.txt"
    image_dir = f"/workspace/minseok_park/data/kitti/images/{seq_id}/"
    seq_output_path = os.path.join(base_output_path, seq_id)
    
    if os.path.exists(seq_output_path): shutil.rmtree(seq_output_path)
    os.makedirs(seq_output_path)

    if not os.path.exists(image_dir): return

    processed_df = parser.calculate_gt_ttc(parser.parse_label(label_path))
    frames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    for frame_file in frames:
        frame_idx = int(frame_file.split('.')[0])
        img = cv2.imread(os.path.join(image_dir, frame_file))
        if img is None: continue

        current_objects = processed_df[processed_df['frame'] == frame_idx]
        
        for _, obj in current_objects.iterrows():
            tid = obj['track_id']
            history = processed_df[(processed_df['track_id'] == tid) & (processed_df['frame'] <= frame_idx)].tail(5)
            
            if len(history) < 5: continue
            
            # [모델별 데이터 분리]
            # SimpleTTC용: 픽셀+거리 정보 (5개 컬럼)
            pix_input = history[['x_pix', 'y_pix', 'w_pix', 'h_pix', 'pos_z']].values
            # LSTM용: 3D 물리 좌표 정보 (3개 컬럼)
            real_3d = history[['pos_x', 'pos_y', 'pos_z']].values
            
            ttc, status = engine.predict(pix_input, real_3d)
            
            # 박스 그리기
            w, h = obj['w_pix'], obj['h_pix']
            x1, y1 = int(obj['x_pix'] - w/2), int(obj['y_pix'] - h/2)
            color = (0, 0, 255) if status == "Danger" else (0, 255, 255) if status == "Warning" else (0, 255, 0)
            
            cv2.rectangle(img, (x1, y1), (int(x1+w), int(y1+h)), color, 2)
            cv2.putText(img, f"{obj['type']} {ttc:.1f}s", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

        cv2.imwrite(f"{seq_output_path}/frame_{frame_idx:06d}.jpg", img)

def main():
    engine = InferenceEngine()
    parser = KittiParser()
    for i in range(0, 12):
        process_sequence(f"{i:04d}", engine, parser, "/workspace/minseok_park/output")

if __name__ == "__main__":
    main()