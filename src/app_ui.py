import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys
import time
import glob
from datetime import datetime, timezone, timedelta
from PIL import Image

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath("/workspace/minseok_park/"))
from src.utils.sgan_parser import SGANParser
from src.inference_engine import InferenceEngine

# --- 1. 페이지 설정 및 스타일 ---
st.set_page_config(page_title="경로예측 및 충돌예측 연구", layout="wide")
st.title("🛰️ 이미지 및 GPS데이터 기반의 이동체 경로예측 및 충돌예측 연구")

# --- KST (한국 표준시) 설정 ---
KST = timezone(timedelta(hours=9))

# --- 2. 사이드바: 설정 및 컨트롤 ---
st.sidebar.header("🛠️ System Settings")
ttc_threshold = st.sidebar.slider("Danger TTC Threshold (s)", 0.5, 3.0, 1.5, 0.1)
warning_threshold = st.sidebar.slider("Warning TTC Threshold (s)", 3.0, 5.0, 4.0, 0.5)

st.sidebar.markdown("---")
run_simulation = st.sidebar.button("▶️ Start Simulation", use_container_width=True)

# --- 3. 데이터 및 엔진 초기화 ---
@st.cache_resource
def init_system():
    parser = SGANParser(fps=2.5)
    engine = InferenceEngine()
    
    train_dir = "/workspace/minseok_park/data/sgan/datasets/eth/train/"
    txt_files = glob.glob(os.path.join(train_dir, "*.txt"))
    
    if not txt_files:
        st.error(f"❌ 에러: {train_dir} 경로에 텍스트 데이터 파일이 없습니다!")
        st.stop()
        
    data_path = txt_files[0]
    df = parser.parse_label(data_path)
    df = parser.calculate_gt_ttc(df)
    return parser, engine, df

parser, engine, full_df = init_system()

# --- 4. 메인 대시보드 레이아웃 (듀얼 뷰) ---
col_radar, col_cam = st.columns(2)

with col_radar:
    st.subheader("📡 Bird's Eye View (Radar)")
    radar_placeholder = st.empty()

with col_cam:
    st.subheader("📷 Camera View (Bounding Box & TTC)")
    cam_placeholder = st.empty()

st.markdown("---")

col_report, col_log = st.columns([1, 2])
with col_report:
    st.subheader("⚠️ Risk Analysis Report")
    alert_placeholder = st.empty()
with col_log:
    st.subheader("📝 Detection Log")
    log_placeholder = st.empty()

# --- 5. 시뮬레이션 로직 ---
if run_simulation:
    frames = sorted(full_df['frame'].unique())
    detection_logs = []

    for f_idx in frames:
        current_df = full_df[full_df['frame'] == f_idx]
        
        fig_radar = go.Figure()
        fig_cam = go.Figure()
        risk_peds = []

        # 🌟 실제 카메라 영상 이미지 로드 로직
        # 해당 프레임 번호의 사진(예: 0240.png)을 찾습니다.
        frame_img_path = f"/workspace/minseok_park/data/sgan/datasets/eth/frames/{int(f_idx):04d}.png" 
        
        if os.path.exists(frame_img_path):
            img = Image.open(frame_img_path)
            # 이미지가 존재하면 Plotly 차트 배경에 꽉 차게 깔아줍니다.
            fig_cam.add_layout_image(
                dict(
                    source=img, xref="paper", yref="paper",
                    x=0, y=1, sizex=1, sizey=1,
                    sizing="stretch", opacity=1, layer="below"
                )
            )

        for _, obj in current_df.iterrows():
            tid = int(obj['track_id'])
            history = full_df[(full_df['track_id'] == tid) & (full_df['frame'] <= f_idx)].tail(5)
            
            if len(history) < 5: continue
            
            pix_input = history[['x_pix', 'y_pix', 'w_pix', 'h_pix', 'pos_z']].values
            real_3d = history[['pos_x', 'pos_y', 'pos_z']].values
            ttc, _ = engine.predict(pix_input, real_3d)

            status = "Safe"
            color = "green"
            if ttc <= ttc_threshold:
                status = "Danger"
                color = "red"
                risk_peds.append({"ID": tid, "TTC": f"{ttc:.2f}s", "Status": "🚨 DANGER"})
            elif ttc <= warning_threshold:
                status = "Warning"
                color = "orange"

            # [1] 레이더 뷰 
            fig_radar.add_trace(go.Scatter(
                x=[obj['pos_x']], y=[obj['pos_z']],
                mode='markers+text', marker=dict(size=15, color=color, line=dict(width=2, color='black')),
                text=[f"ID:{tid}"], textposition="top center", name=status
            ))
            fig_radar.add_trace(go.Scatter(
                x=history['pos_x'], y=history['pos_z'],
                mode='lines', line=dict(color='blue', width=1), showlegend=False, opacity=0.3
            ))

            # [2] 카메라 뷰 (테두리 바운딩 박스)
            pseudo_x = obj['pos_x'] * 30
            pseudo_y = 100 - (obj['pos_z'] * 10)
            
            # 🌟 테두리만 있는 투명한 바운딩 박스
            fig_cam.add_shape(
                type="rect",
                x0=pseudo_x - 15, y0=pseudo_y - 25,
                x1=pseudo_x + 15, y1=pseudo_y + 25,
                line=dict(color=color, width=3),
                fillcolor="rgba(0,0,0,0)" # 안쪽을 투명하게!
            )
            
            # 머리 위 TTC 텍스트
            fig_cam.add_trace(go.Scatter(
                x=[pseudo_x], y=[pseudo_y + 35],
                mode='text', text=[f"ID:{tid} | {ttc:.1f}s"],
                textfont=dict(color=color, size=14, weight="bold"),
                showlegend=False
            ))

        # 레이아웃 업데이트
        fig_radar.update_layout(
            xaxis=dict(range=[-10, 15], title="X Position (m)"),
            yaxis=dict(range=[-5, 15], title="Z Position (m)"),
            height=500, margin=dict(l=0, r=0, b=0, t=0), showlegend=False
        )
        
        # 카메라 뷰 배경 (사진이 없을 때는 어두운 회색)
        fig_cam.update_layout(
            xaxis=dict(range=[-300, 400], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[-50, 200], showgrid=False, zeroline=False, visible=False),
            height=500, margin=dict(l=0, r=0, b=0, t=0), showlegend=False,
            plot_bgcolor="#1E1E1E" 
        )

        radar_placeholder.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{f_idx}")
        cam_placeholder.plotly_chart(fig_cam, use_container_width=True, key=f"cam_{f_idx}")

        # 리포트 및 로그 업데이트 (KST 적용)
        if risk_peds:
            alert_placeholder.error(f"⚠️ {len(risk_peds)}개의 이동체가 충돌 위험 궤적 내에 있습니다.")
        else:
            alert_placeholder.success("✅ 충돌 위험 없음.")

        if risk_peds:
            for p in risk_peds:
                current_kst = datetime.now(KST).strftime("%H:%M:%S")
                detection_logs.insert(0, {"Time": current_kst, **p})
            log_placeholder.dataframe(pd.DataFrame(detection_logs).head(10), use_container_width=True)

        time.sleep(0.1)