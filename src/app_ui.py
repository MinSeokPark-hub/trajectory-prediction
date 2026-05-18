import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys
import time
import glob
import numpy as np
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

sys.path.append(os.path.abspath("/workspace/minseok_park/"))
from src.utils.sgan_parser import SGANParser
from src.utils.nuscenes_parser import NuScenesParser
from src.utils.kitti_parser import KittiParser
from src.inference_engine import InferenceEngine

st.set_page_config(page_title="경로예측 및 충돌예측 연구", layout="wide")

st.markdown("""
<style>
section[data-testid="stSidebar"] > div:first-child { background-color: #1a2035; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🛰️ 이미지 및 GPS 데이터 기반 이동체 경로 예측 및 충돌 예측 연구")
st.caption("Trajectory Prediction and Collision Prediction based on Image & GPS Data")

if 'resume_frame' not in st.session_state:
    st.session_state.resume_frame = None
if 'resume_scene' not in st.session_state:
    st.session_state.resume_scene = None

st.sidebar.header("🛠️ System Settings")
dataset_mode      = st.sidebar.radio("📂 데이터셋 선택", ["nuScenes", "ETH/UCY (SGAN)", "KITTI"])
ttc_threshold     = st.sidebar.slider("Danger TTC Threshold (s)",  0.5, 3.0, 1.5, 0.1)
warning_threshold = st.sidebar.slider("Warning TTC Threshold (s)", 3.0, 5.0, 4.0, 0.5)
frame_skip        = st.sidebar.slider("Frame Skip (빠를수록 ↑)", 1, 5, 1, 1)
sleep_time        = st.sidebar.slider("Frame Delay (s)", 0.0, 1.0, 0.3, 0.05)
st.sidebar.markdown("---")
if dataset_mode == "KITTI":
    KITTI_LABEL_DIR = "/workspace/minseok_park/data/kitti/labels"
    KITTI_IMAGE_DIR = "/workspace/minseok_park/data/kitti/images"
    kitti_sequences = sorted([f.replace('.txt','') for f in os.listdir(KITTI_LABEL_DIR) if f.endswith('.txt')])
    kitti_seq = st.sidebar.selectbox("📁 KITTI 시퀀스", kitti_sequences)
else:
    kitti_seq = None

col_btn1, col_btn2 = st.sidebar.columns(2)
run_simulation    = col_btn1.button("▶️ 시작", use_container_width=True)
stop_simulation   = col_btn2.button("⏹️ 정지", use_container_width=True)
resume_simulation = st.sidebar.button("⏩ 재개", use_container_width=True,
                                      disabled=st.session_state.resume_frame is None)

if run_simulation:
    st.session_state.resume_frame = None
    st.session_state.resume_scene = None

@st.cache_resource
def init_nuscenes():
    DATAROOT = "/workspace/minseok_park/data/nuscenes/v1.0-mini"
    engine = InferenceEngine()
    parser = NuScenesParser(dataroot=DATAROOT, version='v1.0-mini', fps=2.0)
    df = parser.load()
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    return engine, df, nusc, DATAROOT

@st.cache_resource
def init_kitti(seq):
    engine     = InferenceEngine()
    parser     = KittiParser(fps=10.0)
    label_path = f"/workspace/minseok_park/data/kitti/labels/{seq}.txt"
    df         = parser.load(label_path)
    return engine, df, seq

@st.cache_resource
def init_sgan():
    engine = InferenceEngine()
    parser = SGANParser(fps=2.5)
    train_dir = "/workspace/minseok_park/data/sgan/datasets/eth/train/"
    txt_files = glob.glob(os.path.join(train_dir, "*.txt"))
    if not txt_files:
        st.error(f"❌ {train_dir} 에 데이터 없음")
        st.stop()
    df = parser.load(txt_files[0])
    return engine, df

def get_color(ttc, ttc_threshold, warning_threshold):
    if ttc <= ttc_threshold:
        return (255, 0, 0), "red", "Danger"
    elif ttc <= warning_threshold:
        return (255, 165, 0), "orange", "Warning"
    return (0, 200, 0), "green", "Safe"

def project_to_image(nusc, sample_token, obj_translation, obj_size, obj_rotation):
    try:
        sample    = nusc.get('sample', sample_token)
        cam_token = sample['data']['CAM_FRONT']
        cam_data  = nusc.get('sample_data', cam_token)
        cs        = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose  = nusc.get('ego_pose', cam_data['ego_pose_token'])

        K         = np.array(cs['camera_intrinsic'])
        ego_rot   = Quaternion(ego_pose['rotation']).rotation_matrix
        ego_trans = np.array(ego_pose['translation'])
        cam_rot   = Quaternion(cs['rotation']).rotation_matrix
        cam_trans = np.array(cs['translation'])

        w, l, h    = obj_size[0], obj_size[1], obj_size[2]
        cx, cy, cz = obj_translation

        # 객체 로컬 좌표계 8개 꼭짓점 (중심 기준)
        half_l, half_w, half_h = l/2, w/2, h/2
        corners_local = np.array([
            [+half_l, +half_w, -half_h],
            [+half_l, -half_w, -half_h],
            [-half_l, +half_w, -half_h],
            [-half_l, -half_w, -half_h],
            [+half_l, +half_w, +half_h],
            [+half_l, -half_w, +half_h],
            [-half_l, +half_w, +half_h],
            [-half_l, -half_w, +half_h],
        ]).T  # (3, 8)

        # 객체 회전 적용 후 월드 좌표로 이동
        obj_rot_matrix  = Quaternion(obj_rotation).rotation_matrix
        corners_rotated = obj_rot_matrix @ corners_local  # (3, 8)
        corners         = corners_rotated.T + np.array([cx, cy, cz])  # (8, 3)

        px_list, py_list = [], []
        for corner in corners:
            p_ego = ego_rot.T @ (corner - ego_trans)
            p_cam = cam_rot.T @ (p_ego  - cam_trans)
            if p_cam[2] < 0.5:
                continue
            p_img = K @ p_cam
            px_list.append(int(p_img[0] / p_img[2]))
            py_list.append(int(p_img[1] / p_img[2]))

        if not px_list:
            return None

        # 실제 이미지 크기 가져오기
        img_path = os.path.join(nusc.dataroot, cam_data['filename'])
        img      = Image.open(img_path)
        img_w, img_h = img.size

        x1 = max(0,     min(px_list))
        y1 = max(0,     min(py_list))
        x2 = min(img_w, max(px_list))
        y2 = min(img_h, max(py_list))

        if x2 - x1 < 5 or y2 - y1 < 5:
            return None

        return x1, y1, x2, y2

    except Exception:
        return None

def get_cam_image_path(nusc, sample_token, dataroot):
    try:
        sample    = nusc.get('sample', sample_token)
        cam_token = sample['data']['CAM_FRONT']
        cam_data  = nusc.get('sample_data', cam_token)
        return os.path.join(dataroot, cam_data['filename'])
    except Exception:
        return None

def draw_boxes_on_image(img_path, boxes):
    img  = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2, label, color) in boxes:
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, max(0, y1-20), x1+len(label)*8, y1], fill=color)
        draw.text((x1+2, max(2, y1-18)), label, fill=(255, 255, 255))
    return img

if dataset_mode == "nuScenes":
    engine, full_df, nusc, DATAROOT = init_nuscenes()
elif dataset_mode == "KITTI":
    engine, full_df, kitti_seq = init_kitti(kitti_seq)
else:
    engine, full_df = init_sgan()

def _live_header(title):
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
        f'<span style="font-size:17px;font-weight:600">{title}</span>'
        f'<span style="background:#2ecc71;color:white;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:bold">LIVE</span>'
        f'</div>',
        unsafe_allow_html=True
    )

if dataset_mode == "nuScenes":
    col_map, col_cam = st.columns(2)
    with col_map:
        _live_header("🗺️ GPS Map View (자차 중심)")
        map_placeholder = st.empty()
    with col_cam:
        _live_header("📷 Camera View (CAM_FRONT)")
        cam_placeholder = st.empty()
        cam_meta_placeholder = st.empty()
elif dataset_mode == "KITTI":
    col_radar, col_cam = st.columns(2)
    with col_radar:
        _live_header("📡 Bird's Eye View (KITTI)")
        map_placeholder = st.empty()
    with col_cam:
        _live_header("📷 Camera View (KITTI)")
        cam_placeholder = st.empty()
        cam_meta_placeholder = st.empty()
else:
    col_radar, col_cam = st.columns(2)
    with col_radar:
        _live_header("📡 Bird's Eye View (Radar)")
        map_placeholder = st.empty()
    with col_cam:
        _live_header("📷 Camera View")
        cam_placeholder = st.empty()
        cam_meta_placeholder = st.empty()

st.markdown("---")
_s1, _s2, _s3, _s4, _s5 = st.columns(5)
with _s1:
    ttc_ph = st.empty()
with _s2:
    danger_ph = st.empty()
with _s3:
    count_ph = st.empty()
with _s4:
    rttc_ph = st.empty()
with _s5:
    dataset_ph = st.empty()
ttc_ph.metric("🛡️ TTC (최소)", "—")
danger_ph.metric("⚠️ 위험 수준", "—")
count_ph.metric("🚗 충돌 객체", "—")
rttc_ph.metric("⏱️ 예상 재재 TTC", "—")
dataset_ph.metric("🗄️ 데이터셋", dataset_mode)

if run_simulation or resume_simulation:
    if dataset_mode == "nuScenes":
        # scene+frame → sample_token 매핑
        scene_frame_tokens = {}
        for scene in nusc.scene:
            token = scene['first_sample_token']
            f = 0
            while token:
                scene_frame_tokens[(scene['name'], f)] = token
                sample = nusc.get('sample', token)
                token  = sample['next']
                f += 1

        _resume_scene = st.session_state.resume_scene if resume_simulation else None
        _resume_frame = st.session_state.resume_frame if resume_simulation else None

        # scene 순서대로, 각 scene의 프레임 순서대로 순회
        scenes = full_df['scene'].unique()
        for scene_name in scenes:
            if _resume_scene and scene_name < _resume_scene:
                continue

            scene_df = full_df[full_df['scene'] == scene_name]
            frames   = sorted(scene_df['frame'].unique())[::frame_skip]

            for f_idx in frames:
                if _resume_frame is not None and scene_name == _resume_scene and f_idx < _resume_frame:
                    continue

                st.session_state.resume_frame = f_idx
                st.session_state.resume_scene = scene_name

                current_df = scene_df[scene_df['frame'] == f_idx].copy()
                if current_df.empty:
                    continue

                current_df   = current_df.nsmallest(10, 'depth')
                fig_map      = go.Figure()
                risk_objs    = []
                min_ttc_frame = 10.0
                ego_x        = current_df.iloc[0]['ego_x']
                ego_y        = current_df.iloc[0]['ego_y']
                sample_token = scene_frame_tokens.get((scene_name, int(f_idx)), None)
                img_path     = get_cam_image_path(nusc, sample_token, DATAROOT) if sample_token else None
                boxes_to_draw = []

                fig_map.add_trace(go.Scatter(
                    x=[0], y=[0],
                    mode='markers+text',
                    marker=dict(size=18, color='blue', symbol='circle'),
                    text=["🚗 EGO"], textposition="top center", name="자차"
                ))

                for _, obj in current_df.iterrows():
                    tid     = obj['track_id']
                    history = full_df[
                        (full_df['track_id'] == tid) &
                        (full_df['scene']    == scene_name) &
                        (full_df['frame']    <= f_idx)
                    ].tail(5)

                    if len(history) < 2:
                        continue

                    rel_x = float(obj['pos_x'])
                    rel_y = float(obj['pos_y'])
                    depth = float(obj['depth'])
                    vel   = float(obj['velocity']) if pd.notna(obj.get('velocity')) else 0.0
                    ttc   = depth / vel if vel > 0.1 else 10.0
                    ttc   = float(min(max(ttc, 0.1), 10.0))
                    min_ttc_frame = min(min_ttc_frame, ttc)

                    color_rgb, color_str, status = get_color(ttc, ttc_threshold, warning_threshold)
                    if status == "Danger":
                        risk_objs.append({"ID": tid, "TTC": f"{ttc:.2f}s", "Status": "🚨 DANGER"})

                    fig_map.add_trace(go.Scatter(
                        x=[rel_x], y=[rel_y],
                        mode='markers+text',
                        marker=dict(size=12, color=color_str,
                                    line=dict(width=1, color='black')),
                        text=[f"{obj['type']}<br>{ttc:.1f}s"],
                        textposition="top center", name=status
                    ))
                    fig_map.add_trace(go.Scatter(
                        x=history['pos_x'].values,
                        y=history['pos_y'].values,
                        mode='lines', line=dict(color='gray', width=1),
                        showlegend=False, opacity=0.5
                    ))

                    if sample_token and img_path and os.path.exists(img_path):
                        obj_translation = [
                            float(obj['abs_x']),
                            float(obj['abs_y']),
                            float(obj['obj_z']),
                        ]
                        obj_size = [
                            float(obj['w']),
                            float(obj['obj_l']),
                            float(obj['h']),
                        ]
                        obj_rotation = [
                            float(obj['rot_w']),
                            float(obj['rot_x']),
                            float(obj['rot_y']),
                            float(obj['rot_z']),
                        ]
                        result = project_to_image(nusc, sample_token, obj_translation, obj_size, obj_rotation)
                        if result:
                            x1, y1, x2, y2 = result
                            label = f"{obj['type']} {ttc:.1f}s"
                            boxes_to_draw.append((x1, y1, x2, y2, label, color_rgb))

                fig_map.update_layout(
                    xaxis=dict(range=[-15, 15], title="X (m, 자차 기준)"),
                    yaxis=dict(range=[-5,  15], title="Y (m, 자차 기준)", scaleanchor="x"),
                    height=500, margin=dict(l=0, r=0, b=0, t=0),
                    showlegend=False, plot_bgcolor="#e8f4e8"
                )

                if img_path and os.path.exists(img_path):
                    rendered_img = draw_boxes_on_image(img_path, boxes_to_draw)
                    cam_placeholder.image(rendered_img, use_container_width=True)
                else:
                    cam_placeholder.info("카메라 이미지 없음")

                map_placeholder.plotly_chart(fig_map, use_container_width=True, key=f"map_{scene_name}_{f_idx}")

                if min_ttc_frame <= ttc_threshold:
                    frame_status = "🚨 DANGER"
                elif min_ttc_frame <= warning_threshold:
                    frame_status = "⚠️ WARNING"
                else:
                    frame_status = "✅ SAFE"
                ttc_str = f"{min_ttc_frame:.2f} s" if min_ttc_frame < 10.0 else "—"
                ttc_ph.metric("🛡️ TTC (최소)", ttc_str)
                danger_ph.metric("⚠️ 위험 수준", frame_status)
                count_ph.metric("🚗 충돌 객체", len(risk_objs))
                rttc_ph.metric("⏱️ 예상 재재 TTC", ttc_str)
                cam_meta_placeholder.markdown(
                    '<div style="color:#888;font-size:12px;margin-top:4px">'
                    '📷 CAM_FRONT &nbsp;|&nbsp; 🌙 Night &nbsp;|&nbsp; 📐 1600 × 900 &nbsp;|&nbsp; 🎞 2 FPS'
                    '</div>', unsafe_allow_html=True
                )

                if stop_simulation:
                    st.warning("⏹️ 시뮬레이션이 정지되었습니다.")
                    break

                time.sleep(sleep_time)

            if stop_simulation:
                break

    elif dataset_mode == "KITTI":
        _resume_frame = st.session_state.resume_frame if resume_simulation else None
        frames = sorted(full_df['frame'].unique())[::frame_skip]
        for f_idx in frames:
            if _resume_frame is not None and f_idx < _resume_frame:
                continue

            st.session_state.resume_frame = f_idx
            st.session_state.resume_scene = None

            current_df = full_df[full_df['frame'] == f_idx].copy()
            if current_df.empty:
                continue

            current_df = current_df.nsmallest(10, 'depth')
            fig_map    = go.Figure()
            risk_objs  = []
            min_ttc_frame = 10.0
            boxes_to_draw = []

            img_path = f"/workspace/minseok_park/data/kitti/images/{kitti_seq}/{int(f_idx):06d}.png"

            for _, obj in current_df.iterrows():
                tid     = int(obj['track_id'])
                history = full_df[
                    (full_df['track_id'] == tid) &
                    (full_df['frame']    <= f_idx)
                ].tail(5)
                if len(history) < 2:
                    continue

                depth = float(obj['depth'])
                vel   = float(obj['velocity']) if pd.notna(obj.get('velocity')) else 0.0
                ttc   = depth / vel if vel > 0.1 else 10.0
                ttc   = float(min(max(ttc, 0.1), 10.0))
                min_ttc_frame = min(min_ttc_frame, ttc)

                color_rgb, color_str, status = get_color(ttc, ttc_threshold, warning_threshold)
                if status == "Danger":
                    risk_objs.append({"ID": tid, "TTC": f"{ttc:.2f}s", "Status": "🚨 DANGER"})

                fig_map.add_trace(go.Scatter(
                    x=[float(obj['pos_x'])], y=[float(obj['pos_z'])],
                    mode='markers+text',
                    marker=dict(size=15, color=color_str, line=dict(width=2, color='black')),
                    text=[f"{obj['type']}<br>{ttc:.1f}s"], textposition="top center", name=status
                ))
                fig_map.add_trace(go.Scatter(
                    x=history['pos_x'].values, y=history['pos_z'].values,
                    mode='lines', line=dict(color='gray', width=1),
                    showlegend=False, opacity=0.4
                ))

                x1 = int(obj['x_pix'] - obj['w_pix'] / 2)
                y1 = int(obj['y_pix'] - obj['h_pix'] / 2)
                x2 = int(obj['x_pix'] + obj['w_pix'] / 2)
                y2 = int(obj['y_pix'] + obj['h_pix'] / 2)
                label = f"{obj['type']} {ttc:.1f}s"
                boxes_to_draw.append((x1, y1, x2, y2, label, color_rgb))

            fig_map.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers+text',
                marker=dict(size=18, color='blue', symbol='circle'),
                text=["🚗 EGO"], textposition="top center", name="자차"
            ))
            fig_map.update_layout(
                xaxis=dict(range=[-15, 15], title="X (m)"),
                yaxis=dict(range=[-2,  50], title="Z/Depth (m)"),
                height=500, margin=dict(l=0, r=0, b=0, t=0),
                showlegend=False, plot_bgcolor="#e8f4e8"
            )
            map_placeholder.plotly_chart(fig_map, use_container_width=True, key=f"kitti_map_{f_idx}")

            if os.path.exists(img_path):
                rendered = draw_boxes_on_image(img_path, boxes_to_draw)
                cam_placeholder.image(rendered, use_container_width=True)
            else:
                cam_placeholder.info(f"이미지 없음: {img_path}")

            if min_ttc_frame <= ttc_threshold:
                frame_status = "🚨 DANGER"
            elif min_ttc_frame <= warning_threshold:
                frame_status = "⚠️ WARNING"
            else:
                frame_status = "✅ SAFE"
            ttc_str = f"{min_ttc_frame:.2f} s" if min_ttc_frame < 10.0 else "—"
            ttc_ph.metric("🛡️ TTC (최소)", ttc_str)
            danger_ph.metric("⚠️ 위험 수준", frame_status)
            count_ph.metric("🚗 충돌 객체", len(risk_objs))
            rttc_ph.metric("⏱️ 예상 재재 TTC", ttc_str)
            cam_meta_placeholder.markdown(
                f'<div style="color:#888;font-size:12px;margin-top:4px">'
                f'📷 KITTI &nbsp;|&nbsp; Seq: {kitti_seq} &nbsp;|&nbsp; 🎞 10 FPS'
                f'</div>', unsafe_allow_html=True
            )

            if stop_simulation:
                st.warning("⏹️ 시뮬레이션이 정지되었습니다.")
                break

            time.sleep(sleep_time)

    else:
        _resume_frame = st.session_state.resume_frame if resume_simulation else None
        frames = sorted(full_df['frame'].unique())[::frame_skip]
        for f_idx in frames:
            if _resume_frame is not None and f_idx < _resume_frame:
                continue

            st.session_state.resume_frame = f_idx
            st.session_state.resume_scene = None

            current_df = full_df[full_df['frame'] == f_idx].copy()
            if current_df.empty:
                continue

            current_df = current_df.nsmallest(10, 'depth')
            fig_map    = go.Figure()
            fig_cam    = go.Figure()
            risk_objs  = []
            min_ttc_frame = 10.0

            frame_img_path = f"/workspace/minseok_park/data/sgan/datasets/eth/frames/{int(f_idx):04d}.png"
            if os.path.exists(frame_img_path):
                img = Image.open(frame_img_path)
                fig_cam.add_layout_image(dict(
                    source=img, xref="paper", yref="paper",
                    x=0, y=1, sizex=1, sizey=1,
                    sizing="stretch", opacity=1, layer="below"
                ))

            for _, obj in current_df.iterrows():
                tid = int(obj['track_id'])
                history = full_df[
                    (full_df['track_id'] == tid) &
                    (full_df['frame'] <= f_idx)
                ].tail(5)
                if len(history) < 5:
                    continue

                pix_input = history[['x_pix', 'y_pix', 'w_pix', 'h_pix', 'pos_z']].values
                real_3d   = history[['pos_x', 'pos_y', 'pos_z']].values
                ttc, _    = engine.predict(pix_input, real_3d)

                _, color_str, status = get_color(ttc, ttc_threshold, warning_threshold)
                min_ttc_frame = min(min_ttc_frame, ttc)
                if status == "Danger":
                    risk_objs.append({"ID": tid, "TTC": f"{ttc:.2f}s", "Status": "🚨 DANGER"})

                fig_map.add_trace(go.Scatter(
                    x=[obj['pos_x']], y=[obj['pos_z']],
                    mode='markers+text',
                    marker=dict(size=15, color=color_str,
                                line=dict(width=2, color='black')),
                    text=[f"ID:{tid}"], textposition="top center", name=status
                ))
                fig_map.add_trace(go.Scatter(
                    x=history['pos_x'], y=history['pos_z'],
                    mode='lines', line=dict(color='blue', width=1),
                    showlegend=False, opacity=0.3
                ))

                pseudo_x = obj['pos_x'] * 30
                pseudo_y = 100 - (obj['pos_z'] * 10)
                fig_cam.add_shape(
                    type="rect",
                    x0=pseudo_x-15, y0=pseudo_y-25,
                    x1=pseudo_x+15, y1=pseudo_y+25,
                    line=dict(color=color_str, width=3),
                    fillcolor="rgba(0,0,0,0)"
                )
                fig_cam.add_trace(go.Scatter(
                    x=[pseudo_x], y=[pseudo_y+35],
                    mode='text', text=[f"ID:{tid} | {ttc:.1f}s"],
                    textfont=dict(color=color_str, size=14), showlegend=False
                ))

            fig_map.update_layout(
                xaxis=dict(range=[-10, 15], title="X (m)"),
                yaxis=dict(range=[-5,  15], title="Z (m)"),
                height=500, margin=dict(l=0, r=0, b=0, t=0), showlegend=False
            )
            fig_cam.update_layout(
                xaxis=dict(range=[-300, 400], showgrid=False, zeroline=False, visible=False),
                yaxis=dict(range=[-50,  200], showgrid=False, zeroline=False, visible=False),
                height=500, margin=dict(l=0, r=0, b=0, t=0),
                showlegend=False, plot_bgcolor="#1E1E1E"
            )

            map_placeholder.plotly_chart(fig_map, use_container_width=True, key=f"map_{f_idx}")
            cam_placeholder.plotly_chart(fig_cam, use_container_width=True, key=f"cam_{f_idx}")

            if min_ttc_frame <= ttc_threshold:
                frame_status = "🚨 DANGER"
            elif min_ttc_frame <= warning_threshold:
                frame_status = "⚠️ WARNING"
            else:
                frame_status = "✅ SAFE"
            ttc_str = f"{min_ttc_frame:.2f} s" if min_ttc_frame < 10.0 else "—"
            ttc_ph.metric("🛡️ TTC (최소)", ttc_str)
            danger_ph.metric("⚠️ 위험 수준", frame_status)
            count_ph.metric("🚗 충돌 객체", len(risk_objs))
            rttc_ph.metric("⏱️ 예상 재재 TTC", ttc_str)
            cam_meta_placeholder.markdown(
                '<div style="color:#888;font-size:12px;margin-top:4px">'
                '📷 ETH/UCY &nbsp;|&nbsp; 🎞 2.5 FPS'
                '</div>', unsafe_allow_html=True
            )

            if stop_simulation:
                st.warning("⏹️ 시뮬레이션이 정지되었습니다.")
                break

            time.sleep(sleep_time)