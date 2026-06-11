import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys
import time
import glob
import math
import numpy as np
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

sys.path.append(os.path.abspath("/workspace/minseok_park/"))
from src.utils.sgan_parser import SGANParser
from src.utils.nuscenes_parser import NuScenesParser
from src.utils.kitti_parser import KittiParser
from src.inference_engine import InferenceEngine
from src.utils.occlusion_handler import OcclusionHandler
from src.utils.async_worker import AsyncPredictionWorker

st.set_page_config(page_title="경로예측 및 충돌예측 연구", layout="wide")

st.markdown("""
<style>
section[data-testid="stSidebar"] > div:first-child { background-color: #1a2035; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
section[data-testid="stSidebar"] .stButton button {
    background-color: #2563eb;
    color: #ffffff !important;
    border: none;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background-color: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🛰️ 이미지 및 GPS 데이터 기반 이동체 경로 예측 및 충돌 예측 연구")
st.caption("Trajectory Prediction and Collision Prediction based on Image & GPS Data")

if 'resume_frame' not in st.session_state:
    st.session_state.resume_frame = None
if 'resume_scene' not in st.session_state:
    st.session_state.resume_scene = None
if 'ade_history' not in st.session_state:
    st.session_state.ade_history = []
if 'occlusion_handler' not in st.session_state:
    st.session_state.occlusion_handler = None
if 'async_worker' not in st.session_state:
    st.session_state.async_worker = None

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

st.sidebar.header("🛠️ System Settings")
dataset_mode      = st.sidebar.radio("📂 데이터셋 선택", ["nuScenes", "ETH/UCY (SGAN)", "KITTI"])
ttc_threshold     = st.sidebar.slider("Danger TTC Threshold (s)",  0.5, 3.0, 1.5, 0.1)
warning_threshold = st.sidebar.slider("Warning TTC Threshold (s)", 3.0, 5.0, 4.0, 0.5)
frame_skip        = st.sidebar.slider("Frame Skip (빠를수록 ↑)", 1, 5, 1, 1)
sleep_time        = st.sidebar.slider("Frame Delay (s)", 0.0, 1.0, 0.3, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**🗺️ GPS 뷰 설정**")
gps_fixed = st.sidebar.checkbox("📍 자차 중심 고정", value=False)
gps_zoom  = st.sidebar.slider("줌 범위 (m)", 5, 50, 20, 5) if gps_fixed else 20
st.sidebar.markdown("---")
st.sidebar.markdown("**🧠 Sprint 2 기능**")
show_heatmap = st.sidebar.checkbox("🔥 어텐션 히트맵", value=True)
show_ade     = st.sidebar.checkbox("📊 ADE/FDE 비교", value=True)
show_occlusion = st.sidebar.checkbox("👁️ Occlusion 추적", value=True)
st.sidebar.markdown("---")
if dataset_mode == "nuScenes":
    kitti_seq = None
    _, _ns_df, _, _ = init_nuscenes()
    nuscenes_scenes = sorted(_ns_df['scene'].unique().tolist())
    nuscenes_scene = st.sidebar.selectbox("📁 nuScenes 씬", nuscenes_scenes)
elif dataset_mode == "KITTI":
    nuscenes_scene = None
    KITTI_LABEL_DIR = "/workspace/minseok_park/data/kitti/labels"
    KITTI_IMAGE_DIR = "/workspace/minseok_park/data/kitti/images"
    kitti_sequences = sorted([f.replace('.txt','') for f in os.listdir(KITTI_LABEL_DIR) if f.endswith('.txt')])
    kitti_seq = st.sidebar.selectbox("📁 KITTI 시퀀스", kitti_sequences)
else:
    kitti_seq = None
    nuscenes_scene = None

col_btn1, col_btn2 = st.sidebar.columns(2)
run_simulation    = col_btn1.button("▶️ 시작", use_container_width=True)
stop_simulation   = col_btn2.button("⏹️ 정지", use_container_width=True)
resume_simulation = st.sidebar.button("⏩ 재개", use_container_width=True,
                                      disabled=st.session_state.resume_frame is None)

if run_simulation:
    st.session_state.resume_frame = None
    st.session_state.resume_scene = None
    st.session_state.ade_history = []
    _fps = 2.0 if dataset_mode == "nuScenes" else (2.5 if "SGAN" in dataset_mode else 10.0)
    st.session_state.occlusion_handler = OcclusionHandler(fps=_fps)

def get_color(ttc, ttc_threshold, warning_threshold):
    if ttc <= ttc_threshold:
        return (255, 0, 0), "red", "Danger"
    elif ttc <= warning_threshold:
        return (255, 165, 0), "orange", "Warning"
    return (0, 200, 0), "green", "Safe"

def compute_risk_score(ttc):
    return float(np.clip((10.0 - ttc) / 10.0, 0.0, 1.0))

def get_risk_level(risk_score):
    if risk_score >= 0.7:
        return 'High'
    elif risk_score >= 0.4:
        return 'Medium'
    return 'Low'

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
    w, h = img.size
    img = img.resize((w // 2, h // 2))
    return img

if dataset_mode == "nuScenes":
    engine, full_df, nusc, DATAROOT = init_nuscenes()
elif dataset_mode == "KITTI":
    engine, full_df, kitti_seq = init_kitti(kitti_seq)
else:
    engine, full_df = init_sgan()

if run_simulation:
    if st.session_state.async_worker is not None:
        st.session_state.async_worker.stop()
    st.session_state.async_worker = AsyncPredictionWorker(engine.predict_scene, n_workers=2)
    st.session_state.async_worker.start()

def _live_header(title):
    st.markdown(
        f'<div style="font-size:17px;font-weight:600;margin-bottom:6px">{title}</div>',
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
_cd, _cw, _cs = st.columns(3)
with _cd:
    danger_ph = st.empty()
with _cw:
    warning_ph = st.empty()
with _cs:
    safe_ph = st.empty()

def _render_status(danger_list, warning_list, safe_cnt):
    def ttc_tags(lst):
        if not lst:
            return "<span style='color:#aaa'>—</span>"
        return " &nbsp; ".join(f"<code>{t:.1f}s</code>" for t in sorted(lst))

    danger_ph.markdown(
        f'<div style="background:#fee2e2;border-radius:10px;padding:14px;text-align:center">'
        f'<div style="font-size:16px;font-weight:700;color:#dc2626">🚨 DANGER &nbsp; {len(danger_list)}개</div>'
        f'<div style="font-size:13px;margin-top:6px">TTC: {ttc_tags(danger_list)}</div></div>',
        unsafe_allow_html=True
    )
    warning_ph.markdown(
        f'<div style="background:#fef9c3;border-radius:10px;padding:14px;text-align:center">'
        f'<div style="font-size:16px;font-weight:700;color:#ca8a04">⚠️ WARNING &nbsp; {len(warning_list)}개</div>'
        f'<div style="font-size:13px;margin-top:6px">TTC: {ttc_tags(warning_list)}</div></div>',
        unsafe_allow_html=True
    )
    safe_ph.markdown(
        f'<div style="background:#dcfce7;border-radius:10px;padding:14px;text-align:center">'
        f'<div style="font-size:16px;font-weight:700;color:#16a34a">✅ SAFE &nbsp; {safe_cnt}개</div></div>',
        unsafe_allow_html=True
    )

_render_status([], [], 0)

KST = timezone(timedelta(hours=9))

# ── Sprint 2 헬퍼 함수 ────────────────────────────────────────────────────────

_RISK_COLOR = {'high': '#ef4444', 'mid': '#f97316', 'low': '#3b82f6'}

def _build_scene_objects(current_df, full_df, f_idx, scene_name=None, fps=10.0):
    """프레임 내 객체를 social_attention.compute() 입력 형식으로 변환"""
    dt = 1.0 / fps
    objects = []
    for _, obj in current_df.iterrows():
        tid = obj['track_id']
        mask = (full_df['track_id'] == tid) & (full_df['frame'] <= f_idx)
        if scene_name is not None:
            mask &= (full_df['scene'] == scene_name)
        history = full_df[mask].tail(5)
        if len(history) < 2:
            continue
        vel_x = float(history['pos_x'].iloc[-1] - history['pos_x'].iloc[-2]) / dt
        # nuScenes: pos_y가 전방, KITTI: pos_z가 전방
        fwd_col = 'pos_y' if 'pos_y' in history.columns else 'pos_z'
        rel_y  = float(obj[fwd_col]) if fwd_col in obj.index else float(obj['depth'])
        vel_y  = float(history[fwd_col].iloc[-1] - history[fwd_col].iloc[-2]) / dt if fwd_col in history.columns else 0.0
        # LSTM용 3D 이력: [x, forward, depth] — predict_scene()의 history_3d 포맷
        hist_3d_cols = ['pos_x', fwd_col, 'depth']
        history_3d = history[hist_3d_cols].values.astype(float) if all(c in history.columns for c in hist_3d_cols) else None
        objects.append({
            'track_id':   str(tid),
            'rel_x':      float(obj['pos_x']),
            'rel_y':      rel_y,
            'vel_x':      vel_x,
            'vel_y':      vel_y,
            'depth':      float(obj['depth']),
            'history_3d': history_3d,
        })
    return objects

def _render_heatmap(social_result, placeholder):
    """US-09 T6-1: 어텐션 가중치 히트맵"""
    if placeholder is None or not social_result.get('track_ids'):
        return
    weights  = social_result['attention_weights']
    risk_lvl = social_result['risk_levels']
    tids     = social_result['track_ids']
    evasion  = social_result['evasion_flags']

    fig = go.Figure()
    for i, tid in enumerate(tids):
        w     = float(weights[i])
        color = _RISK_COLOR.get(risk_lvl[i], '#6b7280')
        label = f"ID:{tid}<br>w={w:.3f}<br>{risk_lvl[i].upper()}"
        if evasion[i]:
            label += "<br>⚠️회피"
        fig.add_trace(go.Scatter(
            x=[i], y=[w],
            mode='markers+text',
            marker=dict(size=max(20, int(w * 300)), color=color, opacity=0.8,
                        line=dict(width=2, color='white')),
            text=[label], textposition='top center',
            name=tid, showlegend=False,
        ))

    fig.add_trace(go.Bar(
        x=list(range(len(tids))),
        y=weights.tolist(),
        marker_color=[_RISK_COLOR.get(r, '#6b7280') for r in risk_lvl],
        opacity=0.35, showlegend=False,
    ))
    fig.update_layout(
        xaxis=dict(tickvals=list(range(len(tids))),
                   ticktext=[f"ID:{t}" for t in tids], title="객체"),
        yaxis=dict(title="어텐션 가중치", range=[0, max(weights.tolist() + [0.1]) * 1.4]),
        height=300, margin=dict(l=10, r=10, t=10, b=40),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font=dict(color='white'),
    )
    placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def _render_ade_chart(ade_history, placeholder):
    """US-13 T14: Social Attention 전/후 ADE/FDE 비교"""
    if placeholder is None or len(ade_history) < 1:
        return
    frames      = list(range(len(ade_history)))
    ade_no  = [h['no_social']          for h in ade_history]
    ade_yes = [h['social']             for h in ade_history]
    fde_no  = [h.get('fde_no_social', 0) for h in ade_history]
    fde_yes = [h.get('fde_social', 0)    for h in ade_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames, y=ade_no, mode='lines', name='ADE 미적용',
        line=dict(color='#94a3b8', width=2, dash='dot'),
    ))
    fig.add_trace(go.Scatter(
        x=frames, y=ade_yes, mode='lines+markers', name='ADE 적용',
        line=dict(color='#22d3ee', width=2),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=frames, y=fde_no, mode='lines', name='FDE 미적용',
        line=dict(color='#f97316', width=2, dash='dot'),
    ))
    fig.add_trace(go.Scatter(
        x=frames, y=fde_yes, mode='lines+markers', name='FDE 적용',
        line=dict(color='#a78bfa', width=2),
        marker=dict(size=4),
    ))
    avg_ade_no  = float(np.mean(ade_no))  if ade_no  else 0
    avg_ade_yes = float(np.mean(ade_yes)) if ade_yes else 0
    avg_fde_no  = float(np.mean(fde_no))  if fde_no  else 0
    avg_fde_yes = float(np.mean(fde_yes)) if fde_yes else 0
    ade_imp = (avg_ade_no - avg_ade_yes) / (avg_ade_no + 1e-9) * 100
    fde_imp = (avg_fde_no - avg_fde_yes) / (avg_fde_no + 1e-9) * 100
    fig.update_layout(
        title=dict(
            text=(f"ADE 미적용:{avg_ade_no:.3f} 적용:{avg_ade_yes:.3f} ({ade_imp:+.1f}%)  |  "
                  f"FDE 미적용:{avg_fde_no:.3f} 적용:{avg_fde_yes:.3f} ({fde_imp:+.1f}%)"),
            font=dict(size=11, color='white'),
        ),
        xaxis=dict(title="프레임"),
        yaxis=dict(title="ADE (대리 지표)"),
        height=300, margin=dict(l=10, r=10, t=40, b=40),
        legend=dict(orientation='h', y=1.15),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font=dict(color='white'),
    )
    placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def _render_occlusion_info(occlusion_handler, placeholder):
    """US-11: Occlusion 상태 인디케이터"""
    if placeholder is None or occlusion_handler is None:
        return
    stats = occlusion_handler.stats()
    occ_ids = stats['occluded_ids']
    color = "#fef9c3" if occ_ids else "#dcfce7"
    text_color = "#92400e" if occ_ids else "#166534"
    ids_str = ", ".join(str(i) for i in occ_ids) if occ_ids else "없음"
    placeholder.markdown(
        f'<div style="background:{color};border-radius:8px;padding:8px 14px;'
        f'font-size:13px;color:{text_color}">'
        f'👁️ <b>Occlusion 추적</b> — 가림 객체 {stats["occluded_count"]}개 '
        f'(전체 추적 {stats["total_tracked"]}개) &nbsp;|&nbsp; ID: {ids_str}</div>',
        unsafe_allow_html=True,
    )

def _render_risk_heatmap(risk_rows, placeholder):
    """위험도 히트맵: 객체 위치 × 위험도를 Reds 색상으로 시각화"""
    if placeholder is None or not risk_rows:
        return
    depths = [float(r['Depth(m)']) for r in risk_rows]
    scores = [float(r['Risk Score']) for r in risk_rows]
    ids    = [r['ID'] for r in risk_rows]
    levels = [r['Level'] for r in risk_rows]
    fig = go.Figure(go.Scatter(
        x=list(range(len(risk_rows))),
        y=depths,
        mode='markers+text',
        marker=dict(
            size=[max(10, int(s * 40)) for s in scores],
            color=scores, colorscale='Reds', showscale=True,
            colorbar=dict(title='Risk', thickness=12),
            cmin=0, cmax=1, line=dict(width=1, color='white'),
        ),
        text=[f"ID:{i}<br>{lv}" for i, lv in zip(ids, levels)],
        textposition='top center',
    ))
    fig.update_layout(
        xaxis=dict(tickvals=list(range(len(risk_rows))),
                   ticktext=[str(r['ID']) for r in risk_rows], title="객체 ID"),
        yaxis=dict(title="Depth (m)"),
        height=300, margin=dict(l=10, r=10, t=10, b=40),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font=dict(color='white'),
    )
    placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")
if show_heatmap or show_ade:
    _sp2_left, _sp2_right = st.columns(2)
    if show_heatmap:
        with _sp2_left:
            _live_header("🔥 어텐션 가중치 히트맵")
            heatmap_placeholder = st.empty()
    else:
        heatmap_placeholder = None
    if show_ade:
        with _sp2_right:
            _live_header("📊 ADE/FDE 비교 (Social Attention 전/후)")
            ade_placeholder = st.empty()
    else:
        ade_placeholder = None
    if show_occlusion:
        occlusion_info_placeholder = st.empty()
    else:
        occlusion_info_placeholder = None
    st.markdown("---")
else:
    heatmap_placeholder = None
    ade_placeholder = None
    occlusion_info_placeholder = None

if dataset_mode == "nuScenes":
    _risk_left, _risk_right = st.columns(2)
    with _risk_left:
        _live_header("🎯 위험도 분석 테이블")
        risk_table_placeholder = st.empty()
    with _risk_right:
        _live_header("🗺️ 위험도 히트맵")
        risk_heatmap_placeholder = st.empty()
    perf_placeholder = st.empty()
    st.markdown("---")
else:
    risk_table_placeholder = None
    risk_heatmap_placeholder = None
    perf_placeholder = None

st.subheader("📋 위험 감지 로그")
log_placeholder = st.empty()

if 'detection_logs' not in st.session_state:
    st.session_state.detection_logs = []
if run_simulation:
    st.session_state.detection_logs = []

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
        scenes = [nuscenes_scene]
        for scene_name in scenes:
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
                risk_rows    = []
                fig_map      = go.Figure()
                danger_objs  = []
                warning_objs = []
                safe_count   = 0
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
                    risk_score = compute_risk_score(ttc)
                    risk_level = get_risk_level(risk_score)

                    color_rgb, color_str, status = get_color(ttc, ttc_threshold, warning_threshold)
                    if status == "Danger":
                        danger_objs.append(ttc)
                    elif status == "Warning":
                        warning_objs.append(ttc)
                    else:
                        safe_count += 1

                    risk_rows.append({
                        'ID':         tid,
                        'Type':       obj['type'],
                        'TTC(s)':     round(ttc, 2),
                        'Risk Score': round(risk_score, 3),
                        'Level':      risk_level,
                        'Depth(m)':   round(depth, 1),
                    })

                    fig_map.add_trace(go.Scatter(
                        x=[rel_x], y=[rel_y],
                        mode='markers+text',
                        marker=dict(size=max(12, int(risk_score * 30)),
                                    color=color_str, line=dict(width=1, color='black')),
                        text=[f"{obj['type']}<br>TTC:{ttc:.1f}s<br>{risk_level}({risk_score:.2f})"],
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

                # ── Sprint 2: Social Attention + Occlusion ──────────────────
                _scene_objs = _build_scene_objects(
                    current_df, full_df, f_idx, scene_name=scene_name, fps=2.0
                )
                _pred_ms = 0.0
                if _scene_objs:
                    _social = engine.social_attention.compute(_scene_objs)
                    _render_heatmap(_social, heatmap_placeholder)

                    _worker = st.session_state.async_worker
                    _t_pred = time.perf_counter()
                    if _worker is not None:
                        _fid = _worker.submit(_scene_objs, fps=2.0)
                        _ade_result = _worker.get_result(_fid, timeout=0.5)
                    else:
                        _ade_result = engine.predict_scene(_scene_objs, fps=2.0)
                    _pred_ms = (time.perf_counter() - _t_pred) * 1000
                    if _ade_result:
                        st.session_state.ade_history.append({
                            'no_social':     _ade_result['ade_no_social'],
                            'social':        _ade_result['ade_social'],
                            'fde_no_social': _ade_result.get('fde_no_social', 0.0),
                            'fde_social':    _ade_result.get('fde_social', 0.0),
                        })
                        _render_ade_chart(st.session_state.ade_history, ade_placeholder)

                if show_occlusion and st.session_state.occlusion_handler:
                    visible_ids = [str(r['track_id']) for r in _scene_objs]
                    _ext = st.session_state.occlusion_handler.update(visible_ids, _scene_objs)
                    for _vo in _ext:
                        if not _vo.get('occluded'):
                            continue
                        fig_map.add_trace(go.Scatter(
                            x=[_vo['rel_x']], y=[_vo['rel_y']],
                            mode='markers+text',
                            marker=dict(size=14, color='gray', symbol='circle-open',
                                        opacity=0.5, line=dict(width=2, color='gray')),
                            text=[f"(가림) ID:{_vo['track_id']}"],
                            textposition='top center', showlegend=False,
                        ))
                    _render_occlusion_info(st.session_state.occlusion_handler, occlusion_info_placeholder)
                # ───────────────────────────────────────────────────────────

                # Risk table + heatmap + perf card
                if risk_rows:
                    _render_risk_heatmap(risk_rows, risk_heatmap_placeholder)
                    _risk_df = pd.DataFrame(risk_rows).sort_values('Risk Score', ascending=False)
                    if risk_table_placeholder is not None:
                        risk_table_placeholder.dataframe(_risk_df, use_container_width=True, hide_index=True)
                if perf_placeholder is not None:
                    _fps = 1000.0 / max(_pred_ms, 1)
                    perf_placeholder.markdown(
                        f'<div style="background:#1e293b;border-radius:8px;padding:8px 14px;'
                        f'font-size:13px;color:#94a3b8">'
                        f'⚡ <b>추론 성능</b> &nbsp;|&nbsp; 예측 응답: '
                        f'<b style="color:#22d3ee">{_pred_ms:.1f} ms</b>'
                        f' &nbsp;|&nbsp; 추론 FPS: <b style="color:#4ade80">{_fps:.1f}</b>'
                        f' &nbsp;|&nbsp; <span style="color:#64748b;font-size:11px">'
                        f'(렌더링 제외 순수 ML 추론)</span></div>',
                        unsafe_allow_html=True
                    )

                if gps_fixed:
                    fig_map.update_layout(
                        xaxis=dict(range=[-gps_zoom, gps_zoom], title="X (m, 자차 기준)"),
                        yaxis=dict(range=[-gps_zoom, gps_zoom], title="Y (m, 자차 기준)", scaleanchor="x"),
                        height=500, margin=dict(l=0, r=0, b=0, t=0),
                        showlegend=False, plot_bgcolor="#e8f4e8"
                    )
                else:
                    all_x = current_df['pos_x'].tolist()
                    all_y = current_df['pos_y'].tolist()
                    pad = 10
                    x_min = min(all_x + [0]) - pad
                    x_max = max(all_x + [0]) + pad
                    y_min = min(all_y + [0]) - pad
                    y_max = max(all_y + [0]) + pad
                    fig_map.update_layout(
                        xaxis=dict(range=[x_min, x_max], title="X (m, 자차 기준)"),
                        yaxis=dict(range=[y_min, y_max], title="Y (m, 자차 기준)", scaleanchor="x"),
                        height=500, margin=dict(l=0, r=0, b=0, t=0),
                        showlegend=False, plot_bgcolor="#e8f4e8"
                    )

                if img_path and os.path.exists(img_path):
                    rendered_img = draw_boxes_on_image(img_path, boxes_to_draw)
                    cam_placeholder.image(rendered_img, use_container_width=True)
                else:
                    cam_placeholder.info("카메라 이미지 없음")

                map_placeholder.plotly_chart(fig_map, use_container_width=True, key=f"map_{scene_name}_{f_idx}", config={'displayModeBar': False})

                _render_status(danger_objs, warning_objs, safe_count)
                now_kst = datetime.now(KST).strftime("%H:%M:%S")
                for ttc_val in danger_objs:
                    st.session_state.detection_logs.insert(0, {"시간(KST)": now_kst, "수준": "🚨 DANGER", "TTC": f"{ttc_val:.2f}s", "데이터셋": "nuScenes"})
                for ttc_val in warning_objs:
                    st.session_state.detection_logs.insert(0, {"시간(KST)": now_kst, "수준": "⚠️ WARNING", "TTC": f"{ttc_val:.2f}s", "데이터셋": "nuScenes"})
                if st.session_state.detection_logs:
                    log_placeholder.dataframe(pd.DataFrame(st.session_state.detection_logs).head(20), use_container_width=True)
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

            current_df   = current_df.nsmallest(10, 'depth')
            fig_map      = go.Figure()
            danger_objs  = []
            warning_objs = []
            safe_count   = 0
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

                color_rgb, color_str, status = get_color(ttc, ttc_threshold, warning_threshold)
                if status == "Danger":
                    danger_objs.append(ttc)
                elif status == "Warning":
                    warning_objs.append(ttc)
                else:
                    safe_count += 1

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

            # ── Sprint 2: Social Attention + Occlusion ──────────────────────
            _scene_objs = _build_scene_objects(current_df, full_df, f_idx, fps=10.0)
            if _scene_objs:
                _social = engine.social_attention.compute(_scene_objs)
                _render_heatmap(_social, heatmap_placeholder)

                _worker = st.session_state.async_worker
                if _worker is not None:
                    _fid = _worker.submit(_scene_objs, fps=10.0)
                    _ade_result = _worker.get_result(_fid, timeout=0.5)
                else:
                    _ade_result = engine.predict_scene(_scene_objs, fps=10.0)
                if _ade_result:
                    st.session_state.ade_history.append({
                        'no_social':     _ade_result['ade_no_social'],
                        'social':        _ade_result['ade_social'],
                        'fde_no_social': _ade_result.get('fde_no_social', 0.0),
                        'fde_social':    _ade_result.get('fde_social', 0.0),
                    })
                    _render_ade_chart(st.session_state.ade_history, ade_placeholder)

            if show_occlusion and st.session_state.occlusion_handler:
                visible_ids = [str(r['track_id']) for r in _scene_objs] if _scene_objs else []
                _ext = st.session_state.occlusion_handler.update(visible_ids, _scene_objs or [])
                for _vo in _ext:
                    if not _vo.get('occluded'):
                        continue
                    fig_map.add_trace(go.Scatter(
                        x=[_vo['rel_x']], y=[_vo['rel_y']],
                        mode='markers+text',
                        marker=dict(size=14, color='gray', symbol='circle-open',
                                    opacity=0.5, line=dict(width=2, color='gray')),
                        text=[f"(가림) ID:{_vo['track_id']}"],
                        textposition='top center', showlegend=False,
                    ))
                _render_occlusion_info(st.session_state.occlusion_handler, occlusion_info_placeholder)
            # ─────────────────────────────────────────────────────────────────

            if gps_fixed:
                fig_map.update_layout(
                    xaxis=dict(range=[-gps_zoom, gps_zoom], title="X (m)"),
                    yaxis=dict(range=[-gps_zoom, gps_zoom], title="Z/Depth (m)"),
                    height=500, margin=dict(l=0, r=0, b=0, t=0),
                    showlegend=False, plot_bgcolor="#e8f4e8"
                )
            else:
                fig_map.update_layout(
                    xaxis=dict(range=[-15, 15], title="X (m)"),
                    yaxis=dict(range=[-2,  50], title="Z/Depth (m)"),
                    height=500, margin=dict(l=0, r=0, b=0, t=0),
                    showlegend=False, plot_bgcolor="#e8f4e8"
                )
            map_placeholder.plotly_chart(fig_map, use_container_width=True, key=f"kitti_map_{f_idx}", config={'displayModeBar': False})

            if os.path.exists(img_path):
                rendered = draw_boxes_on_image(img_path, boxes_to_draw)
                cam_placeholder.image(rendered, use_container_width=True)
            else:
                cam_placeholder.info(f"이미지 없음: {img_path}")

            _render_status(danger_objs, warning_objs, safe_count)
            now_kst = datetime.now(KST).strftime("%H:%M:%S")
            for ttc_val in danger_objs:
                st.session_state.detection_logs.insert(0, {"시간(KST)": now_kst, "수준": "🚨 DANGER", "TTC": f"{ttc_val:.2f}s", "데이터셋": "KITTI"})
            for ttc_val in warning_objs:
                st.session_state.detection_logs.insert(0, {"시간(KST)": now_kst, "수준": "⚠️ WARNING", "TTC": f"{ttc_val:.2f}s", "데이터셋": "KITTI"})
            if st.session_state.detection_logs:
                log_placeholder.dataframe(pd.DataFrame(st.session_state.detection_logs).head(20), use_container_width=True)
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

            current_df   = current_df.nsmallest(10, 'depth')
            fig_map      = go.Figure()
            fig_cam      = go.Figure()
            danger_objs  = []
            warning_objs = []
            safe_count   = 0

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
                if status == "Danger":
                    danger_objs.append(ttc)
                elif status == "Warning":
                    warning_objs.append(ttc)
                else:
                    safe_count += 1

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

            # ── Sprint 2: Social Attention + Occlusion ──────────────────────
            _scene_objs = _build_scene_objects(current_df, full_df, f_idx, fps=2.5)
            if _scene_objs:
                _social = engine.social_attention.compute(_scene_objs)
                _render_heatmap(_social, heatmap_placeholder)

                _worker = st.session_state.async_worker
                if _worker is not None:
                    _fid = _worker.submit(_scene_objs, fps=2.5)
                    _ade_result = _worker.get_result(_fid, timeout=0.5)
                else:
                    _ade_result = engine.predict_scene(_scene_objs, fps=2.5)
                if _ade_result:
                    st.session_state.ade_history.append({
                        'no_social':     _ade_result['ade_no_social'],
                        'social':        _ade_result['ade_social'],
                        'fde_no_social': _ade_result.get('fde_no_social', 0.0),
                        'fde_social':    _ade_result.get('fde_social', 0.0),
                    })
                    _render_ade_chart(st.session_state.ade_history, ade_placeholder)

            if show_occlusion and st.session_state.occlusion_handler:
                visible_ids = [str(r['track_id']) for r in _scene_objs] if _scene_objs else []
                _ext = st.session_state.occlusion_handler.update(visible_ids, _scene_objs or [])
                for _vo in _ext:
                    if not _vo.get('occluded'):
                        continue
                    fig_map.add_trace(go.Scatter(
                        x=[_vo['rel_x']], y=[_vo['rel_y']],
                        mode='markers+text',
                        marker=dict(size=14, color='gray', symbol='circle-open',
                                    opacity=0.5, line=dict(width=2, color='gray')),
                        text=[f"(가림) ID:{_vo['track_id']}"],
                        textposition='top center', showlegend=False,
                    ))
                _render_occlusion_info(st.session_state.occlusion_handler, occlusion_info_placeholder)
            # ─────────────────────────────────────────────────────────────────

            map_placeholder.plotly_chart(fig_map, use_container_width=True, key=f"map_{f_idx}", config={'displayModeBar': False})
            cam_placeholder.plotly_chart(fig_cam, use_container_width=True, key=f"cam_{f_idx}", config={'displayModeBar': False})

            _render_status(danger_objs, warning_objs, safe_count)
            now_kst = datetime.now(KST).strftime("%H:%M:%S")
            for ttc_val in danger_objs:
                st.session_state.detection_logs.insert(0, {"시간(KST)": now_kst, "수준": "🚨 DANGER", "TTC": f"{ttc_val:.2f}s", "데이터셋": "ETH/UCY"})
            for ttc_val in warning_objs:
                st.session_state.detection_logs.insert(0, {"시간(KST)": now_kst, "수준": "⚠️ WARNING", "TTC": f"{ttc_val:.2f}s", "데이터셋": "ETH/UCY"})
            if st.session_state.detection_logs:
                log_placeholder.dataframe(pd.DataFrame(st.session_state.detection_logs).head(20), use_container_width=True)
            cam_meta_placeholder.markdown(
                '<div style="color:#888;font-size:12px;margin-top:4px">'
                '📷 ETH/UCY &nbsp;|&nbsp; 🎞 2.5 FPS'
                '</div>', unsafe_allow_html=True
            )

            if stop_simulation:
                st.warning("⏹️ 시뮬레이션이 정지되었습니다.")
                break

            time.sleep(sleep_time)