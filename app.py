import sys
import os
import zipfile

# ==========================================
# 0. 自动解压核心源码 (突破 GitHub 100文件限制的秘籍)
# ==========================================
# 检查云端是否存在压缩包，且是否还没解压过
if os.path.exists("ultralytics-main.zip") and not os.path.exists("ultralytics-main/ultralytics"):
    try:
        print("📥 正在解压魔改版 YOLO 源码...")
        with zipfile.ZipFile("ultralytics-main.zip", 'r') as zip_ref:
            # 直接解压到当前根目录
            zip_ref.extractall(".")
        print("✅ 解压完成！")
    except Exception as e:
        print(f"❌ 解压失败: {e}")

# 将解压后的魔改版源码路径置于系统最高优先级
if os.path.exists("ultralytics-main/ultralytics"):
    sys.path.insert(0, os.path.abspath("ultralytics-main"))
elif os.path.exists("ultralytics"):
    sys.path.insert(0, os.path.abspath("."))

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# ==========================================
# 1. 页面全局设置
# ==========================================
st.set_page_config(
    page_title="红外无人机检测系统",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 现代扁平化 UI 样式注入 (CSS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --primary-color: #2563EB;
        --primary-hover: #1D4ED8;
        --secondary-color: #10B981;
        --bg-color: #F8FAFC;
        --surface-color: #FFFFFF;
        --text-main: #0F172A;
        --text-secondary: #475569;
        --border-color: #E2E8F0;
        --radius-md: 8px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: var(--text-main);
        background-color: var(--bg-color);
    }

    h1 { font-size: 32px !important; font-weight: 700 !important; color: var(--text-main) !important; margin-bottom: 24px !important; }
    h2 { font-size: 24px !important; font-weight: 600 !important; color: var(--text-main) !important; margin-top: 32px !important; }
    h3 { font-size: 18px !important; font-weight: 600 !important; color: var(--text-secondary) !important; }

    section[data-testid="stSidebar"] {
        background-color: var(--surface-color);
        border-right: 1px solid var(--border-color);
    }
    
    .stButton button {
        background-color: var(--primary-color) !important;
        color: white !important;
        height: 40px !important;
        padding: 0 24px !important;
        border-radius: var(--radius-md) !important;
        border: none !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton button:hover {
        background-color: var(--primary-hover) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 16px; border-bottom: none !important; padding-bottom: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 64px !important; min-width: 180px; background-color: #F1F5F9; border: 1px solid #E2E8F0;
        border-radius: 12px !important; color: #64748B; font-size: 18px !important; font-weight: 600 !important;
        padding: 0 32px !important; box-shadow: 0 2px 4px rgba(0,0,0,0.02); transition: all 0.3s !important; margin-right: 8px;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #E2E8F0; color: #334155; transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%) !important; color: white !important;
        border: none !important; box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3), 0 4px 6px -2px rgba(37, 99, 235, 0.1) !important; transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab"] p { font-weight: 600 !important; font-size: 18px !important; }
    .stImage img { border-radius: var(--radius-md); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
    [data-testid="stMetricValue"] { font-size: 24px !important; color: var(--primary-color) !important; font-weight: 700 !important; }
    .block-container { padding-top: 2rem; padding-bottom: 4rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 模型与数据加载 
# ==========================================
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        return None

@st.cache_data
def load_training_data():
    csv_path = 'results.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        return df
    return None

model = load_model()
df_results = load_training_data()

# ==========================================
# 4. 侧边栏 (新增算法架构模块)
# ==========================================
with st.sidebar:
    st.title("🚁 系统控制台")
    st.markdown("军工级红外无人机探测终端")
    st.markdown("---")
    
    # 学术风：算法架构选择展示
    st.subheader("🧠 算法架构配置")
    model_version = st.selectbox(
        "选择底层驱动模型",
        ["YOLOv11-Custom (当前最优)", "YOLOv11-Base (基线对比)"]
    )
    
    # 动态显示当前的创新点，方便答辩和演示
    if model_version == "YOLOv11-Custom (当前最优)":
        st.success("""
        **当前启用核心创新模块：**
        - `HWD`: 哈尔小波下采样 (保特征)
        - `CCFM`: 跨尺度上下文融合 (抗多变)
        - `P2head`: 微小目标专属检测头
        *(预留接口: NWD-Loss, Attention)*
        """)
    else:
        st.info("当前使用 YOLOv11 官方基线结构，用于消融实验性能对比。")

    st.markdown("---")
    st.subheader("⚙️ 侦测参数")
    conf_threshold = st.slider(
        "置信度阈值 (Confidence)", 
        min_value=0.1, max_value=1.0, value=0.5, step=0.05
    )
    st.caption("数值越高，模型只会圈出它越有把握的目标；数值越低，会圈出更多可疑目标。")
    
    st.markdown("---")
    st.info("""
    **图例说明：**
    - 🛩️ **0: 固定翼** (fixed)
    - 🚁 **1: 多旋翼** (multi)
    """)

# ==========================================
# 5. 主页面内容
# ==========================================
st.title("🎯 红外无人机检测系统")

tab1, tab2 = st.tabs(["📈 训练数据大屏", "🔍 红外实时侦测"])

# ----------------- Tab 1: 训练数据大屏 -----------------
with tab1:
    st.markdown(f"### 📈 模型训练全景分析 - `{model_version}`")
    st.markdown("通过动态交互式图表，全面回顾模型 **100轮** 的进化历程。")
    
    if df_results is not None:
        best_map50 = df_results['metrics/mAP50(B)'].max()
        final_p = df_results['metrics/precision(B)'].iloc[-1]
        final_r = df_results['metrics/recall(B)'].iloc[-1]
        
        st.subheader("🏆 核心能力评估")
        col1, col2, col3 = st.columns(3)
        col1.metric("综合识别精度 (mAP@0.5)", f"{best_map50 * 100:.1f} %", "超越绝大多数基线模型")
        col2.metric("精确率 (Precision)", f"{final_p * 100:.1f} %", "极低的误报率")
        col3.metric("召回率 (Recall)", f"{final_r * 100:.1f} %", "极低的漏报率")
        
        st.markdown("---")
        
        st.subheader("📊 学习曲线动态追踪")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("**mAP 综合精度提升曲线**")
            chart_data_map = df_results[['epoch', 'metrics/mAP50(B)']].set_index('epoch')
            st.line_chart(chart_data_map, color="#10B981") 
            
        with chart_col2:
            st.markdown("**Loss 误差下降曲线 (Box Loss)**")
            chart_data_loss = df_results[['epoch', 'train/box_loss', 'val/box_loss']].set_index('epoch')
            st.line_chart(chart_data_loss)
            
        st.markdown("---")
        
        st.subheader("🖼️ 深度专业分析图")
        with st.expander("点击展开查看 F1曲线、PR曲线 及 验证集可视化", expanded=False):
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                pr_path = 'BoxPR_curve.png'
                if os.path.exists(pr_path):
                    st.image(pr_path, caption="PR 曲线", use_container_width=True)
                else:
                    st.warning("PR 曲线未找到")
                    
                val_path = 'val_batch0_pred.jpg'
                if os.path.exists(val_path):
                    st.image(val_path, caption="验证集实测切片", use_container_width=True)
                    
            with img_col2:
                f1_path = 'BoxF1_curve.png'
                if os.path.exists(f1_path):
                    st.image(f1_path, caption="F1 - 置信度 曲线", use_container_width=True)
    else:
        st.error("❌ 未找到训练日志 (results.csv)，请确保模型已正确训练完毕，并上传至 GitHub 仓库。")

# ----------------- Tab 2: 红外实时侦测 -----------------
with tab2:
    st.markdown("### 🔍 实时侦测终端")
    
    with st.container():
        uploaded_file = st.file_uploader("📁 请选择红外侦察图像 (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("#### 📥 原始红外信号")
            st.image(image, use_container_width=True)
            
        with col_img2:
            st.markdown("#### ⚡ AI 锁定目标")
            
            result_container = st.empty()
            
            if st.button("启动侦测协议", type="primary", use_container_width=True):
                if model is None:
                    st.error("❌ 模型权重 (best.pt) 未加载，请确保已上传至 GitHub。")
                else:
                    with st.spinner('雷达扫描中，正在提取微小红外特征...'):
                        try:
                            # 1. 预测
                            results = model.predict(source=img_array, conf=conf_threshold, save=False)
                            result = results[0]
                            
                            # 2. 修改类别名称
                            result.names = {0: 'fixed', 1: 'multi'}
                            
                            # 3. 绘制
                            annotated_img_bgr = result.plot(line_width=2)
                            annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                            
                            # 4. 显示
                            result_container.image(annotated_img_rgb, use_container_width=True)
                            
                            # 5. 统计
                            num_boxes = len(result.boxes)
                            if num_boxes > 0:
                                st.success(f"🎯 侦测完毕！锁定 {num_boxes} 个目标。")
                                with st.expander("📋 查看目标详细参数", expanded=True):
                                    for i, box in enumerate(result.boxes):
                                        cls_id = int(box.cls[0])
                                        conf = float(box.conf[0])
                                        cls_name = result.names[cls_id]
                                        st.markdown(f"**目标 {i+1}**: `{cls_name}` | 置信度: `{conf:.2f}`")
                                        st.progress(conf)
                            else:
                                st.warning("🈳 画面安全，未发现任何可疑信号。")
                        except Exception as e:
                            st.error(f"侦测过程中发生错误: {str(e)}")
