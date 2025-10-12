import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go
import io

# ==============================================================================
# 1. Hàm Phân tích Điểm Bất Thường
# ==============================================================================

def analyze_outliers(df_grades, subject_cols, threshold=2.0):
    """
    Thực hiện tính toán số lượng học sinh có điểm bất thường theo Z-score 
    cho từng môn học ở mỗi lớp.
    """
    # Loại bỏ các hàng không có thông tin lớp
    df_grades.dropna(subset=['lop'], inplace=True)
    
    # Chuyển đổi các cột điểm sang dạng số (numeric)
    for col in subject_cols:
        df_grades[col] = pd.to_numeric(df_grades[col], errors='coerce')

    # Khởi tạo DataFrame để lưu kết quả đếm số lượng outliers
    unique_classes = df_grades['lop'].unique()
    outlier_counts = pd.DataFrame(index=unique_classes, columns=subject_cols)
    
    for lop in unique_classes:
        df_class = df_grades[df_grades['lop'] == lop]
        
        for subject in subject_cols:
            # Lấy các điểm số không phải NaN của môn học trong lớp đó
            scores = df_class[subject].dropna()
            
            if len(scores) >= 2: # Cần ít nhất 2 điểm để tính Z-score
                # Tính Z-score cho các điểm số
                z_scores = zscore(scores)
                
                # Đếm số lượng điểm bất thường: |Z| > ngưỡng
                num_outliers = (np.abs(z_scores) > threshold).sum()
                
                # Lưu kết quả
                outlier_counts.loc[lop, subject] = num_outliers
            else:
                outlier_counts.loc[lop, subject] = 0

    # Xử lý kết quả cuối cùng
    outlier_counts = outlier_counts.fillna(0).astype(int)
    
    # Tính tổng số outliers theo lớp và theo môn
    outlier_counts['Tổng Outliers Lớp'] = outlier_counts.sum(axis=1)
    outlier_counts.loc['Tổng Outliers Môn'] = outlier_counts.drop('Tổng Outliers Lớp', axis=1).sum(axis=0)
    
    return outlier_counts.iloc[:-1, :], outlier_counts.iloc[-1, :-1] # Trả về bảng chính và hàng tổng môn

# ==============================================================================
# 2. Thiết lập Giao diện Streamlit
# ==============================================================================

st.set_page_config(layout="wide")

st.title("📊 Phân Tích Điểm Bất Thường theo Z-score")
st.markdown("---")

# ----------------- Tải File -----------------
st.header("1. Tải File Dữ liệu (.csv)")
uploaded_file = st.file_uploader(
    "Vui lòng tải lên file **Student_GradeSummary.csv** (chứa cột 'lop', 'Toan', 'Van', ...)", 
    type=['csv']
)

if uploaded_file is not None:
    # Đọc file
    try:
        df_grades = pd.read_csv(uploaded_file)
        st.success(f"Đã tải thành công file: **{uploaded_file.name}**")
        
        # ----------------- Thiết lập Tham số -----------------
        st.header("2. Thiết lập Tham số Phân tích")

        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Thiết lập ngưỡng Z-score (mặc định là 2.0 theo yêu cầu)
            z_score_threshold = st.slider(
                "Chọn Ngưỡng Z-score ($\left| Z \\right| >$):", 
                min_value=1.5, 
                max_value=3.5, 
                value=2.0, 
                step=0.1,
                help="Điểm số được coi là bất thường nếu độ lệch chuẩn của nó so với trung bình lớp lớn hơn ngưỡng này."
            )
        
        with col2:
            st.info(f"Đang sử dụng ngưỡng **$\left| Z \\right| > {z_score_threshold}$** cho phân tích.")

        # ----------------- Thực hiện Phân tích -----------------
        
        # Danh sách các cột điểm cần phân tích (cần điều chỉnh nếu tên cột khác)
        subject_cols = ['Toan', 'Van', 'Ly', 'Hoa', 'Ngoaingu', 'Su', 'Tin', 'Sinh', 'Dia']
        
        # Kiểm tra xem các cột cần thiết có tồn tại không
        missing_cols = [col for col in ['lop'] + subject_cols if col not in df_grades.columns]
        
        if missing_cols:
            st.error(f"Lỗi: File CSV thiếu các cột cần thiết: {', '.join(missing_cols)}. Vui lòng kiểm tra lại file **Student_GradeSummary.csv**.")
        else:
            st.header("3. Kết Quả Phân Tích")
            
            # Thực hiện phân tích
            outlier_counts_df, total_outliers_by_subject = analyze_outliers(
                df_grades.copy(), 
                subject_cols, 
                z_score_threshold
            )
            
            # ----------------- Hiển thị Bảng Dữ liệu -----------------
            st.subheader("Bảng Tổng Kết Số Học Sinh Có Điểm Bất Thường")
            st.markdown(f"*(Ngưỡng $\left| Z \\right| > {z_score_threshold}$)*")
            
            # Hiển thị bảng
            st.dataframe(outlier_counts_df.style.background_gradient(cmap='Blues'), use_container_width=True)
            
            # Hiển thị hàng tổng môn
            st.markdown("---")
            total_row_df = pd.DataFrame([total_outliers_by_subject]).T.rename(columns={0: 'Tổng Outliers Môn'})
            st.dataframe(total_row_df.T.style.background_gradient(cmap='Reds'), use_container_width=True)
            st.markdown("---")


            # ----------------- Trực quan hóa Heatmap -----------------
            st.subheader("Biểu Đồ Heatmap")
            
            # Chuẩn bị dữ liệu cho heatmap (chỉ lấy các cột môn học)
            heatmap_data = outlier_counts_df.drop(columns=['Tổng Outliers Lớp']).reset_index().melt(
                id_vars='index', var_name='Môn Học', value_name='Số HS Outlier'
            ).rename(columns={'index': 'Lớp'})

            fig_heatmap = px.density_heatmap(
                heatmap_data, 
                x="Môn Học", 
                y="Lớp", 
                z="Số HS Outlier",
                color_continuous_scale="Viridis",
                title=f"Số Học Sinh Có Điểm Bất Thường ($\left| Z \\right| > {z_score_threshold}$) theo Lớp và Môn Học",
                text_auto=True # Hiển thị số lượng trực tiếp trên ô
            )
            
            # Điều chỉnh layout
            fig_heatmap.update_layout(
                yaxis={'categoryorder': 'array', 'categoryarray': sorted(outlier_counts_df.index.tolist(), reverse=True)}, # Sắp xếp lớp
                xaxis={'categoryorder': 'array', 'categoryarray': subject_cols}, # Sắp xếp môn học
                height=600 
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi xử lý file. Vui lòng kiểm tra định dạng file: {e}")