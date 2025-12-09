import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import folium
from streamlit_folium import st_folium
import json
import plotly.express as px  # pip install plotly nếu chưa có

# ==========================================
# 1. LOAD DATA
# ==========================================
DATA_PATH = "."  # Thư mục hiện tại (cùng với app.py)
CSV_FILE = "real_estate_cleaned.csv"

@st.cache_data
def load_data():
    csv_path = os.path.join(DATA_PATH, CSV_FILE)
    if not os.path.exists(csv_path):
        st.error(f"Không tìm thấy file CSV '{CSV_FILE}' trong thư mục hiện tại!")
        return None

    df = pd.read_csv(csv_path)

    # Chuyển price/area sang số nếu có
    for col in ["price", "area"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df = load_data()
if df is None:
    st.stop()
# ==========================================
# Chuẩn hóa tên cột
# ==========================================
rename_map = {
    "title": "Tiêu đề",
    "description": "Mô tả",
    "price": "Giá bán (VND)",
    "area": "Diện tích (m²)",
    "bedrooms": "Phòng ngủ",
    "bathrooms": "Toilet",
    "city": "Thành phố",
    "district": "Quận/Huyện",
    "legal_status": "Pháp lý",
    "date": "Ngày đăng",
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

if "Giá bán (VND)" in df.columns:
    df["Giá (tỷ)"] = df["Giá bán (VND)"] / 1_000_000_000

# ==========================================
# Sidebar menu
# ==========================================
menu = st.sidebar.radio(
    "Chọn danh mục",
    [
        "4.1 Phân tích mô tả",
        "4.2 Phân tích theo khu vực (Spatial)",
        "4.3 Phân tích theo thời gian (Temporal)",
        "4.4 Phân tích chuyên sâu (EDA nâng cao)",
    ]
)

st.title("📊 Exploratory Data Analysis (EDA) – Bất động sản")

# ==========================================
# 4.1 Phân tích mô tả
# ==========================================
if menu == "4.1 Phân tích mô tả":
    st.header("4.1 Phân tích mô tả")

    if "Giá (tỷ)" in df.columns:
        st.subheader("📌 Phân bố giá (TỶ VND)")
        fig, ax = plt.subplots()
        ax.hist(df["Giá (tỷ)"].dropna(), bins=40)
        ax.set_xlabel("Giá (tỷ VND)")
        ax.set_ylabel("Tần suất")
        st.pyplot(fig)
        with st.expander("📄 Xem dữ liệu giá"):
            st.dataframe(df[["Giá (tỷ)"]].dropna())

    if "Diện tích (m²)" in df.columns:
        st.subheader("📌 Phân bố diện tích")
        fig, ax = plt.subplots()
        ax.hist(df["Diện tích (m²)"].dropna(), bins=40)
        ax.set_xlabel("Diện tích (m²)")
        ax.set_ylabel("Tần suất")
        st.pyplot(fig)
        with st.expander("📄 Xem dữ liệu diện tích"):
            st.dataframe(df[["Diện tích (m²)"]].dropna())

    if "Giá (tỷ)" in df.columns and "Quận/Huyện" in df.columns:
        st.subheader("📌 Boxplot giá theo quận")
        fig, ax = plt.subplots(figsize=(11,5))
        df.boxplot(column="Giá (tỷ)", by="Quận/Huyện", ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)
        with st.expander("📄 Xem dữ liệu giá theo quận"):
            st.dataframe(df[["Quận/Huyện", "Giá (tỷ)"]])

    if "Quận/Huyện" in df.columns:
        st.subheader("📌 Top 10 quận đắt nhất (giá trung bình)")
        st.dataframe(df.groupby("Quận/Huyện")["Giá (tỷ)"].mean().sort_values(ascending=False).head(10))

        st.subheader("📌 Top 10 quận nhiều nguồn cung nhất")
        st.dataframe(df["Quận/Huyện"].value_counts().head(10))

# ==========================================
# 4.2 Phân tích theo khu vực (Spatial)
# ==========================================
elif menu == "4.2 Phân tích theo khu vực (Spatial)":
    st.header("4.2 Phân tích theo khu vực (Spatial)")

# Giá trung bình theo quận   
    if "Quận/Huyện" in df.columns:
        st.subheader("📌 Giá trung bình theo quận (tỷ VND)")
        # Tạo dataframe sắp xếp từ thấp → cao
        avg_price_df = df.groupby("Quận/Huyện")["Giá (tỷ)"].mean().reset_index()
        avg_price_df = avg_price_df.sort_values("Giá (tỷ)", ascending=True)

        # Vẽ bar chart bằng Plotly
        fig = px.bar(
            avg_price_df,
            x="Quận/Huyện",
            y="Giá (tỷ)",
            labels={"Giá (tỷ)": "Giá trung bình (tỷ VND)", "Quận/Huyện": "Quận"},
            title="Giá trung bình theo quận (tỷ VND)"
        )
        fig.update_layout(xaxis_tickangle=-45)  # xoay nhãn trục X
        st.plotly_chart(fig, use_container_width=True)

# Giá/m² theo quận
    if "Giá bán (VND)" in df.columns and "Diện tích (m²)" in df.columns:
        st.subheader("📌 Giá/m² theo quận")
        df["Giá/m²"] = df["Giá bán (VND)"] / df["Diện tích (m²)"]
        price_m2_df = df.groupby("Quận/Huyện")["Giá/m²"].mean().reset_index()
        price_m2_df = price_m2_df.sort_values("Giá/m²", ascending=True)

        fig = px.bar(
            price_m2_df,
            x="Quận/Huyện",
            y="Giá/m²",
            labels={"Giá/m²": "Giá/m² trung bình (VND)", "Quận/Huyện": "Quận"},
            title="Giá/m² trung bình theo quận"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# Bản đồ giá trung bình theo quận
    st.subheader("🗺️ Bản đồ giá trung bình theo quận (TP.HCM)")
    geojson_path = os.path.join(DATA_PATH, "ho-chi-minh-city.geojson")
    if not os.path.exists(geojson_path):
        st.error("Không tìm thấy file GeoJSON 'ho-chi-minh-city.geojson'!")
    else:
        with open(geojson_path, "r", encoding="utf-8") as f:
            geo_data = json.load(f)

    df_quan = df["Quận/Huyện"].unique().tolist()
    for feature, quan in zip(geo_data["features"], df_quan):
        feature["properties"]["Ten_QH"] = quan

    avg_price_map = df.groupby("Quận/Huyện")["Giá (tỷ)"].mean().reset_index()
    m = folium.Map(location=[10.7769, 106.7009], zoom_start=11)
    folium.Choropleth(
        geo_data=geo_data,
        data=avg_price_map,
        columns=["Quận/Huyện", "Giá (tỷ)"],
        key_on="feature.properties.Ten_QH",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        nan_fill_color="white",
        legend_name="Giá trung bình (tỷ VND)"
    ).add_to(m)
    st_folium(m, width=700, height=500)

# ==========================================
# 4.3 Phân tích theo thời gian
# ==========================================
elif menu == "4.3 Phân tích theo thời gian (Temporal)":
    st.header("4.3 Phân tích theo thời gian")
    if "Ngày đăng" in df.columns:
        df["Ngày đăng"] = pd.to_datetime(df["Ngày đăng"], errors="coerce")
        df["Năm"] = df["Ngày đăng"].dt.year

        st.subheader("📌 Xu hướng giá theo năm")
        trend = df.groupby("Năm")["Giá (tỷ)"].mean()
        st.line_chart(trend)

        st.subheader("📌 Số lượng tin đăng theo tháng")
        count_by_month = df.resample("M", on="Ngày đăng").size()
        st.line_chart(count_by_month)

        st.subheader("📌 Quận biến động giá mạnh nhất")
        var_district = df.groupby("Quận/Huyện")["Giá (tỷ)"].std().sort_values(ascending=False).head(10)
        st.dataframe(var_district)

# ==========================================
# 4.4 EDA nâng cao
# ==========================================
elif menu == "4.4 Phân tích chuyên sâu (EDA nâng cao)":
    st.header("4.4 Phân tích chuyên sâu")

    st.subheader("📌 Scatter plot: Giá vs Diện tích")
    fig, ax = plt.subplots()
    ax.scatter(df["Diện tích (m²)"], df["Giá (tỷ)"], alpha=0.4)
    ax.set_xlabel("Diện tích (m²)")
    ax.set_ylabel("Giá (tỷ)")
    st.pyplot(fig)

    st.subheader("📌 Ma trận tương quan")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap="YlGnBu")
    fig.colorbar(cax)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(range(len(numeric_df.columns)))
    ax.set_xticklabels(numeric_df.columns, rotation=45, ha="right")  # nghiêng 45 độ
    ax.set_yticks(range(len(numeric_df.columns)))
    ax.set_yticklabels(numeric_df.columns)

    st.pyplot(fig)


# Giá theo phòng ngủ
    if "Phòng ngủ" in df.columns:
        st.subheader("📌 Giá theo phòng ngủ")
        df_avg = df.groupby("Phòng ngủ")["Giá (tỷ)"].mean().reset_index()
        df_avg = df_avg.sort_values("Giá (tỷ)", ascending=True)
        fig = px.bar(df_avg, x="Phòng ngủ", y="Giá (tỷ)", labels={"Giá (tỷ)": "Giá trung bình (tỷ VND)"})
        st.plotly_chart(fig, use_container_width=True)

# Giá theo số toilet
    if "Toilet" in df.columns:
        st.subheader("📌 Giá theo số toilet")
        df_avg = df.groupby("Toilet")["Giá (tỷ)"].mean().reset_index()
        df_avg = df_avg.sort_values("Giá (tỷ)", ascending=True)
        fig = px.bar(df_avg, x="Toilet", y="Giá (tỷ)", labels={"Giá (tỷ)": "Giá trung bình (tỷ VND)"})
        st.plotly_chart(fig, use_container_width=True)

# Giá theo pháp lý
    if "Pháp lý" in df.columns:
        st.subheader("📌 Giá theo pháp lý")
        df_avg = df.groupby("Pháp lý")["Giá (tỷ)"].mean().reset_index()
        df_avg = df_avg.sort_values("Giá (tỷ)", ascending=True)
        fig = px.bar(df_avg, x="Pháp lý", y="Giá (tỷ)", labels={"Giá (tỷ)": "Giá trung bình (tỷ VND)"})
        st.plotly_chart(fig, use_container_width=True)
