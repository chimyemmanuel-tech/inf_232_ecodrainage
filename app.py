import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score

# --- CONFIGURATION & UI THEME ---
st.set_page_config(page_title="EcoDrain Analytics v2.0", layout="wide", page_icon="🌊")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- DATABASE LAYER ---
@st.cache_resource
def init_connection() -> Client:
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception as e:
        st.error("Database connection failed. Check your Streamlit Secrets.")
        return None

supabase = init_connection()

def fetch_data():
    try:
        res = supabase.table("drainage_waste").select("*").execute()
        raw_df = pd.DataFrame(res.data)
        if not raw_df.empty:
            # CLEANING: Drop any rows missing critical ML features
            raw_df = raw_df.dropna(subset=['lat', 'lon', 'volume', 'plastic_type', 'hub_type', 'risk_level'])
        return raw_df
    except:
        return pd.DataFrame()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🌊 INF 232 EC2 Project")
st.sidebar.markdown("**Focus:** Urban Plastic Waste Flow")
menu = st.sidebar.selectbox("Navigate System", 
    ["Live Dashboard", "Field Data Collection", "EC2 Analytics Engine", "Documentation"])

df = fetch_data()

# --- PAGE 1: DASHBOARD ---
if menu == "Live Dashboard":
    st.title("📍 Infrastructure Monitoring Dashboard")
    
    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Observations", len(df))
        c2.metric("Waste Volume (L)", f"{df['volume'].sum():,.0f}")
        c3.metric("Critical Points", len(df[df['volume'] > 50]))
        c4.metric("System Health", "Active", delta="Stable")

        st.subheader("Geospatial Distribution of Drainage Blockages")
        fig_map = px.scatter_mapbox(df, lat="lat", lon="lon", color="plastic_type", size="volume",
                                    hover_name="hub_type", zoom=12, mapbox_style="carto-positron",
                                    height=600, color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("System is ready. Please proceed to 'Field Data Collection' to input the first report.")

# --- PAGE 2: DATA COLLECTION ---
elif menu == "Field Data Collection":
    st.title("📥 Field Data Entry")
    with st.form("entry_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            p_type = st.selectbox("Plastic Category", ["PET Bottles", "LDPE Bags", "Multi-layer Sachets", "Other"])
            vol = st.number_input("Estimated Volume (Liters)", min_value=0.1, step=0.5)
            hub = st.selectbox("Proximity to Urban Hub", ["Market", "Residential Hub", "Industrial Area", "Transport Terminal"])
        with col2:
            lat = st.number_input("GPS Latitude", format="%.6f", value=4.0500)
            lon = st.number_input("GPS Longitude", format="%.6f", value=9.7000)
            risk = st.select_slider("Immediate Blockage Risk", options=["Low", "Medium", "High"])
            
        submitted = st.form_submit_button("Submit to Cloud")
        if submitted:
            payload = {"plastic_type": p_type, "volume": vol, "hub_type": hub, "lat": lat, "lon": lon, "risk_level": risk}
            supabase.table("drainage_waste").insert(payload).execute()
            st.success("Data synchronized successfully.")
            st.balloons()
            st.rerun()

# --- PAGE 3: EC2 ANALYTICS ENGINE ---
elif menu == "EC2 Analytics Engine":
    st.title("📊 Advanced Statistical Modeling (EC2)")
    
    if len(df) < 8:
        st.warning("Insufficient data. Please add at least 8 entries in 'Field Data Collection' to run models.")
    else:
        # Pre-processing
        le = LabelEncoder()
        df_ml = df.copy()
        for col in ['plastic_type', 'hub_type', 'risk_level']:
            df_ml[f'{col}_enc'] = le.fit_transform(df_ml[col])

        tabs = st.tabs(["1 & 2: Linear Regression", "3: Dimensionality (PCA)", "4 & 5: Classification"])

        with tabs[0]:
            st.header("Linear Regression Analysis")
            X_multi = df_ml[['plastic_type_enc', 'hub_type_enc', 'lat', 'lon']]
            y = df_ml['volume']
            model_m = LinearRegression().fit(X_multi, y)
            st.latex(r"V = \beta_0 + \beta_1(Type) + \beta_2(Hub) + \epsilon")
            st.metric("Multiple Regression R²", f"{r2_score(y, model_m.predict(X_multi)):.4f}")

        with tabs[1]:
            st.header("Techniques de réduction (PCA)")
            features = ['volume', 'lat', 'lon', 'plastic_type_enc', 'hub_type_enc']
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(df_ml[features])
            
            pca = PCA(n_components=2)
            pc = pca.fit_transform(x_scaled)
            pca_df = pd.DataFrame(pc, columns=['PC1', 'PC2'])
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color=df['plastic_type'], title="PCA Variance Mapping")
            st.plotly_chart(fig_pca, use_container_width=True)

        with tabs[2]:
            st.header("Supervised & Unsupervised Classification")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Unsupervised: K-Means")
                # Error prevention: Using lat/lon for clustering
                X_km = df_ml[['lat', 'lon']]
                kmeans = KMeans(n_components=3, n_init=10, random_state=42).fit(X_km)
                df_ml['Cluster'] = kmeans.labels_
                st.plotly_chart(px.scatter(df_ml, x='lon', y='lat', color='Cluster', title="Waste Hotspots"))
            
            with col_b:
                st.subheader("Supervised: Random Forest")
                X_rf = df_ml[['volume', 'plastic_type_enc', 'hub_type_enc']]
                y_rf = df['risk_level']
                rf = RandomForestClassifier(random_state=42).fit(X_rf, y_rf)
                st.success("Random Forest model trained successfully.")

elif menu == "Documentation":
    st.title("📖 Project Documentation")
    st.write("### Technical Specification:")
    st.markdown("- **Backend:** Supabase (PostgreSQL)\n- **Frontend:** Streamlit\n- **Models:** Scikit-Learn\n- **Requirement:** INF 232 EC2")
