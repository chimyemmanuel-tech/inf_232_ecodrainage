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

# Professional Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .reportview-container .main .block-container { padding-top: 2rem; }
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
        return pd.DataFrame(res.data)
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
        # KPI Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Observations", len(df))
        c2.metric("Waste Volume (L)", f"{df['volume'].sum():,.0f}")
        c3.metric("Critical Points", len(df[df['volume'] > 50]))
        c4.metric("System Health", "Active", delta="Stable")

        # Geospatial Map
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
    st.markdown("Use this form to log plastic accumulation in the urban drainage network.")
    
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
        try:
            payload = {"plastic_type": p_type, "volume": vol, "hub_type": hub, "lat": lat, "lon": lon, "risk_level": risk}
            supabase.table("drainage_waste").insert(payload).execute()
            st.success("Data synchronized successfully.")
            st.balloons()
            st.rerun()
        except Exception as e:
            st.error(f"Connection lost. Please try submitting again. Error: {e}")

# --- PAGE 3: EC2 ANALYTICS ENGINE ---
elif menu == "EC2 Analytics Engine":
    st.title("📊 Advanced Statistical Modeling (EC2)")
    
    if len(df) < 8:
        st.warning("Insufficient data. Minimum 8 entries required to run valid statistical models.")
    else:
        # Pre-processing for ML
        le = LabelEncoder()
        df_ml = df.copy()
        df_ml['type_enc'] = le.fit_transform(df['plastic_type'])
        df_ml['hub_enc'] = le.fit_transform(df['hub_type'])
        df_ml['risk_enc'] = le.fit_transform(df['risk_level'])

        tabs = st.tabs(["1 & 2: Linear Regression", "3: Dimensionality (PCA)", "4 & 5: Classification"])

        with tabs[0]:
            st.header("Linear Regression Analysis")
            st.write("Modeling the relationship between Waste Volume and Location/Source.")
            
            # Simple Regression
            X_simple = df_ml[['lat']]
            y = df_ml['volume']
            model_s = LinearRegression().fit(X_simple, y)
            
            # Multiple Regression
            X_multi = df_ml[['type_enc', 'hub_enc', 'lat', 'lon']]
            model_m = LinearRegression().fit(X_multi, y)
            
            st.latex(r"V = \beta_0 + \beta_1(Type) + \beta_2(Hub) + \epsilon")
            st.metric("Multiple Regression R²", f"{r2_score(y, model_m.predict(X_multi)):.4f}")

        with tabs[1]:
            st.header("Techniques de réduction (PCA)")
            st.write("Reducing multidimensional field data into Principal Components.")
            features = ['volume', 'lat', 'lon', 'type_enc', 'hub_enc']
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
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(df_ml[['lat', 'lon']])
                df['Cluster'] = kmeans.labels_
                st.plotly_chart(px.scatter(df, x='lon', y='lat', color='Cluster', title="Waste Hotspots"))
            
            with col_b:
                st.subheader("Supervised: Random Forest")
                # Predict Risk Level based on Volume and Type
                X_cls = df_ml[['volume', 'type_enc', 'hub_enc']]
                y_cls = df_ml['risk_level']
                rf = RandomForestClassifier().fit(X_cls, y_cls)
                st.success("Model trained to predict Blockage Risk.")

elif menu == "Documentation":
    st.title("📖 Project Documentation")
    st.info("Project: INF 232 EC2 - Analysis and Collection of Urban Infrastructure Data")
    st.write("""
    ### Technical Specification:
    - **Backend:** Supabase (PostgreSQL)
    - **Frontend:** Streamlit Framework
    - **Models:** Scikit-Learn (PCA, KMeans, LinearRegression)
    - **Visualization:** Plotly Geospatial Engine
    """)
