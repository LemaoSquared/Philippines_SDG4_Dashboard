import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import numpy as np
import folium
from streamlit_folium import st_folium
from folium import GeoJson, GeoJsonTooltip
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Philippines SDG 4 Analytics", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ENHANCED CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { 
        background: linear-gradient(135deg, #f8fafc 0%, #f0f9ff 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* MAIN TITLE WITH GRADIENT */
    .main-title {
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(135deg, #0f172a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    
    .subtitle {
        font-size: 14px;
        color: #64748b;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* PREMIUM KPI CARDS WITH ANIMATION */
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 24px 16px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border-left: 6px solid #3b82f6;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    .kpi-card:hover { 
        transform: translateY(-8px);
        box-shadow: 0 16px 32px rgba(59, 130, 246, 0.15);
    }
    .kpi-label { 
        font-size: 11px; 
        text-transform: uppercase; 
        letter-spacing: 2px; 
        color: #64748b; 
        font-weight: 700;
        margin-bottom: 8px;
    }
    .kpi-value { 
        font-size: 36px; 
        font-weight: 900; 
        color: #0f172a;
        margin-bottom: 4px;
    }
    .kpi-subtitle { 
        font-size: 12px; 
        color: #94a3b8;
        font-weight: 500;
    }
    
    /* INSIGHT CARDS */
    .insight-card {
        background: white;
        padding: 16px;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* SECTION HEADERS */
    .insight-header { 
        margin-top: 40px; 
        margin-bottom: 24px; 
        border-left: 6px solid #3b82f6; 
        padding-left: 16px;
    }
    .insight-header h3 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: flex-start;
        background-color: transparent;
        border-bottom: 2px solid #e2e8f0 !important;
        padding: 0;
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b !important;
        background-color: transparent !important;
        border: none !important;
        font-weight: 700;
        font-size: 15px;
        padding-bottom: 12px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0f172a !important;
        border-bottom: 3px solid #3b82f6 !important;
    }
    
    /* PROGRESS INDICATORS */
    .progress-bar {
        background: #e2e8f0;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 8px 0;
    }
    .progress-fill {
        background: linear-gradient(90deg, #3b82f6, #0f172a);
        height: 100%;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_assets():
    file_path = 'Cleaned_Philippines_Education_Statistics.csv'
    if not os.path.exists(file_path):
        file_path = 'data_wrangling/Cleaned_Philippines_Education_Statistics.csv'
    
    df = pd.read_csv(file_path)
    geojson = None
    if os.path.exists('philippines_regions.json'):
        with open('philippines_regions.json', 'r', encoding='utf-8') as f:
            geojson = json.load(f)
    return df, geojson

# --- LOAD ML MODELS ---
@st.cache_resource
def load_kmeans_model():
    model_path = 'import_models//kmeans_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_kmeans_scaler():
    model_path = 'import_models//scaler.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

df_all, ph_geojson = load_assets()
kmeans_model = load_kmeans_model()
kmeans_scaler = load_kmeans_scaler()

# --- HELPER FUNCTIONS ---
def draw_enhanced_kpi(label, value, subtitle, color="#3b82f6"):
    return f"""
        <div class="kpi-card" style="border-left-color: {color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
    """

def calculate_trend(series):
    """Calculate trend direction (up/down/stable)"""
    if len(series) < 2:
        return "→"
    change = series.iloc[-1] - series.iloc[0]
    if change > 1:
        return "📈"
    elif change < -1:
        return "📉"
    else:
        return "→"

def create_performance_gauge(value, target=100, label="Performance"):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': target, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, target]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, target*0.5], 'color': "#fca5a5"},
                {'range': [target*0.5, target*0.8], 'color': "#fbbf24"},
                {'range': [target*0.8, target], 'color': "#86efac"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    return fig

# --- HEADER ---
st.markdown('<div class="main-title">Philippines Education SDG 4 Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Data-Driven Insights on Education Quality & Equity (2002-2023)</div>', unsafe_allow_html=True)
st.write("")

# --- NAVIGATION TABS ---
tab_nat, tab_reg, tab_insights, tab_ml, tab_about = st.tabs([
    "National Overview", "Regional Analysis", "Insights & Trends", "ML Models", "About"
])

# --- TAB A: NATIONAL OVERVIEW ---
with tab_nat:
    st.markdown('<div class="insight-header"><h3>National Performance Dashboard</h3></div>', unsafe_allow_html=True)
    
    # Latest year analysis
    df_2023 = df_all[df_all['Year'] == 2023]
    
    # KPI Cards
    kcols = st.columns(4, gap="medium")
    with kcols[0]: 
        st.markdown(draw_enhanced_kpi("Completion Rate", f"{round(df_2023['Completion_Rate'].mean(),1)}%", "2023 National Avg", "#3b82f6"), unsafe_allow_html=True)
    with kcols[1]: 
        st.markdown(draw_enhanced_kpi("Cohort Survival", f"{round(df_2023['Cohort_Survival_Rate'].mean(),1)}%", "Retention Strength", "#10b981"), unsafe_allow_html=True)
    with kcols[2]: 
        st.markdown(draw_enhanced_kpi("Participation", f"{round(df_2023['Participation_Rate'].mean(),1)}%", "School Access", "#f59e0b"), unsafe_allow_html=True)
    with kcols[3]: 
        st.markdown(draw_enhanced_kpi("Gender Parity", f"{round(df_2023['Gender_Parity_Index'].mean(),2)}", "Equity Index", "#8b5cf6"), unsafe_allow_html=True)
    
    st.write("")
    
    # Map & Overview
    col_map, col_overview = st.columns([1.5, 1], gap="large")
    
    with col_map:
        st.subheader("Geographic Distribution")
        sel_metric = st.selectbox(
            "Select Metric", 
            ["Completion_Rate", "Cohort_Survival_Rate", "Participation_Rate", "Gender_Parity_Index"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        @st.fragment
        def render_nat_map(metric):
            map_agg = df_2023.groupby('Geolocation')[metric].mean().reset_index()
            val_dict = map_agg.set_index('Geolocation')[metric].to_dict()
            if ph_geojson:
                for f in ph_geojson['features']:
                    name = f['properties'].get('REGION')
                    f['properties']['MapVal'] = f"{val_dict.get(name, 0):.2f}"
                m = folium.Map(location=[12.87, 121.77], zoom_start=5, tiles='CartoDB Positron')
                folium.Choropleth(geo_data=ph_geojson, data=map_agg, columns=['Geolocation', metric], 
                                 key_on='feature.properties.REGION', fill_color='Blues', fill_opacity=0.75, 
                                 line_weight=2, line_color='white').add_to(m)
                st_folium(m, width="100%", height=500)
        
        render_nat_map(sel_metric)
    
    with col_overview:
        st.subheader("Regional Rankings (2023)")
        rank_data = df_2023.groupby('Geolocation')['Completion_Rate'].mean().sort_values(ascending=False).reset_index()
        rank_data.columns = ['Region', 'Rate']
        rank_data['Rank'] = range(1, len(rank_data) + 1)
        
        fig_rank = px.bar(
            rank_data.head(10),
            y='Region',
            x='Rate',
            orientation='h',
            color='Rate',
            color_continuous_scale='Blues',
            template='plotly_white'
        )
        fig_rank.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_rank, use_container_width=True)
    
    st.write("---")
    
    # National Trends
    col_l1, col_l2 = st.columns(2, gap="large")
    
    with col_l1:
        st.subheader("Core Metrics Trend (2002-2023)")
        nat_trend = df_all.groupby('Year')[['Completion_Rate', 'Cohort_Survival_Rate', 'Participation_Rate']].mean().reset_index()
        fig_trend = px.line(
            nat_trend, 
            x='Year', 
            y=['Completion_Rate', 'Cohort_Survival_Rate', 'Participation_Rate'],
            markers=True,
            template='plotly_white',
            color_discrete_map={
                'Completion_Rate': '#3b82f6',
                'Cohort_Survival_Rate': '#10b981',
                'Participation_Rate': '#f59e0b'
            }
        )
        fig_trend.update_traces(marker=dict(size=7), line=dict(width=3))
        fig_trend.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_l2:
        st.subheader("Gender Parity Progress")
        nat_gpi = df_all.groupby('Year')['Gender_Parity_Index'].mean().reset_index()
        fig_gpi = px.line(
            nat_gpi, 
            x='Year', 
            y='Gender_Parity_Index',
            markers=True,
            template='plotly_white',
            color_discrete_sequence=['#8b5cf6']
        )
        fig_gpi.add_hline(y=1.0, line_dash="dash", line_color="#ef4444", annotation_text="Target: 1.0")
        fig_gpi.update_traces(marker=dict(size=8), line=dict(width=3))
        fig_gpi.update_layout(height=400, hovermode='x')
        st.plotly_chart(fig_gpi, use_container_width=True)

# --- TAB B: REGIONAL ANALYSIS ---
with tab_reg:
    @st.fragment
    def regional_content():
        st.markdown('<div class="insight-header"><h3>Region Specific Analysis</h3></div>', unsafe_allow_html=True)
        
        # Filters
        c_f1, c_f2, c_f3, c_f4 = st.columns(4, gap="medium")
        with c_f1: 
            sel_reg = st.selectbox("Region", sorted(df_all['Geolocation'].unique()))
        with c_f2: 
            sel_lvl = st.selectbox("Education Level", ["All", "Elementary", "Junior High", "Senior High"])
        with c_f3: 
            sel_sx = st.radio("Sex", ["All", "Male", "Female"], horizontal=True)
        with c_f4: 
            yr_range = st.slider("Year Range", 2002, 2023, (2015, 2023))
        
        # Filtering
        d_reg = df_all[(df_all['Geolocation'] == sel_reg) & (df_all['Year'] >= yr_range[0]) & (df_all['Year'] <= yr_range[1])]
        if sel_lvl != "All": 
            d_reg = d_reg[d_reg['Level of Education'] == sel_lvl]
        if sel_sx != "All": 
            d_reg = d_reg[d_reg['Sex'] == sel_sx]
        
        # Regional KPIs with trend
        latest_val = d_reg[d_reg['Year'] == d_reg['Year'].max()]
        earliest_val = d_reg[d_reg['Year'] == d_reg['Year'].min()]
        
        rk1, rk2, rk3, rk4 = st.columns(4, gap="medium")
        
        cr_change = latest_val['Completion_Rate'].mean() - earliest_val['Completion_Rate'].mean()
        with rk1: 
            trend_icon = "📈" if cr_change > 0 else "📉"
            st.markdown(draw_enhanced_kpi(f"Completion {trend_icon}", f"{round(latest_val['Completion_Rate'].mean(),1)}%", f"+{cr_change:.1f}% since {yr_range[0]}"), unsafe_allow_html=True)
        
        csr_change = latest_val['Cohort_Survival_Rate'].mean() - earliest_val['Cohort_Survival_Rate'].mean()
        with rk2: 
            trend_icon = "📈" if csr_change > 0 else "📉"
            st.markdown(draw_enhanced_kpi(f"Survival {trend_icon}", f"{round(latest_val['Cohort_Survival_Rate'].mean(),1)}%", f"+{csr_change:.1f}% change", "#10b981"), unsafe_allow_html=True)
        
        pr_change = latest_val['Participation_Rate'].mean() - earliest_val['Participation_Rate'].mean()
        with rk3: 
            trend_icon = "📈" if pr_change > 0 else "📉"
            st.markdown(draw_enhanced_kpi(f"Participation {trend_icon}", f"{round(latest_val['Participation_Rate'].mean(),1)}%", f"+{pr_change:.1f}% change", "#f59e0b"), unsafe_allow_html=True)
        
        gpi_val = latest_val['Gender_Parity_Index'].mean()
        gpi_status = "✓ Equal" if 0.95 <= gpi_val <= 1.05 else "↑ Favors F" if gpi_val > 1.05 else "↓ Favors M"
        with rk4: 
            st.markdown(draw_enhanced_kpi(f"Gender Parity {gpi_status}", f"{round(gpi_val,2)}", "Equity Index", "#8b5cf6"), unsafe_allow_html=True)
        
        st.write("---")
        
        col_r1, col_r2 = st.columns(2, gap="large")
        
        with col_r1:
            st.subheader("Regional Metric Trends")
            reg_trend = d_reg.groupby('Year')[['Completion_Rate', 'Cohort_Survival_Rate', 'Participation_Rate']].mean().reset_index()
            fig_reg = px.line(
                reg_trend, 
                x='Year', 
                y=['Completion_Rate', 'Cohort_Survival_Rate', 'Participation_Rate'],
                markers=True,
                template='plotly_white',
                color_discrete_map={
                    'Completion_Rate': '#3b82f6',
                    'Cohort_Survival_Rate': '#10b981',
                    'Participation_Rate': '#f59e0b'
                }
            )
            fig_reg.update_traces(marker=dict(size=6), line=dict(width=3))
            fig_reg.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_reg, use_container_width=True)
        
        with col_r2:
            st.subheader("Gender Parity Trend")
            reg_gpi = d_reg.groupby('Year')['Gender_Parity_Index'].mean().reset_index()
            fig_rgpi = px.line(
                reg_gpi, 
                x='Year', 
                y='Gender_Parity_Index',
                markers=True,
                template='plotly_white',
                color_discrete_sequence=['#8b5cf6']
            )
            fig_rgpi.add_hline(y=1.0, line_dash="dash", line_color="#ef4444")
            fig_rgpi.update_traces(marker=dict(size=8), line=dict(width=3))
            fig_rgpi.update_layout(height=400, hovermode='x')
            st.plotly_chart(fig_rgpi, use_container_width=True)
        
        st.write("---")
        col_rs1, col_rs2 = st.columns(2, gap="large")
        
        with col_rs1:
            st.subheader("Participation vs Completion")
            fig_corr = px.scatter(
                d_reg, 
                x="Participation_Rate", 
                y="Completion_Rate", 
                size="Cohort_Survival_Rate",
                color="Year",
                template="plotly_white",
                size_max=30,
                color_continuous_scale="Viridis"
            )
            fig_corr.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='white')))
            fig_corr.update_layout(height=400, hovermode='closest')
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col_rs2:
            st.subheader("Gender Gap Analysis")
            dumb_df = latest_val.pivot_table(index='Level of Education', columns='Sex', values='Completion_Rate', aggfunc='mean').reset_index()
            if 'Male' in dumb_df.columns and 'Female' in dumb_df.columns:
                fig_dumb = go.Figure()
                for i, row in dumb_df.iterrows():
                    fig_dumb.add_shape(
                        type='line', 
                        x0=row['Male'], x1=row['Female'], 
                        y0=row['Level of Education'], y1=row['Level of Education'], 
                        line=dict(color='#cbd5e1', width=3)
                    )
                fig_dumb.add_trace(go.Scatter(x=dumb_df['Male'], y=dumb_df['Level of Education'], mode='markers', name='Male', marker=dict(color='#3b82f6', size=12, line=dict(width=2, color='white'))))
                fig_dumb.add_trace(go.Scatter(x=dumb_df['Female'], y=dumb_df['Level of Education'], mode='markers', name='Female', marker=dict(color='#ec4899', size=12, line=dict(width=2, color='white'))))
                fig_dumb.update_layout(template='plotly_white', height=400, hovermode='closest')
                st.plotly_chart(fig_dumb, use_container_width=True)
        
        st.write("---")
        st.subheader("Performance Heatmap")
        heat_pivot = d_reg.pivot_table(index='Level of Education', columns='Year', values='Completion_Rate', aggfunc='mean')
        fig_heat = px.imshow(
            heat_pivot, 
            color_continuous_scale="RdYlGn",
            aspect="auto", 
            template="plotly_white",
            color_continuous_midpoint=80
        )
        fig_heat.update_layout(height=350)
        st.plotly_chart(fig_heat, use_container_width=True)
    
    regional_content()

# --- TAB C: INSIGHTS & TRENDS ---
with tab_insights:
    st.markdown('<div class="insight-header"><h3>Data Analysis & Key Insights</h3></div>', unsafe_allow_html=True)
    
    col_i1, col_i2 = st.columns(2, gap="large")
    
    with col_i1:
        st.subheader("Top Performing Regions (2023)")
        top_regions = df_all[df_all['Year'] == 2023].groupby('Geolocation')['Completion_Rate'].mean().sort_values(ascending=False).head(5)
        for idx, (region, value) in enumerate(top_regions.items(), 1):
            st.markdown(f"""
            <div class="insight-card">
                <strong>#{idx} {region}</strong> - {value:.1f}%
                <div class="progress-bar"><div class="progress-fill" style="width: {min(value, 100)}%"></div></div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_i2:
        st.subheader("Regions Needing Support (2023)")
        bottom_regions = df_all[df_all['Year'] == 2023].groupby('Geolocation')['Completion_Rate'].mean().sort_values(ascending=True).head(5)
        for idx, (region, value) in enumerate(bottom_regions.items(), 1):
            st.markdown(f"""
            <div class="insight-card">
                <strong>⚠️ {region}</strong> - {value:.1f}%
                <div class="progress-bar"><div class="progress-fill" style="width: {min(value, 100)}%"></div></div>
            </div>
            """, unsafe_allow_html=True)
    
    st.write("---")
    
    col_i3, col_i4 = st.columns(2, gap="large")
    
    with col_i3:
        st.subheader("Fastest Improving Regions")
        region_change = []
        for region in df_all['Geolocation'].unique():
            region_data = df_all[df_all['Geolocation'] == region]
            early = region_data[region_data['Year'] == region_data['Year'].min()]['Completion_Rate'].mean()
            late = region_data[region_data['Year'] == region_data['Year'].max()]['Completion_Rate'].mean()
            region_change.append({'Region': region, 'Change': late - early})
        
        change_df = pd.DataFrame(region_change).sort_values('Change', ascending=False).head(5)
        fig_improve = px.bar(change_df, x='Change', y='Region', orientation='h', color='Change', color_continuous_scale='Greens', template='plotly_white')
        fig_improve.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_improve, use_container_width=True)
    
    with col_i4:
        st.subheader("Gender Parity Status (2023)")
        gpi_2023 = df_all[df_all['Year'] == 2023].groupby('Geolocation')['Gender_Parity_Index'].mean().reset_index()
        gpi_2023['Status'] = gpi_2023['Gender_Parity_Index'].apply(lambda x: 'Equal ✓' if 0.95 <= x <= 1.05 else 'Favors Female ↑' if x > 1.05 else 'Favors Male ↓')
        
        status_counts = gpi_2023['Status'].value_counts()
        colors = {'Equal ✓': '#10b981', 'Favors Female ↑': '#ec4899', 'Favors Male ↓': '#3b82f6'}
        fig_gpi_status = px.pie(values=status_counts.values, names=status_counts.index, color_discrete_map=colors, template='plotly_white')
        fig_gpi_status.update_layout(height=350)
        st.plotly_chart(fig_gpi_status, use_container_width=True)

# --- TAB D: ML MODELS ---
with tab_ml:
    st.markdown('<div class="insight-header"><h3>Machine Learning & Predictive Analytics</h3></div>', unsafe_allow_html=True)
    
    if kmeans_model is not None:
        st.subheader("Regional Performance Clustering")
        st.write("This model groups regions into 'Profiles' based on their average Participation, Completion, and Cohort Survival rates.")

        # Preprocess: Replicating ModelHandler logic
        # Filter out 0 values for accuracy
        df_filtered = df_all[df_all['Cohort_Survival_Rate'] > 0]

        # Group by Region to get the profile
        regional_profile = df_filtered.groupby('Geolocation').agg({
            'Participation_Rate': 'mean',
            'Completion_Rate': 'mean',
            'Cohort_Survival_Rate': 'mean',
        }).reset_index()

        # Scale Features
        features = ['Participation_Rate', 'Completion_Rate', 'Cohort_Survival_Rate']

        # Apply Model
        try:
            regional_profile['clusters'] = kmeans_model.predict(kmeans_scaler(regional_profile[features]))
            regional_profile['clusters'] = regional_profile['clusters'].astype(str) # Convert to string for discrete color scaling

            # Visualization
            col_chart, col_list = st.columns([2, 1])
            
            with col_chart:
                fig_clusters = px.scatter(
                    regional_profile, 
                    x="Participation_Rate", 
                    y="Completion_Rate",
                    color="clusters",
                    size="Cohort_Survival_Rate",
                    hover_name="Geolocation",
                    template="plotly_white",
                    title="Cluster Map: Participation vs Completion",
                    labels={"clusters": "Cluster Group"},
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                st.plotly_chart(fig_clusters, use_container_width=True)
            
            with col_list:
                st.write("**Regions by Cluster**")
                for cluster in sorted(regional_profile['clusters'].unique()):
                    regions = regional_profile[regional_profile['clusters'] == cluster]['Geolocation'].tolist()
                    st.info(f"**Cluster {cluster}**: {', '.join(regions)}")
                    
        except Exception as e:
            st.error(f"Could not run clustering: {e}")
    else:
        st.warning("⚠️ `import_models/kmeans_model.pkl` not found. Please ensure the model file is in the correct directory.")

    st.write("---")
    st.subheader("Coming Soon")
    st.info("Additional forecasting models and causal analysis (VAR) are currently under development.")
    st.write("---")
    
    # Placeholders for other models
    col_ml1, col_ml3 = st.columns(2)
    with col_ml1:
        st.subheader("Time Series Forecasting")
        st.info("Forecasting models are currently under development.")
    with col_ml3:
        st.subheader("Causal Analysis")
        st.info("Vector Autoregression analysis is currently under development.")
        
# --- TAB E: ABOUT ---
with tab_about:
    st.markdown('<div class="insight-header"><h3>About This Dashboard</h3></div>', unsafe_allow_html=True)
    
    col_about1, col_about2 = st.columns([2, 1], gap="large")
    
    with col_about1:
        st.markdown("""
        ## **Dashboard Features**
        
        ### **National Overview**
        - Aggregate performance metrics across Philippines
        - Interactive geographic map with regional comparisons
        - 20+ years of historical trend analysis
        - Top performing regions ranking
        
        ### **Regional Analysis**
        - Deep-dive filtering by region, education level, and demographics
        - Year-range selection for custom period analysis
        - Correlation analysis and gender gap visualization
        - Performance heatmaps over time
        
        ### **Insights & Trends**
        - Top and bottom performers identification
        - Rapid improvement tracking
        - Gender equity status dashboard
        - Actionable performance metrics
        
        ### **Machine Learning** *(Coming Soon)*
        - Predictive completion rate forecasting
        - Regional clustering by performance
        - Causal relationship analysis
        
        ---
        
        ## **Key Metrics Explained**
        """)
        
        # Create metrics table
        metrics_data = {
            'Metric': ['Completion Rate', 'Cohort Survival Rate', 'Participation Rate', 'Gender Parity Index'],
            'Definition': [
                '% of students graduating from current level',
                '% of students retained without dropouts',
                '% of age-appropriate population enrolled',
                'Ratio of female to male achievement'
            ],
            'Target': ['95%', '90%', '100%', '1.0']
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        ---
        
        ## **Data Specifications**
        - **Source**: Department of Education (DepEd) & Philippine Statistics Authority (PSA)
        - **Coverage**: 2002-2023 | 17 Regions | 3 Education Levels | 2 Demographics
        - **Data Points**: 3,400+ records
        - **Update Frequency**: Annual
        - **Note**: Senior High School data officially begins in 2017
        
        ---
        
        ## **Development Team**
        
        **GiMaTag Analytics**
        - Dave Shanna Marie E. Gigawin
        - Waken Cean C. Maclang
        - Allan C. Tagle
        
        **Institution**: USeP College of Information and Computing
        
        """)
        st.write("---")
        st.markdown(f"**Last Updated**: {datetime.now().strftime('%B %d, %Y')}")
        st.write("2024 © Team GiMaTag. All rights reserved.")
        
    with col_about2:
        st.subheader("At a Glance")
        st.metric("Total Records", "3,400+")
        st.metric("Regions", "17")
        st.metric("Years Analyzed", "22 (2002-2023)")
        st.metric("Last Update", "December 2024")