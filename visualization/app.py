import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import folium
from streamlit_folium import st_folium
from folium import GeoJson, GeoJsonTooltip

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Philippines SDG 4 Analytics", 
    page_icon="🎓", 
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f1f5f9; }
    
    /* TOP-RIGHT NAVBAR TABS Styling */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: flex-end; 
        background-color: transparent; 
        border-bottom: none !important;
        padding-right: 50px;
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b !important; 
        background-color: transparent !important;
        border: none !important;
        font-weight: 700;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #0f172a !important;
        border-bottom: 3px solid #3b82f6 !important;
    }

    /* Metric Cards */
    .metric-card {
        background: white; padding: 24px; border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0; text-align: center;
        height: 140px; display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 14px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { font-size: 32px; font-weight: 800; color: #1e293b; }
    
    /* Section Headers */
    .insight-header { margin-top: 40px; margin-bottom: 20px; border-left: 5px solid #3b82f6; padding-left: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_assets():
    df = pd.read_csv('data_wrangling/Cleaned_Philippines_Education_Statistics.csv')
    geojson = None
    if os.path.exists('philippines_regions.json'):
        with open('philippines_regions.json', 'r', encoding='utf-8') as f:
            geojson = json.load(f)
    return df, geojson

df_all, ph_geojson = load_assets()

# --- HEADER ---
st.title("Philippines Education SDG 4 Tracker")

# --- NAVIGATION TABS ---
tab_overview, tab_analysis = st.tabs(["National Overview", "Detailed Analysis"])

with tab_overview:
    st.markdown('<div class="insight-header"><h3>National Geographic Overview (2023)</h3></div>', unsafe_allow_html=True)
    
    # KPIs
    df_2023 = df_all[df_all['Year'] == 2023]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Completion</div><div class="metric-value">{round(df_2023["Completion_Rate"].mean(),1)}%</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">Survival</div><div class="metric-value">{round(df_2023["Cohort_Survival_Rate"].mean(),1)}%</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Participation</div><div class="metric-value">{round(df_2023["Participation_Rate"].mean(),1)}%</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">Gender Index</div><div class="metric-value">{round(df_2023["Gender_Parity_Index"].mean(),2)}</div></div>', unsafe_allow_html=True)

    st.write("")
    
    @st.fragment
    def render_landing_map():
        with st.spinner("📍 Loading National Map..."):
            map_agg = df_2023.groupby('Geolocation')['Completion_Rate'].mean().reset_index()
            val_dict = map_agg.set_index('Geolocation')['Completion_Rate'].to_dict()
            if ph_geojson:
                for feature in ph_geojson['features']:
                    name = feature['properties'].get('REGION')
                    feature['properties']['Hover_Val'] = f"{val_dict.get(name, 0):.1f}%"
                m = folium.Map(location=[12.87, 121.77], zoom_start=6, tiles='CartoDB Positron')
                folium.Choropleth(geo_data=ph_geojson, data=map_agg, columns=['Geolocation', 'Completion_Rate'], key_on='feature.properties.REGION', fill_color='YlGnBu', fill_opacity=0.8, line_opacity=0.2).add_to(m)
                folium.GeoJson(ph_geojson, style_function=lambda x: {'fillColor': '#ffffff00', 'color': '#ffffff00'}, tooltip=folium.GeoJsonTooltip(fields=['REGION', 'Hover_Val'], aliases=['Region:', 'Rate:'])).add_to(m)
                st_folium(m, width="100%", height=600, key="landing_map_static")
    render_landing_map()

with tab_analysis:
    @st.fragment
    def analysis_content():
        st.markdown('<div class="insight-header"><h3>Interactive Regional Analysis</h3></div>', unsafe_allow_html=True)
        
        # 1. Filters (Row 1)
        f1, f2, f3, f4 = st.columns(4)
        with f1: sel_region = st.selectbox("Region Focus", ["All"] + sorted(df_all['Geolocation'].unique().tolist()), index=0)
        with f2: sel_level = st.selectbox("Education Level", ["All", "Elementary", "Junior High", "Senior High"], index=0)
        with f3: sel_sex = st.radio("Gender Focus", ["All", "Male", "Female"], index=0, horizontal=True)
        with f4: sel_year = st.selectbox("Target Year", sorted(df_all['Year'].unique(), reverse=True), index=0)

        # Apply Filters
        d = df_all[df_all['Year'] == int(sel_year)]
        if sel_region != "All": d = d[d['Geolocation'] == sel_region]
        if sel_level != "All": d = d[d['Level of Education'] == sel_level]
        if sel_sex != "All": d = d[d['Sex'] == sel_sex]

        # 2. Main Analytics (Row 2)
        st.write("---")
        col_t, col_g = st.columns([2, 1])
        with col_t:
            st.subheader("Progress Trends")
            hist_df = df_all.copy()
            if sel_region != "All": hist_df = hist_df[hist_df['Geolocation'] == sel_region]
            if sel_level != "All": hist_df = hist_df[hist_df['Level of Education'] == sel_level]
            trend_agg = hist_df.groupby('Year')[['Completion_Rate', 'Cohort_Survival_Rate']].mean().reset_index()
            fig_trend = px.line(trend_agg, x='Year', y=['Completion_Rate', 'Cohort_Survival_Rate'], markers=True, template='plotly_white')
            st.plotly_chart(fig_trend, use_container_width=True)
        with col_g:
            st.subheader("Gender Parity Gauge")
            gpi_val = d['Gender_Parity_Index'].mean() if not d.empty else 1.0
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=gpi_val, gauge={'axis': {'range': [0.5, 1.5]}, 'bar': {'color': "#0f172a"}, 'steps': [{'range': [0.9, 1.1], 'color': "#dcfce7"}]}))
            fig_gauge.update_layout(height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)

        # 3. ADVANCED INSIGHTS (NEW Row 3)
        st.write("---")
        col_dumb, col_scat = st.columns(2)
        
        with col_dumb:
            st.subheader("Gender Gap Disparity")
            # Create Dumbbell Chart
            dumb_df = d.pivot_table(index='Geolocation', columns='Sex', values='Completion_Rate').reset_index()
            if 'Male' in dumb_df.columns and 'Female' in dumb_df.columns:
                fig_dumb = go.Figure()
                for i, row in dumb_df.iterrows():
                    fig_dumb.add_shape(type='line', x0=row['Male'], x1=row['Female'], y0=row['Geolocation'], y1=row['Geolocation'], line=dict(color='#cbd5e1', width=3))
                fig_dumb.add_trace(go.Scatter(x=dumb_df['Male'], y=dumb_df['Geolocation'], mode='markers', name='Male', marker=dict(color='#94a3b8', size=10)))
                fig_dumb.add_trace(go.Scatter(x=dumb_df['Female'], y=dumb_df['Geolocation'], mode='markers', name='Female', marker=dict(color='#3b82f6', size=10)))
                fig_dumb.update_layout(template='plotly_white', height=450, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Completion Rate (%)")
                st.plotly_chart(fig_dumb, use_container_width=True)

        with col_scat:
            st.subheader("Participation vs Graduation")
            fig_scat = px.scatter(d, x="Participation_Rate", y="Completion_Rate", color="Geolocation", size="Cohort_Survival_Rate", hover_name="Geolocation", template="plotly_white")
            fig_scat.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_scat, use_container_width=True)

        # 4. Long-term Matrix (Row 4)
        st.write("---")
        st.subheader("Regional Performance Matrix (10-Year History)")
        # Heatmap Data
        heat_df = df_all.copy()
        if sel_level != "All": heat_df = heat_df[heat_df['Level of Education'] == sel_level]
        if sel_sex != "All": heat_df = heat_df[heat_df['Sex'] == sel_sex]
        heat_pivot = heat_df[heat_df['Year'] >= 2013].pivot_table(index='Geolocation', columns='Year', values='Completion_Rate', aggfunc='mean')
        fig_heat = px.imshow(heat_pivot, labels=dict(x="Year", y="Region", color="Rate %"), color_continuous_scale="YlGnBu", aspect="auto")
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

    analysis_content()