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

# --- ENHANCED CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    [data-testid="stSidebarNav"] {display: none;}
    
    /* TOP-RIGHT NAVBAR */
    .stTabs [data-baseweb="tab-list"] {
        display: flex; justify-content: flex-end; 
        background-color: transparent; border-bottom: none !important;
        padding-right: 50px; gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b !important; background-color: transparent !important;
        border: none !important; font-weight: 700; font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #0f172a !important; border-bottom: 3px solid #3b82f6 !important;
    }

    /* UPGRADED KPI CARDS */
    .kpi-card {
        background: white;
        padding: 20px 10px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 5px solid #3b82f6;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .kpi-card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
    .kpi-label { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; font-weight: 600; margin-bottom: 5px; }
    .kpi-value { font-size: 28px; font-weight: 800; color: #1e293b; }
    .kpi-subtitle { font-size: 11px; color: #94a3b8; }

    .insight-header { margin-top: 30px; margin-bottom: 20px; border-left: 5px solid #3b82f6; padding-left: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_assets():
    # Adjusted path to match root directory or wrangling folder
    file_path = 'Cleaned_Philippines_Education_Statistics.csv'
    if not os.path.exists(file_path):
        file_path = 'data_wrangling/Cleaned_Philippines_Education_Statistics.csv'
        
    df = pd.read_csv(file_path)
    geojson = None
    if os.path.exists('philippines_regions.json'):
        with open('philippines_regions.json', 'r', encoding='utf-8') as f:
            geojson = json.load(f)
    return df, geojson

df_all, ph_geojson = load_assets()

# --- KPI HELPER FUNCTION ---
def draw_enhanced_kpi(label, value, subtitle, color="#3b82f6"):
    return f"""
        <div class="kpi-card" style="border-left-color: {color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
    """

# --- HEADER ---
st.title("Philippines Education SDG 4 Tracker")

# --- NAVIGATION TABS ---
tab_nat, tab_reg, tab_ml, tab_about = st.tabs([
    "National Overview", "Regional Analysis", "Machine Learning", "About"
])

# --- TAB A: NATIONAL OVERVIEW ---
with tab_nat:
    st.markdown('<div class="insight-header"><h3>A. National Performance Dashboard</h3></div>', unsafe_allow_html=True)
    
    # 1. National KPIs
    df_2023 = df_all[df_all['Year'] == 2023]
    kcols = st.columns(4)
    with kcols[0]: st.markdown(draw_enhanced_kpi("Completion Rate", f"{round(df_2023['Completion_Rate'].mean(),1)}%", "National Avg (2023)", "#3b82f6"), unsafe_allow_html=True)
    with kcols[1]: st.markdown(draw_enhanced_kpi("Cohort Survival", f"{round(df_2023['Cohort_Survival_Rate'].mean(),1)}%", "Retention Strength", "#10b981"), unsafe_allow_html=True)
    with kcols[2]: st.markdown(draw_enhanced_kpi("Participation", f"{round(df_2023['Participation_Rate'].mean(),1)}%", "Access to School", "#f59e0b"), unsafe_allow_html=True)
    with kcols[3]: st.markdown(draw_enhanced_kpi("Gender Parity", f"{round(df_2023['Gender_Parity_Index'].mean(),2)}", "Index (Target 1.0)", "#8b5cf6"), unsafe_allow_html=True)

    # 2. Map & Scatter
    st.write("")
    col_map, col_scat = st.columns([1.5, 1])
    with col_map:
        st.subheader("Geographic Distribution")
        sel_metric = st.selectbox("Metric for Map", ["Completion_Rate", "Cohort_Survival_Rate", "Participation_Rate", "Gender_Parity_Index"], key="nat_map_sel")
        @st.fragment
        def render_nat_map(metric):
            with st.spinner("📍 Updating map..."):
                map_agg = df_2023.groupby('Geolocation')[metric].mean().reset_index()
                val_dict = map_agg.set_index('Geolocation')[metric].to_dict()
                if ph_geojson:
                    for f in ph_geojson['features']:
                        name = f['properties'].get('REGION')
                        f['properties']['MapVal'] = f"{val_dict.get(name, 0):.2f}"
                    m = folium.Map(location=[12.87, 121.77], zoom_start=5, tiles='CartoDB Positron')
                    folium.Choropleth(geo_data=ph_geojson, data=map_agg, columns=['Geolocation', metric], 
                                     key_on='feature.properties.REGION', fill_color='YlGnBu', fill_opacity=0.7).add_to(m)
                    folium.GeoJson(ph_geojson, style_function=lambda x: {'fillColor': '#ffffff00', 'color': '#ffffff00'}, 
                                   tooltip=folium.GeoJsonTooltip(fields=['REGION', 'MapVal'], aliases=['Region:', 'Value:'])).add_to(m)
                    st_folium(m, width="100%", height=500, key=f"nat_map_{metric}")
        render_nat_map(sel_metric)

    with col_scat:
        st.subheader("Participation vs Graduation")
        st.plotly_chart(px.scatter(df_2023, x="Participation_Rate", y="Completion_Rate", size="Cohort_Survival_Rate",
                                  color="Level of Education", hover_name="Geolocation", template="plotly_white"), use_container_width=True)

    # 3. National Trends
    st.write("---")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.subheader("CR, CSR, PR National Trend")
        nat_trend = df_all.groupby('Year')[['Completion_Rate', 'Cohort_Survival_Rate', 'Participation_Rate']].mean().reset_index()
        st.plotly_chart(px.line(nat_trend, x='Year', y=['Completion_Rate', 'Cohort_Survival_Rate', 'Participation_Rate'], markers=True, template='plotly_white'), use_container_width=True)
    with col_l2:
        st.subheader("GPI National Trend")
        nat_gpi = df_all.groupby('Year')['Gender_Parity_Index'].mean().reset_index()
        st.plotly_chart(px.line(nat_gpi, x='Year', y='Gender_Parity_Index', markers=True, template='plotly_white', color_discrete_sequence=['#3b82f6']), use_container_width=True)

# --- TAB B: REGIONAL ANALYSIS ---
with tab_reg:
    @st.fragment
    def regional_content():
        st.markdown('<div class="insight-header"><h3>B. Region Specific Deep Dive</h3></div>', unsafe_allow_html=True)
        
        # Filters
        c_f1, c_f2, c_f3, c_f4 = st.columns(4)
        with c_f1: sel_reg = st.selectbox("Choose A Region", sorted(df_all['Geolocation'].unique()))
        with c_f2: sel_lvl = st.selectbox("Level of Education", ["All", "Elementary", "Junior High", "Senior High"])
        with c_f3: sel_sx = st.radio("Sex", ["All", "Male", "Female"], horizontal=True)
        with c_f4: yr_range = st.slider("Select Year Range", 2002, 2023, (2015, 2023))

        # Filtering
        d_reg = df_all[(df_all['Geolocation'] == sel_reg) & (df_all['Year'] >= yr_range[0]) & (df_all['Year'] <= yr_range[1])]
        if sel_lvl != "All": d_reg = d_reg[d_reg['Level of Education'] == sel_lvl]
        if sel_sx != "All": d_reg = d_reg[d_reg['Sex'] == sel_sx]
        
        # Regional KPIs
        latest_val = d_reg[d_reg['Year'] == d_reg['Year'].max()]
        rk1, rk2, rk3, rk4 = st.columns(4)
        with rk1: st.markdown(draw_enhanced_kpi("Completion", f"{round(latest_val['Completion_Rate'].mean(),1)}%", "Regional Value"), unsafe_allow_html=True)
        with rk2: st.markdown(draw_enhanced_kpi("Survival", f"{round(latest_val['Cohort_Survival_Rate'].mean(),1)}%", "Regional Value", "#10b981"), unsafe_allow_html=True)
        with rk3: st.markdown(draw_enhanced_kpi("Participation", f"{round(latest_val['Participation_Rate'].mean(),1)}%", "Regional Value", "#f59e0b"), unsafe_allow_html=True)
        with rk4: st.markdown(draw_enhanced_kpi("Gender Parity", f"{round(latest_val['Gender_Parity_Index'].mean(),2)}", "Regional Value", "#8b5cf6"), unsafe_allow_html=True)

        st.write("---")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.subheader("Regional Metric Trends")
            reg_trend = d_reg.groupby('Year')[['Completion_Rate', 'Cohort_Survival_Rate', 'Participation_Rate']].mean().reset_index()
            st.plotly_chart(px.line(reg_trend, x='Year', y=['Completion_Rate', 'Cohort_Survival_Rate', 'Participation_Rate'], markers=True, template='plotly_white'), use_container_width=True)
        with col_r2:
            st.subheader("Regional GPI Trend")
            reg_gpi = d_reg.groupby('Year')['Gender_Parity_Index'].mean().reset_index()
            st.plotly_chart(px.line(reg_gpi, x='Year', y='Gender_Parity_Index', markers=True, template='plotly_white', color_discrete_sequence=['#3b82f6']), use_container_width=True)

        # DEEPER INSIGHTS (RESTORED)
        st.write("---")
        col_rs1, col_rs2 = st.columns(2)
        with col_rs1:
            st.subheader("Correlation: Participation vs Completion")
            st.plotly_chart(px.scatter(d_reg, x="Participation_Rate", y="Completion_Rate", size="Cohort_Survival_Rate", color="Year", template="plotly_white"), use_container_width=True)
        with col_rs2:
            st.subheader("Gender Completion Gap (Dumbbell)")
            dumb_df = latest_val.pivot_table(index='Geolocation', columns='Sex', values='Completion_Rate').reset_index()
            if 'Male' in dumb_df.columns and 'Female' in dumb_df.columns:
                fig_dumb = go.Figure()
                for i, row in dumb_df.iterrows():
                    fig_dumb.add_shape(type='line', x0=row['Male'], x1=row['Female'], y0=row['Geolocation'], y1=row['Geolocation'], line=dict(color='#cbd5e1', width=3))
                fig_dumb.add_trace(go.Scatter(x=dumb_df['Male'], y=dumb_df['Geolocation'], mode='markers', name='Male', marker=dict(color='#94a3b8', size=10)))
                fig_dumb.add_trace(go.Scatter(x=dumb_df['Female'], y=dumb_df['Geolocation'], mode='markers', name='Female', marker=dict(color='#3b82f6', size=10)))
                fig_dumb.update_layout(template='plotly_white', height=400, margin=dict(t=10))
                st.plotly_chart(fig_dumb, use_container_width=True)

        st.write("---")
        st.subheader("Historical Performance Matrix (Heatmap)")
        heat_pivot = d_reg.pivot_table(index='Geolocation', columns='Year', values='Completion_Rate', aggfunc='mean')
        st.plotly_chart(px.imshow(heat_pivot, color_continuous_scale="YlGnBu", aspect="auto", template="plotly_white"), use_container_width=True)

    regional_content()

# --- TAB C: MACHINE LEARNING (Placeholders) ---
with tab_ml:
    st.markdown('<div class="insight-header"><h3>C. Machine Learning Models</h3></div>', unsafe_allow_html=True)
    st.info("🚧 This section is under development. Placeholders for Waken's integration.")
    col_ml1, col_ml2, col_ml3 = st.columns(3)
    with col_ml1:
        st.subheader("Completion Predictor")
        st.markdown("*Future Forecasting using Linear Regression.*")
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    with col_ml2:
        st.subheader("Clustering")
        st.markdown("*Grouping regions based on Performance similarity.*")
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103533.png", width=100)
    with col_ml3:
        st.subheader("Vector Autoregression")
        st.markdown("*Analyzing Lead-Lag Metric interactions.*")
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103460.png", width=100)

# --- TAB D: ABOUT ---
with tab_about:
    st.markdown('<div class="insight-header"><h3>D. Descriptive Narrative</h3></div>', unsafe_allow_html=True)
    
    col_about1, col_about2 = st.columns([2, 1])
    with col_about1:
        st.markdown("""
        ### **How to Use This Dashboard**
        1. **National Overview**: A high-level look at the Philippines. Use the **Map Metric Selector** to see how regions compare visually across different indicators.
        2. **Regional Analysis**: Select your home region to see localized trends. Use the **Year Slider** to focus on specific historical periods.
        3. **Machine Learning**: Predictive insights into future completion rates (Coming Soon).

        ### **Metric Legend**
        - **Completion Rate (CR)**: Percentage of students who graduate from their current level.
        - **Cohort Survival Rate (CSR)**: Efficiency of the system in preventing dropouts.
        - **Participation Rate (PR)**: Enrollment percentage relative to the age-appropriate population.
        - **Gender Parity Index (GPI)**: Equity measure (Target is **1.0**). Below 1.0 favors Males, above 1.0 favors Females.

        ### **Dataset Details**
        - **Data Source**: DepEd and PSA Administrative Records (2002-2023).
        - **Senior High Note**: SHS data officially begins in **2017**.
        """)
        st.write("---")
        st.markdown("#### **Developed by GiMaTag: Dave Shanna Marie E. Gigawin, Waken Cean C. Maclang, and Allan C. Tagle**")
    
    with col_about2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Sustainable_Development_Goal_4.png/1200px-Sustainable_Development_Goal_4.png", caption="Goal 4: Quality Education")