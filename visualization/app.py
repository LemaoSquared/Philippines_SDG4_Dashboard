import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Philippines SDG 4 Dashboard", 
    page_icon="🎓", 
    layout="wide"
)

# --- COORDINATES FOR GEO DESIGN (Philippine Regions) ---
region_coords = {
    'NCR': [14.5995, 120.9842], 'CAR': [17.4705, 121.0926], 'Region I': [16.0236, 120.3326],
    'Region II': [17.8183, 121.8440], 'Region III': [15.4828, 120.7120], 'Region IV-A': [14.1008, 121.0794],
    'Region IV-B': [10.8503, 119.2707], 'Region V': [13.4350, 123.4751], 'Region VI': [10.9967, 122.5806],
    'Region VII': [10.0270, 123.4751], 'Region VIII': [11.3323, 124.9813], 'Region IX': [7.8202, 122.3845],
    'Region X': [8.1814, 124.4606], 'Region XI': [7.3042, 126.0893], 'Region XII': [6.4709, 124.8475],
    'Region XIII': [8.9620, 125.7506], 'BARMM': [7.0315, 124.3166]
}

# --- CUSTOM CSS FOR CLEAN & UNIQUE UI ---
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #0f172a; color: white; }
    [data-testid="stSidebar"] * { color: white !important; }
    
    .metric-card {
        background: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-bottom: 4px solid #3b82f6; text-align: center;
    }
    .metric-label { font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { font-size: 28px; font-weight: 800; color: #1e293b; }
    
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #1e293b; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING & WRANGLING ---
@st.cache_data
def load_and_clean_data():
    # Load
    cc_df = pd.read_csv('data_wrangling/Cleaned_Completion and Cohort Survival Rate.csv')
    gpi_df = pd.read_csv('data_wrangling/Cleaned_Gender Parity Index.csv')
    pr_df = pd.read_csv('data_wrangling/Cleaned_Participation Rate.csv')
    
    years = [str(y) for y in range(2002, 2024)]
    
    # Melt to Long Format
    cc_l = cc_df.melt(id_vars=['Geolocation', 'Level of Education', 'Sex', 'Metric'], 
                      value_vars=years, var_name='Year', value_name='Val')
    gpi_l = gpi_df.melt(id_vars=['Geolocation', 'Level of Education'], 
                        value_vars=years, var_name='Year', value_name='Val')
    pr_l = pr_df.melt(id_vars=['Geolocation', 'Sex'], 
                       value_vars=years, var_name='Year', value_name='Val')
    
    # Filter out 0 values for Senior High (Pre-2016) to avoid skewing averages
    cc_l = cc_l[cc_l['Val'] > 0]
    gpi_l = gpi_l[gpi_l['Val'] > 0]
    pr_l = pr_l[pr_l['Val'] > 0]
    
    return cc_l, gpi_l, pr_l

df_cc, df_gpi, df_pr = load_and_clean_data()

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Flag_of_the_Philippines.svg/255px-Flag_of_the_Philippines.svg.png", width=100)
    st.header("Dashboard Filters")
    
    sel_region = st.selectbox("📍 Select Geolocation", ["All"] + sorted(df_cc['Geolocation'].unique().tolist()))
    sel_level = st.selectbox("📚 Level of Education", ["All", "Elementary", "Junior High", "Senior High"])
    sel_sex = st.radio("👤 Sex", ["All", "Male", "Female"], horizontal=True)
    sel_year = st.select_slider("📅 Target Year", options=sorted(df_cc['Year'].unique(), reverse=True), value='2023')

# --- DATA FILTERING LOGIC ---
def apply_filters(df, r, l=None, s=None):
    d = df[df['Year'] == sel_year]
    if r != "All": d = d[d['Geolocation'] == r]
    if l and l != "All": d = d[d['Level of Education'] == l]
    if s and s != "All" and 'Sex' in d.columns: d = d[d['Sex'] == s]
    return d

f_cc = apply_filters(df_cc, sel_region, sel_level, sel_sex)
f_gpi = apply_filters(df_gpi, sel_region, sel_level)
f_pr = apply_filters(df_pr, sel_region, None, sel_sex)

# --- MAIN UI ---
st.title("🎓 Philippines Education SDG 4 Tracker")
st.markdown(f"Displaying data for **{sel_region}** | **{sel_level}** | **{sel_year}**")

# --- ROW 1: KPI CARDS ---
c1, c2, c3, c4 = st.columns(4)

def kpi_card(col, label, val, suffix="%"):
    with col:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{round(val,1) if val else 0}{suffix}</div></div>""", unsafe_allow_html=True)

kpi_card(c1, "Completion Rate", f_cc[f_cc['Metric']=='Completion Rate']['Val'].mean())
kpi_card(c2, "Cohort Survival", f_cc[f_cc['Metric']=='Cohort Survival Rate']['Val'].mean())
kpi_card(c3, "Participation Rate", f_pr['Val'].mean())
kpi_card(c4, "Gender Parity Index", f_gpi['Val'].mean(), suffix="")

st.write("")

# --- ROW 2: GEO DESIGN (MAP) ---
st.subheader("🗺️ National Geographic Performance")
map_data = df_cc[(df_cc['Year'] == sel_year) & (df_cc['Metric'] == 'Completion Rate')]
if sel_level != "All": map_data = map_data[map_data['Level of Education'] == sel_level]
map_agg = map_data.groupby('Geolocation')['Val'].mean().reset_index()
map_agg['Lat'] = map_agg['Geolocation'].map(lambda x: region_coords.get(x, [0,0])[0])
map_agg['Lon'] = map_agg['Geolocation'].map(lambda x: region_coords.get(x, [0,0])[1])

fig_map = px.scatter_geo(
    map_agg, lat='Lat', lon='Lon', size='Val', color='Val',
    hover_name='Geolocation', color_continuous_scale='Blues',
    projection="natural earth", scope='asia', title="Regional Completion Rate Distribution"
)
fig_map.update_geos(center=dict(lon=122, lat=12), projection_scale=6, visible=False, showcoastlines=True)
fig_map.update_layout(height=500, margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# --- ROW 3: TRENDS & GENDER ---
st.write("---")
col_t, col_g = st.columns([2, 1])

with col_t:
    st.subheader("📈 Progress Trend (2002-2023)")
    t_cc = df_cc.copy()
    if sel_region != "All": t_cc = t_cc[t_cc['Geolocation'] == sel_region]
    if sel_level != "All": t_cc = t_cc[t_cc['Level of Education'] == sel_level]
    if sel_sex != "All": t_cc = t_cc[t_cc['Sex'] == sel_sex]
    
    trend_data = t_cc.groupby(['Year', 'Metric'])['Val'].mean().reset_index()
    fig_trend = px.line(trend_data, x='Year', y='Val', color='Metric', markers=True,
                        color_discrete_sequence=['#3b82f6', '#94a3b8'], template='plotly_white')
    st.plotly_chart(fig_trend, use_container_width=True)

with col_g:
    st.subheader("⚖️ Gender Parity Index")
    gpi_val = f_gpi['Val'].mean() if not f_gpi.empty else 1.0
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = gpi_val,
        gauge = {'axis': {'range': [0.5, 1.5]}, 'bar': {'color': "#1e293b"},
                 'steps': [{'range': [0, 0.9], 'color': "#fca5a5"},
                           {'range': [0.9, 1.1], 'color': "#86efac"},
                           {'range': [1.1, 1.5], 'color': "#fca5a5"}]}))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption("Green Area = Parity (0.9 - 1.1)")

# --- ROW 4: BENCHMARKING ---
st.write("---")
st.subheader("📊 Regional Benchmarking")
bench_data = df_cc[(df_cc['Year'] == sel_year) & (df_cc['Metric'] == 'Completion Rate')]
if sel_level != "All": bench_data = bench_data[bench_data['Level of Education'] == sel_level]
bench_agg = bench_data.groupby('Geolocation')['Val'].mean().sort_values().reset_index()
bench_agg['Color'] = bench_agg['Geolocation'].apply(lambda x: '#3b82f6' if x == sel_region else '#cbd5e1')

fig_bench = px.bar(bench_agg, x='Val', y='Geolocation', orientation='h', text_auto='.1f')
fig_bench.update_traces(marker_color=bench_agg['Color'])
fig_bench.update_layout(template='plotly_white', xaxis_title="Completion Rate (%)", yaxis_title="")
st.plotly_chart(fig_bench, use_container_width=True)