import pandas as pd
import json

# Load your CSV data
cc_df = pd.read_csv('data_wrangling/Cleaned_Completion and Cohort Survival Rate.csv')

print("=== DATA STRUCTURE ===")
print(f"Shape: {cc_df.shape}")
print(f"\nColumns: {cc_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(cc_df.head())

print("\n=== GEOLOCATION VALUES ===")
geos = cc_df['Geolocation'].unique()
print(f"Unique Geolocations ({len(geos)}):")
for geo in sorted(geos):
    print(f"  - '{geo}'")

print("\n=== MAPPING ===")
region_map = {
    'NCR': 'NCR', 'CAR': 'CAR', 'BARMM': 'BARMM',
    'Region I': 'REGION I', 'Region II': 'REGION II', 'Region III': 'REGION III',
    'Region IV-A': 'REGION IV-A', 'Region IV-B': 'REGION IV-B', 'Region V': 'REGION V',
    'Region VI': 'REGION VI', 'Region VII': 'REGION VII', 'Region VIII': 'REGION VIII',
    'Region IX': 'REGION IX', 'Region X': 'REGION X', 'Region XI': 'REGION XI',
    'Region XII': 'REGION XII', 'Region XIII': 'REGION XIII'
}

cc_df['MapID'] = cc_df['Geolocation'].map(region_map)

print("Mapping results:")
for geo in sorted(geos):
    mapped = region_map.get(geo, 'NOT FOUND')
    print(f"  '{geo}' → '{mapped}'")

print("\n=== MELTING DATA ===")
years = [str(y) for y in range(2002, 2024)]
cc_l = cc_df.melt(id_vars=['Geolocation', 'Level of Education', 'Sex', 'Metric'], 
                   value_vars=years, var_name='Year', value_name='Val')
cc_l = cc_l[cc_l['Val'] > 0]
cc_l['MapID'] = cc_l['Geolocation'].map(region_map)
print(f"Melted shape: {cc_l.shape}")
print(f"Sample melted data:")
print(cc_l.head())

print("\n=== AFTER FILTERING (2023, Completion Rate) ===")
f_cc = cc_l[(cc_l['Year'] == '2023') & (cc_l['Metric'] == 'Completion Rate')]
print(f"Filtered rows: {len(f_cc)}")
print(f"\nUnique MapID values ({len(f_cc['MapID'].unique())}):")
for mapid in sorted(f_cc['MapID'].unique()):
    count = len(f_cc[f_cc['MapID'] == mapid])
    print(f"  - {mapid}: {count} rows")

print("\n=== AGGREGATED DATA ===")
map_agg = f_cc.groupby('MapID')['Val'].mean().reset_index()
map_agg.rename(columns={'MapID': 'REGION'}, inplace=True)
print(map_agg.to_string())

print("\n=== GEOJSON REGIONS ===")
with open('philippines_regions.geojson', 'r') as f:
    geojson = json.load(f)

geojson_regions = set()
for feature in geojson['features']:
    region = feature['properties'].get('REGION')
    geojson_regions.add(region)

print(f"Unique regions in GeoJSON ({len(geojson_regions)}):")
for region in sorted(geojson_regions):
    print(f"  - {region}")

print("\n=== MATCHING CHECK ===")
data_regions = set(map_agg['REGION'].unique())
matching = data_regions & geojson_regions
missing_in_geojson = data_regions - geojson_regions
missing_in_data = geojson_regions - data_regions

print(f"Matching regions: {len(matching)}")
if missing_in_geojson:
    print(f"❌ In data but NOT in GeoJSON: {missing_in_geojson}")
if missing_in_data:
    print(f"⚠️  In GeoJSON but NOT in data: {missing_in_data}")
if not missing_in_geojson and not missing_in_data:
    print("✓ Perfect match!")