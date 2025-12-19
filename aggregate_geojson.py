import json
from pathlib import Path
from collections import defaultdict

# Load the GeoJSON
with open('philippines_regions.geojson', 'r', encoding='utf-8') as f:
    original_geojson = json.load(f)

print(f"Original features: {len(original_geojson['features'])}")

# Group features by region
regions_dict = defaultdict(list)
for feature in original_geojson['features']:
    region = feature['properties'].get('REGION', 'UNKNOWN')
    regions_dict[region].append(feature)

print(f"Grouped into {len(regions_dict)} regions")

# Create new GeoJSON with merged geometries per region
new_features = []
for region_name, features in sorted(regions_dict.items()):
    # For now, just take the first feature's geometry (or we could merge them)
    # Merging geometries is complex, so we'll create a MultiPolygon
    
    geometries = []
    for feature in features:
        if feature['geometry']['type'] == 'Polygon':
            geometries.append(feature['geometry']['coordinates'])
        elif feature['geometry']['type'] == 'MultiPolygon':
            geometries.extend(feature['geometry']['coordinates'])
    
    # Create a MultiPolygon feature
    new_feature = {
        "type": "Feature",
        "properties": {"REGION": region_name},
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": geometries
        }
    }
    new_features.append(new_feature)
    print(f"✓ {region_name}: {len(features)} source features → 1 merged feature")

# Create aggregated GeoJSON
aggregated_geojson = {
    "type": "FeatureCollection",
    "features": new_features
}

# Save
output_path = Path.cwd() / "philippines_regions_aggregated.geojson"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(aggregated_geojson, f, ensure_ascii=False, indent=2)

print(f"\n✓ Saved aggregated GeoJSON with {len(new_features)} features")
print(f"✓ File: {output_path}")

# Verify
print(f"\nRegions in aggregated file:")
for feature in aggregated_geojson['features']:
    print(f"  - {feature['properties']['REGION']}")