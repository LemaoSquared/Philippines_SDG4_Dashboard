import json
from pathlib import Path
from collections import Counter

# Load the GeoJSON file
geojson_file = Path.cwd() / "philippines_regions.geojson"

if not geojson_file.exists():
    print(f"❌ File not found: {geojson_file}")
else:
    with open(geojson_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data['features'])} features\n")
    
    # Get all property keys across all features
    all_keys = set()
    for feature in data['features']:
        if 'properties' in feature:
            all_keys.update(feature['properties'].keys())
    
    print("Available property keys:")
    for key in sorted(all_keys):
        print(f"  - {key}")
    
    # Get all unique REGION values
    regions = Counter()
    for feature in data['features']:
        if 'properties' in feature:
            region = feature['properties'].get('REGION', 'UNKNOWN')
            regions[region] += 1
    
    print(f"\nRegions found ({len(regions)} unique):")
    for region, count in sorted(regions.items()):
        print(f"  - {region}: {count} features")
    
    # Show first feature
    print(f"\nFirst feature properties:")
    print(json.dumps(data['features'][0].get('properties', {}), indent=2))