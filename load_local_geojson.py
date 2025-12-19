import json
from pathlib import Path

# Path to your downloaded files
downloads_path = Path.home() / "Downloads" / "philippines-json-maps-master" / "2023" / "geojson" / "regions" / "medres"

# Region code to name mapping
region_mapping = {
    '100000000': 'REGION I',
    '200000000': 'REGION II', 
    '300000000': 'REGION III',
    '400000000': 'REGION IV-A',
    '500000000': 'REGION V',
    '600000000': 'REGION VI',
    '700000000': 'REGION VII',
    '800000000': 'REGION VIII',
    '900000000': 'REGION IX',
    '1000000000': 'REGION X',
    '1100000000': 'REGION XI',
    '1200000000': 'REGION XII',
    '1300000000': 'REGION XIII',
    '1400000000': 'NCR',
    '1600000000': 'CAR',
    '1700000000': 'BARMM',
    '1900000000': 'REGION IV-B'
}

unified = {"type": "FeatureCollection", "features": []}

print(f"Looking for files in: {downloads_path}\n")

if not downloads_path.exists():
    print(f"❌ Path not found: {downloads_path}")
    print("Make sure the folder structure is correct")
else:
    # Find all provdists JSON files
    json_files = sorted(downloads_path.glob("provdists-*.json"))
    print(f"Found {len(json_files)} regional files\n")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract region code from filename
            filename = json_file.stem  # e.g., "provdists-region-100000000.0.01"
            region_code = filename.split('-')[-1].split('.')[0]  # Extract "100000000"
            region_name = region_mapping.get(region_code, f"Unknown-{region_code}")
            
            # Add region name to all features
            if "features" in data:
                for feature in data["features"]:
                    if "properties" not in feature:
                        feature["properties"] = {}
                    feature["properties"]["REGION"] = region_name
                    unified["features"].append(feature)
                
                print(f"✓ {region_name}: {len(data['features'])} features")
        except Exception as e:
            print(f"✗ Error reading {json_file.name}: {e}")
    
    # Save unified file to current directory
    output_file = Path.cwd() / "philippines_regions.geojson"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unified, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Combined {len(unified['features'])} total features")
    print(f"✓ Saved to: {output_file}")
    
    # Print sample
    if unified['features']:
        print(f"\nSample feature:")
        print(json.dumps(unified['features'][0]['properties'], indent=2))