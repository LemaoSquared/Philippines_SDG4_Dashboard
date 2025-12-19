import json
import glob
import os

# 1. Setup - Get the folder where THIS script is actually saved
# This ensures it finds the JSON files even if run from a different folder
script_dir = os.path.dirname(os.path.abspath(__file__))
file_pattern = os.path.join(script_dir, "provdists-region-*.json")
files = glob.glob(file_pattern)

if not files:
    print(f"❌ Error: No region files found in: {script_dir}")
else:
    combined_geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Mapping codes to CSV Geolocation names
    mapping = {
        '1300000000': 'NCR', '1400000000': 'CAR', '100000000': 'Region I', 
        '200000000': 'Region II', '300000000': 'Region III', '400000000': 'Region IV-A', 
        '1700000000': 'Region IV-B', '500000000': 'Region V', '600000000': 'Region VI', 
        '700000000': 'Region VII', '800000000': 'Region VIII', '900000000': 'Region IX', 
        '1000000000': 'Region X', '1100000000': 'Region XI', '1200000000': 'Region XII', 
        '1600000000': 'Region XIII', '1900000000': 'BARMM'
    }

    print(f"🔄 Merging {len(files)} files found in {script_dir}...")

    for f_path in files:
        with open(f_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            filename = os.path.basename(f_path)
            
            # Extract PSGC from filename
            psgc = filename.split('-')[-1].split('.')[0]
            region_name = mapping.get(psgc, "Unknown")
            
            for feature in data['features']:
                feature['properties']['REGION'] = region_name
            
            combined_geojson['features'].extend(data['features'])

    # Save the file to the MAIN folder (one level up from regions-json)
    output_path = os.path.join(os.path.dirname(script_dir), 'philippines_regions.json')
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(combined_geojson, out_f)
    
    print(f"✅ Success! Created unified file at: {output_path}")