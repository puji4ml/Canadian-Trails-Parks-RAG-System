import requests
import json
import os
import time
from tqdm import tqdm
from typing import List, Dict
import pandas as pd

class CanadaTrailsCollector:
    def __init__(self):
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        self.output_dir = "./RAG/data/raw"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Split large provinces into smaller sub-regions
        self.regions = {
            "BC_South": {"bounds": [48.0, -125.0, 52.0, -114.0], "name": "British Columbia South"},
            "BC_North": {"bounds": [52.0, -135.0, 60.0, -120.0], "name": "British Columbia North"},
            
            "AB_South": {"bounds": [49.0, -120.0, 53.0, -110.0], "name": "Alberta South"},
            "AB_North": {"bounds": [53.0, -120.0, 60.0, -110.0], "name": "Alberta North"},
            
            "ON_South": {"bounds": [42.0, -83.0, 46.0, -74.0], "name": "Ontario South"},
            "ON_Central": {"bounds": [46.0, -90.0, 50.0, -77.0], "name": "Ontario Central"},
            "ON_North": {"bounds": [50.0, -95.0, 57.0, -80.0], "name": "Ontario North"},
            
            "QC_South": {"bounds": [45.0, -79.0, 48.0, -65.0], "name": "Quebec South"},
            "QC_North": {"bounds": [48.0, -79.0, 55.0, -57.0], "name": "Quebec North"},
            
            "SK": {"bounds": [49.0, -110.0, 55.0, -101.0], "name": "Saskatchewan"},
            "MB": {"bounds": [49.0, -102.0, 55.0, -95.0], "name": "Manitoba"},
            
            "NS": {"bounds": [43.4, -66.5, 47.0, -59.7], "name": "Nova Scotia"},
            "NB": {"bounds": [44.5, -69.0, 48.0, -64.0], "name": "New Brunswick"},
            "PE": {"bounds": [45.9, -64.5, 47.1, -61.9], "name": "Prince Edward Island"},
            "NL": {"bounds": [46.6, -59.5, 52.0, -52.6], "name": "Newfoundland"},
        }
    
    def build_overpass_query(self, bounds: List[float]) -> str:
        """Build smaller, more focused query"""
        bbox = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"
        
        # Simplified query with shorter timeout
        query = f"""
        [out:json][timeout:60][bbox:{bbox}];
        (
          way["highway"="path"]["name"];
          way["highway"="footway"]["name"];
          way["route"="hiking"]["name"];
          relation["route"="hiking"]["name"];
        );
        out body;
        """
        
        return query
    
    def fetch_trails_for_region(self, region_code: str, region_data: Dict, retry_count=2) -> List[Dict]:
        """Fetch trail data with retry logic"""
        bounds = region_data["bounds"]
        name = region_data["name"]
        
        print(f"\n🗺️  Fetching trails for {name}...")
        
        query = self.build_overpass_query(bounds)
        
        for attempt in range(retry_count):
            try:
                print(f"  Attempt {attempt + 1}/{retry_count}...", end=" ")
                
                response = requests.post(
                    self.overpass_url,
                    data={'data': query},
                    timeout=90
                )
                
                if response.status_code == 429:
                    print("⏳ Rate limited, waiting 60s...")
                    time.sleep(60)
                    continue
                
                response.raise_for_status()
                data = response.json()
                elements = data.get('elements', [])
                
                print(f"✅ Retrieved {len(elements)} elements")
                return elements
                
            except requests.exceptions.Timeout:
                print(f"⚠️  Timeout")
                if attempt < retry_count - 1:
                    wait_time = 30 * (attempt + 1)
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
            except requests.exceptions.RequestException as e:
                print(f"❌ Error: {e}")
                if attempt < retry_count - 1:
                    time.sleep(30)
        
        print(f"  ⚠️  Failed after {retry_count} attempts")
        return []
    
    def process_trail_element(self, element: Dict, region: str) -> Dict:
        """Process trail element into structured format"""
        tags = element.get('tags', {})
        
        trail_data = {
            "id": f"trail_{region}_{element.get('id')}",
            "name": tags.get('name', 'Unnamed Trail'),
            "region": region,
            "osm_id": element.get('id'),
            
            # Trail info
            "trail_type": tags.get('highway', tags.get('route', 'unknown')),
            "surface": tags.get('surface', 'unknown'),
            "difficulty": tags.get('sac_scale', tags.get('difficulty', 'unknown')),
            "trail_visibility": tags.get('trail_visibility', 'unknown'),
            "access": tags.get('access', 'public'),
            
            # Permissions
            "bicycle": tags.get('bicycle', 'unknown'),
            "horse": tags.get('horse', 'unknown'),
            "wheelchair": tags.get('wheelchair', 'unknown'),
            
            # Management
            "operator": tags.get('operator', ''),
            "network": tags.get('network', ''),
            "description": tags.get('description', ''),
            "website": tags.get('website', ''),
            
            # All tags
            "all_tags": tags
        }
        
        # Generate content
        trail_data["content"] = self.generate_content(trail_data)
        trail_data["word_count"] = len(trail_data["content"].split())
        
        return trail_data
    
    def generate_content(self, trail: Dict) -> str:
        """Generate RAG-friendly content"""
        parts = []
        
        # Title
        parts.append(f"# {trail['name']}")
        parts.append(f"**Region**: {trail['region']}, Canada\n")
        
        # Description
        desc_parts = [f"{trail['name']} is a trail in {trail['region']}, Canada."]
        
        if trail['trail_type'] != 'unknown':
            desc_parts.append(f"It is a {trail['trail_type']} trail.")
        
        if trail['difficulty'] != 'unknown':
            desc_parts.append(f"Difficulty: {trail['difficulty']}.")
        
        if trail['surface'] != 'unknown':
            desc_parts.append(f"Surface: {trail['surface']}.")
        
        if trail['trail_visibility'] != 'unknown':
            desc_parts.append(f"Trail visibility: {trail['trail_visibility']}.")
        
        if trail['bicycle'] == 'yes':
            desc_parts.append("Bicycles are permitted.")
        
        if trail['horse'] == 'yes':
            desc_parts.append("Horseback riding is permitted.")
        
        if trail['wheelchair'] == 'yes':
            desc_parts.append("Wheelchair accessible.")
        
        if trail['operator']:
            desc_parts.append(f"Managed by {trail['operator']}.")
        
        if trail['network']:
            desc_parts.append(f"Part of {trail['network']} network.")
        
        if trail['description']:
            desc_parts.append(trail['description'])
        
        if trail['website']:
            desc_parts.append(f"More info: {trail['website']}")
        
        parts.append(' '.join(desc_parts))
        
        return '\n\n'.join(parts)
    
    def collect_all_regions(self):
        """Collect with progress tracking"""
        all_trails = []
        failed_regions = []
        
        print(f"🍁 Canadian Trails Collection")
        print(f"📍 Collecting from {len(self.regions)} sub-regions")
        print("="*60)
        
        for region_code, region_data in self.regions.items():
            print(f"\n{'='*60}")
            print(f"Region: {region_data['name']} ({region_code})")
            print(f"{'='*60}")
            
            # Fetch data
            elements = self.fetch_trails_for_region(region_code, region_data)
            
            if not elements:
                failed_regions.append(region_code)
                print(f"⚠️  No data retrieved for {region_code}")
                time.sleep(10)  # Wait longer before next region
                continue
            
            # Process trails
            region_trails = []
            for element in tqdm(elements, desc=f"Processing {region_code}"):
                if element.get('tags', {}).get('name'):
                    trail = self.process_trail_element(element, region_data['name'])
                    region_trails.append(trail)
            
            all_trails.extend(region_trails)
            
            print(f"✅ Collected {len(region_trails)} trails from {region_code}")
            
            # Save regional file
            if region_trails:
                region_file = os.path.join(self.output_dir, f"trails_{region_code.lower()}.json")
                with open(region_file, 'w', encoding='utf-8') as f:
                    json.dump(region_trails, f, indent=2, ensure_ascii=False)
            
            # Important: Wait between regions to avoid rate limiting
            print("⏳ Waiting 15 seconds before next region...")
            time.sleep(15)
        
        # Save combined data
        if all_trails:
            combined_file = os.path.join(self.output_dir, "canada_trails_all.json")
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(all_trails, f, indent=2, ensure_ascii=False)
            
            self.print_statistics(all_trails, failed_regions)
        else:
            print("\n❌ No trails collected!")
        
        return all_trails
    
    def print_statistics(self, trails: List[Dict], failed: List[str]):
        """Print collection statistics"""
        print(f"\n{'='*60}")
        print("📊 COLLECTION SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n✅ Successfully collected:")
        print(f"  Total trails: {len(trails)}")
        print(f"  Total words: {sum(t['word_count'] for t in trails):,}")
        print(f"  Avg words/trail: {sum(t['word_count'] for t in trails) / len(trails):.0f}")
        
        # Regional breakdown
        df = pd.DataFrame(trails)
        print(f"\n📍 By Region:")
        region_counts = df['region'].value_counts()
        for region, count in region_counts.items():
            print(f"  {region}: {count} trails")
        
        # Trail types
        print(f"\n🚶 Trail Types:")
        type_counts = df['trail_type'].value_counts()
        for ttype, count in type_counts.head(5).items():
            print(f"  {ttype}: {count}")
        
        if failed:
            print(f"\n⚠️  Failed regions: {', '.join(failed)}")
        
        print(f"\n📄 Data saved to: {self.output_dir}/")
        print(f"{'='*60}")

if __name__ == "__main__":
    collector = CanadaTrailsCollector()
    trails = collector.collect_all_regions()
    
    if trails:
        print(f"\n🎉 Success! Collected {len(trails)} trails")
    else:
        print("\n⚠️  Collection failed. See alternative solutions below.")