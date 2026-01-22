import requests
from bs4 import BeautifulSoup
import json
import os
from tqdm import tqdm
import time
from typing import List, Dict
import re

class ParksCanadaCollector:
    def __init__(self):
        self.base_url = "https://www.pc.gc.ca"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.output_dir = "./RAG/data/raw"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_all_parks_list(self) -> List[Dict]:
        """
        Comprehensive list of Parks Canada locations
        Organized by type and province
        """
        parks = {
            # National Parks - British Columbia
            "BC": [
                {"name": "Glacier National Park", "url": "/en/pn-np/bc/glacier", "type": "National Park", "province": "BC"},
                {"name": "Gulf Islands National Park Reserve", "url": "/en/pn-np/bc/gulf", "type": "National Park Reserve", "province": "BC"},
                {"name": "Gwaii Haanas National Park Reserve", "url": "/en/pn-np/bc/gwaiihaanas", "type": "National Park Reserve", "province": "BC"},
                {"name": "Kootenay National Park", "url": "/en/pn-np/bc/kootenay", "type": "National Park", "province": "BC"},
                {"name": "Mount Revelstoke National Park", "url": "/en/pn-np/bc/revelstoke", "type": "National Park", "province": "BC"},
                {"name": "Pacific Rim National Park Reserve", "url": "/en/pn-np/bc/pacificrim", "type": "National Park Reserve", "province": "BC"},
                {"name": "Yoho National Park", "url": "/en/pn-np/bc/yoho", "type": "National Park", "province": "BC"},
            ],
            
            # National Parks - Alberta
            "AB": [
                {"name": "Banff National Park", "url": "/en/pn-np/ab/banff", "type": "National Park", "province": "AB"},
                {"name": "Elk Island National Park", "url": "/en/pn-np/ab/elkisland", "type": "National Park", "province": "AB"},
                {"name": "Jasper National Park", "url": "/en/pn-np/ab/jasper", "type": "National Park", "province": "AB"},
                {"name": "Waterton Lakes National Park", "url": "/en/pn-np/ab/waterton", "type": "National Park", "province": "AB"},
                {"name": "Wood Buffalo National Park", "url": "/en/pn-np/nt/woodbuffalo", "type": "National Park", "province": "AB/NT"},
            ],
            
            # National Parks - Saskatchewan
            "SK": [
                {"name": "Grasslands National Park", "url": "/en/pn-np/sk/grasslands", "type": "National Park", "province": "SK"},
                {"name": "Prince Albert National Park", "url": "/en/pn-np/sk/princealbert", "type": "National Park", "province": "SK"},
            ],
            
            # National Parks - Manitoba
            "MB": [
                {"name": "Riding Mountain National Park", "url": "/en/pn-np/mb/riding", "type": "National Park", "province": "MB"},
                {"name": "Wapusk National Park", "url": "/en/pn-np/mb/wapusk", "type": "National Park", "province": "MB"},
            ],
            
            # National Parks - Ontario
            "ON": [
                {"name": "Bruce Peninsula National Park", "url": "/en/pn-np/on/bruce", "type": "National Park", "province": "ON"},
                {"name": "Georgian Bay Islands National Park", "url": "/en/pn-np/on/geobai", "type": "National Park", "province": "ON"},
                {"name": "Point Pelee National Park", "url": "/en/pn-np/on/pelee", "type": "National Park", "province": "ON"},
                {"name": "Pukaskwa National Park", "url": "/en/pn-np/on/pukaskwa", "type": "National Park", "province": "ON"},
                {"name": "Thousand Islands National Park", "url": "/en/pn-np/on/thousand", "type": "National Park", "province": "ON"},
            ],
            
            # National Parks - Quebec
            "QC": [
                {"name": "Forillon National Park", "url": "/en/pn-np/qc/forillon", "type": "National Park", "province": "QC"},
                {"name": "La Mauricie National Park", "url": "/en/pn-np/qc/mauricie", "type": "National Park", "province": "QC"},
                {"name": "Mingan Archipelago National Park Reserve", "url": "/en/pn-np/qc/mingan", "type": "National Park Reserve", "province": "QC"},
            ],
            
            # National Parks - New Brunswick
            "NB": [
                {"name": "Fundy National Park", "url": "/en/pn-np/nb/fundy", "type": "National Park", "province": "NB"},
                {"name": "Kouchibouguac National Park", "url": "/en/pn-np/nb/kouchibouguac", "type": "National Park", "province": "NB"},
            ],
            
            # National Parks - Nova Scotia
            "NS": [
                {"name": "Cape Breton Highlands National Park", "url": "/en/pn-np/ns/cbreton", "type": "National Park", "province": "NS"},
                {"name": "Kejimkujik National Park", "url": "/en/pn-np/ns/kejimkujik", "type": "National Park", "province": "NS"},
            ],
            
            # National Parks - Prince Edward Island
            "PE": [
                {"name": "Prince Edward Island National Park", "url": "/en/pn-np/pe/pei-ipe", "type": "National Park", "province": "PE"},
            ],
            
            # National Parks - Newfoundland and Labrador
            "NL": [
                {"name": "Gros Morne National Park", "url": "/en/pn-np/nl/grosmorne", "type": "National Park", "province": "NL"},
                {"name": "Terra Nova National Park", "url": "/en/pn-np/nl/terranova", "type": "National Park", "province": "NL"},
                {"name": "Torngat Mountains National Park", "url": "/en/pn-np/nl/torngats", "type": "National Park", "province": "NL"},
            ],
            
            # National Parks - Yukon
            "YT": [
                {"name": "Ivvavik National Park", "url": "/en/pn-np/yt/ivvavik", "type": "National Park", "province": "YT"},
                {"name": "Kluane National Park and Reserve", "url": "/en/pn-np/yt/kluane", "type": "National Park and Reserve", "province": "YT"},
                {"name": "Vuntut National Park", "url": "/en/pn-np/yt/vuntut", "type": "National Park", "province": "YT"},
            ],
            
            # National Parks - Northwest Territories
            "NT": [
                {"name": "Aulavik National Park", "url": "/en/pn-np/nt/aulavik", "type": "National Park", "province": "NT"},
                {"name": "Nahanni National Park Reserve", "url": "/en/pn-np/nt/nahanni", "type": "National Park Reserve", "province": "NT"},
                {"name": "Tuktut Nogait National Park", "url": "/en/pn-np/nt/tuktutnogait", "type": "National Park", "province": "NT"},
            ],
            
            # National Parks - Nunavut
            "NU": [
                {"name": "Auyuittuq National Park", "url": "/en/pn-np/nu/auyuittuq", "type": "National Park", "province": "NU"},
                {"name": "Qausuittuq National Park", "url": "/en/pn-np/nu/qausuittuq", "type": "National Park", "province": "NU"},
                {"name": "Quttinirpaaq National Park", "url": "/en/pn-np/nu/quttinirpaaq", "type": "National Park", "province": "NU"},
                {"name": "Sirmilik National Park", "url": "/en/pn-np/nu/sirmilik", "type": "National Park", "province": "NU"},
                {"name": "Ukkusiksalik National Park", "url": "/en/pn-np/nu/ukkusiksalik", "type": "National Park", "province": "NU"},
            ],
            
            # National Historic Sites - Major ones
            "Historic": [
                {"name": "Fort Anne National Historic Site", "url": "/en/lhn-nhs/ns/fortanne", "type": "National Historic Site", "province": "NS"},
                {"name": "Fortress of Louisbourg National Historic Site", "url": "/en/lhn-nhs/ns/louisbourg", "type": "National Historic Site", "province": "NS"},
                {"name": "L'Anse aux Meadows National Historic Site", "url": "/en/lhn-nhs/nl/meadows", "type": "National Historic Site", "province": "NL"},
                {"name": "Signal Hill National Historic Site", "url": "/en/lhn-nhs/nl/signalhill", "type": "National Historic Site", "province": "NL"},
                {"name": "The Forks National Historic Site", "url": "/en/lhn-nhs/mb/forks", "type": "National Historic Site", "province": "MB"},
            ],
            
            # National Marine Conservation Areas
            "Marine": [
                {"name": "Fathom Five National Marine Park", "url": "/en/amnc-nmca/on/fathomfive", "type": "National Marine Park", "province": "ON"},
                {"name": "Saguenay-St. Lawrence Marine Park", "url": "/en/amnc-nmca/qc/saguenay", "type": "National Marine Park", "province": "QC"},
                {"name": "Lake Superior National Marine Conservation Area", "url": "/en/amnc-nmca/on/superior", "type": "National Marine Conservation Area", "province": "ON"},
            ],
        }
        
        # Flatten the dictionary into a list
        all_parks = []
        for province, park_list in parks.items():
            all_parks.extend(park_list)
        
        return all_parks
    
    def fetch_park_page(self, url: str, retry_count: int = 3) -> str:
        """Fetch park page with retry logic"""
        full_url = f"{self.base_url}{url}"
        
        for attempt in range(retry_count):
            try:
                response = self.session.get(full_url, timeout=30)
                response.raise_for_status()
                return response.text
            except requests.exceptions.Timeout:
                if attempt < retry_count - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                else:
                    raise
            except requests.exceptions.RequestException as e:
                if attempt < retry_count - 1:
                    time.sleep(5)
                    continue
                else:
                    raise
    
    def extract_park_info(self, html: str, park_data: Dict) -> Dict:
        """Extract comprehensive information from park page"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Initialize park info
        park_info = {
            "name": park_data["name"],
            "type": park_data["type"],
            "province": park_data["province"],
            "url": f"{self.base_url}{park_data['url']}",
            "source": "Parks Canada"
        }
        
        # Extract main description
        description_parts = []
        
        # Method 1: Look for intro/summary sections
        intro = soup.find('div', class_='intro') or soup.find('div', class_='summary')
        if intro:
            description_parts.append(intro.get_text(strip=True))
        
        # Method 2: Get paragraphs from main content
        main_content = soup.find('main') or soup.find('article')
        if main_content:
            paragraphs = main_content.find_all('p', limit=10)
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:  # Filter out short navigation text
                    description_parts.append(text)
        
        park_info["description"] = '\n\n'.join(description_parts[:5])  # Top 5 paragraphs
        
        # Extract specific sections
        park_info["activities"] = self.extract_activities(soup)
        park_info["features"] = self.extract_features(soup)
        park_info["facilities"] = self.extract_facilities(soup)
        park_info["visitor_info"] = self.extract_visitor_info(soup)
        
        # Extract metadata
        park_info["area"] = self.extract_area(soup)
        park_info["established"] = self.extract_established(soup)
        park_info["unesco_site"] = self.check_unesco(soup)
        
        # Generate comprehensive content for RAG
        park_info["content"] = self.generate_content(park_info)
        park_info["word_count"] = len(park_info["content"].split())
        
        return park_info
    
    def extract_activities(self, soup: BeautifulSoup) -> List[str]:
        """Extract available activities"""
        activities = []
        
        # Look for activities section
        activities_section = soup.find('section', id=re.compile('.*activities.*', re.I))
        if not activities_section:
            activities_section = soup.find('div', class_=re.compile('.*activities.*', re.I))
        
        if activities_section:
            # Find list items or links
            items = activities_section.find_all(['li', 'a'])
            for item in items:
                text = item.get_text(strip=True)
                if text and len(text) < 50:  # Activity names are usually short
                    activities.append(text)
        
        # Common keywords to look for
        keywords = ['hiking', 'camping', 'fishing', 'swimming', 'canoeing', 'kayaking', 
                   'wildlife viewing', 'photography', 'cycling', 'backcountry', 'skiing',
                   'snowshoeing', 'climbing', 'boating']
        
        text_content = soup.get_text().lower()
        for keyword in keywords:
            if keyword in text_content and keyword not in [a.lower() for a in activities]:
                activities.append(keyword.title())
        
        return list(set(activities))[:15]  # Limit to 15 unique activities
    
    def extract_features(self, soup: BeautifulSoup) -> List[str]:
        """Extract natural and cultural features"""
        features = []
        
        # Look for features in text
        keywords = ['mountain', 'lake', 'river', 'waterfall', 'glacier', 'forest', 'beach',
                   'canyon', 'cliff', 'wildlife', 'heritage', 'historic', 'cultural']
        
        text_content = soup.get_text().lower()
        for keyword in keywords:
            if keyword in text_content:
                features.append(keyword.title())
        
        return list(set(features))
    
    def extract_facilities(self, soup: BeautifulSoup) -> List[str]:
        """Extract available facilities"""
        facilities = []
        
        keywords = ['visitor centre', 'campground', 'parking', 'picnic area', 'washroom',
                   'trail', 'interpretive', 'lookout', 'viewpoint', 'shelter']
        
        text_content = soup.get_text().lower()
        for keyword in keywords:
            if keyword in text_content:
                facilities.append(keyword.title())
        
        return list(set(facilities))
    
    def extract_visitor_info(self, soup: BeautifulSoup) -> Dict:
        """Extract visitor information"""
        info = {
            "fees_required": "fees" in soup.get_text().lower() or "admission" in soup.get_text().lower(),
            "camping_available": "camping" in soup.get_text().lower() or "campground" in soup.get_text().lower(),
            "reservation_required": "reservation" in soup.get_text().lower() or "booking" in soup.get_text().lower()
        }
        return info
    
    def extract_area(self, soup: BeautifulSoup) -> str:
        """Extract park area/size"""
        text = soup.get_text()
        
        # Look for area patterns
        area_patterns = [
            r'(\d+[,\d]*)\s*km²',
            r'(\d+[,\d]*)\s*square kilometres',
            r'(\d+[,\d]*)\s*hectares'
        ]
        
        for pattern in area_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(0)
        
        return "Not specified"
    
    def extract_established(self, soup: BeautifulSoup) -> str:
        """Extract establishment year"""
        text = soup.get_text()
        
        # Look for establishment patterns
        patterns = [
            r'established in (\d{4})',
            r'created in (\d{4})',
            r'founded in (\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(1)
        
        return "Not specified"
    
    def check_unesco(self, soup: BeautifulSoup) -> bool:
        """Check if it's a UNESCO World Heritage Site"""
        text = soup.get_text().lower()
        return 'unesco' in text or 'world heritage' in text
    
    def generate_content(self, park_info: Dict) -> str:
        """Generate comprehensive RAG-friendly content"""
        parts = []
        
        # Title and header
        parts.append(f"# {park_info['name']}")
        parts.append(f"**Type**: {park_info['type']}")
        parts.append(f"**Province/Territory**: {park_info['province']}")
        parts.append("")
        
        # Description
        if park_info['description']:
            parts.append("## Description")
            parts.append(park_info['description'])
            parts.append("")
        
        # Key Information
        parts.append("## Key Information")
        if park_info['area'] != "Not specified":
            parts.append(f"- **Area**: {park_info['area']}")
        if park_info['established'] != "Not specified":
            parts.append(f"- **Established**: {park_info['established']}")
        if park_info['unesco_site']:
            parts.append(f"- **UNESCO World Heritage Site**: Yes")
        parts.append("")
        
        # Activities
        if park_info['activities']:
            parts.append("## Activities")
            parts.append("Visitors can enjoy the following activities:")
            for activity in park_info['activities']:
                parts.append(f"- {activity}")
            parts.append("")
        
        # Features
        if park_info['features']:
            parts.append("## Natural and Cultural Features")
            parts.append(f"The park features: {', '.join(park_info['features'])}")
            parts.append("")
        
        # Facilities
        if park_info['facilities']:
            parts.append("## Facilities")
            parts.append(f"Available facilities include: {', '.join(park_info['facilities'])}")
            parts.append("")
        
        # Visitor Information
        parts.append("## Visitor Information")
        if park_info['visitor_info']['fees_required']:
            parts.append("- Entry fees apply")
        if park_info['visitor_info']['camping_available']:
            parts.append("- Camping facilities available")
        if park_info['visitor_info']['reservation_required']:
            parts.append("- Reservations required for some activities")
        parts.append("")
        
        # Footer
        parts.append("---")
        parts.append(f"*Source: Parks Canada*")
        parts.append(f"*More information: {park_info['url']}*")
        
        return '\n'.join(parts)
    
    def collect_all_parks(self):
        """Main collection method with progress tracking"""
        parks = self.get_all_parks_list()
        collected_parks = []
        failed_parks = []
        
        print(f"🏞️  Parks Canada Collection")
        print(f"📍 Total locations to collect: {len(parks)}")
        print("="*60)
        
        # Organize by province for better progress tracking
        by_province = {}
        for park in parks:
            prov = park['province']
            if prov not in by_province:
                by_province[prov] = []
            by_province[prov].append(park)
        
        for province, park_list in by_province.items():
            print(f"\n{'='*60}")
            print(f"Province/Territory: {province} ({len(park_list)} locations)")
            print(f"{'='*60}")
            
            for park in tqdm(park_list, desc=f"Collecting {province}"):
                try:
                    # Fetch page
                    html = self.fetch_park_page(park['url'])
                    
                    # Extract information
                    park_info = self.extract_park_info(html, park)
                    
                    collected_parks.append(park_info)
                    
                    # Rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"\n  ❌ Failed: {park['name']} - {e}")
                    failed_parks.append(park['name'])
                    time.sleep(5)
                    continue
        
        # Save results
        self.save_results(collected_parks, failed_parks)
        
        return collected_parks
    
    def save_results(self, parks: List[Dict], failed: List[str]):
        """Save collected data and generate statistics"""
        
        # Save main data
        output_file = os.path.join(self.output_dir, "parks_canada_complete.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parks, f, indent=2, ensure_ascii=False)
        
        # Save by province
        by_province = {}
        for park in parks:
            prov = park['province']
            if prov not in by_province:
                by_province[prov] = []
            by_province[prov].append(park)
        
        for province, park_list in by_province.items():
            prov_file = os.path.join(self.output_dir, f"parks_{province.lower().replace('/', '_')}.json")
            with open(prov_file, 'w', encoding='utf-8') as f:
                json.dump(park_list, f, indent=2, ensure_ascii=False)
        
        # Generate statistics
        self.print_statistics(parks, failed)
    
    def print_statistics(self, parks: List[Dict], failed: List[str]):
        """Print collection statistics"""
        print(f"\n{'='*60}")
        print("📊 PARKS CANADA COLLECTION SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n✅ Successfully collected:")
        print(f"  Total parks: {len(parks)}")
        print(f"  Total words: {sum(p['word_count'] for p in parks):,}")
        print(f"  Avg words/park: {sum(p['word_count'] for p in parks) / len(parks):.0f}")
        
        # By type
        print(f"\n🏞️  By Type:")
        types = {}
        for park in parks:
            ptype = park['type']
            types[ptype] = types.get(ptype, 0) + 1
        
        for ptype, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ptype}: {count}")
        
        # By province
        print(f"\n📍 By Province/Territory:")
        provinces = {}
        for park in parks:
            prov = park['province']
            provinces[prov] = provinces.get(prov, 0) + 1
        
        for prov, count in sorted(provinces.items(), key=lambda x: x[1], reverse=True):
            print(f"  {prov}: {count}")
        
        # UNESCO sites
        unesco_count = sum(1 for p in parks if p['unesco_site'])
        print(f"\n🌍 UNESCO World Heritage Sites: {unesco_count}")
        
        # Failed
        if failed:
            print(f"\n⚠️  Failed to collect ({len(failed)}):")
            for name in failed[:10]:  # Show first 10
                print(f"  - {name}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")
        
        print(f"\n📄 Data saved to: {self.output_dir}/")
        print(f"{'='*60}")

if __name__ == "__main__":
    collector = ParksCanadaCollector()
    parks = collector.collect_all_parks()
    
    if parks:
        print(f"\n🎉 Success! Collected {len(parks)} Parks Canada locations")
    else:
        print("\n❌ Collection failed")