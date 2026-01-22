import json
import os
from typing import List, Dict

def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON file safely"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"⚠️  Invalid JSON: {filepath}")
        return []

def combine_datasets():
    """Combine trails and parks into unified knowledge base"""
    
    print("🔄 Combining Canadian Wilderness Data")
    print("="*60)
    
    data_dir = "./RAG/data/raw"
    output_dir = "./RAG/data/processed"

    os.makedirs(output_dir, exist_ok=True)
    
    all_documents = []
    stats = {
        "trails": 0,
        "parks": 0,
        "total_words": 0
    }
    
    # Load trails
    trails_file = os.path.join(data_dir, "canada_trails_all.json")
    if os.path.exists(trails_file):
        print("📍 Loading trails data...")
        trails = load_json_file(trails_file)
        
        # Standardize trail format for RAG
        for trail in trails:
            doc = {
                "id": trail.get("id"),
                "title": trail.get("name"),
                "type": "Trail",
                "region": trail.get("region"),
                "province": trail.get("province", ""),
                "content": trail.get("content", ""),
                "metadata": {
                    "source": "OpenStreetMap",
                    "trail_type": trail.get("trail_type"),
                    "difficulty": trail.get("difficulty"),
                    "surface": trail.get("surface"),
                    "all_tags": trail.get("all_tags", {})
                }
            }
            all_documents.append(doc)
        
        stats["trails"] = len(trails)
        print(f"  ✅ Loaded {len(trails):,} trails")
    
    # Load parks
    parks_file = os.path.join(data_dir, "parks_canada_complete.json")
    if os.path.exists(parks_file):
        print("🏞️  Loading parks data...")
        parks = load_json_file(parks_file)
        
        # Standardize park format for RAG
        for park in parks:
            doc = {
                "id": f"park_{park.get('name', '').replace(' ', '_')}",
                "title": park.get("name"),
                "type": park.get("type", "Park"),
                "region": park.get("province"),
                "province": park.get("province"),
                "content": park.get("content", ""),
                "metadata": {
                    "source": "Parks Canada",
                    "activities": park.get("activities", []),
                    "features": park.get("features", []),
                    "facilities": park.get("facilities", []),
                    "area": park.get("area"),
                    "established": park.get("established"),
                    "unesco_site": park.get("unesco_site", False),
                    "url": park.get("url")
                }
            }
            all_documents.append(doc)
        
        stats["parks"] = len(parks)
        print(f"  ✅ Loaded {len(parks)} parks")
    
    # Filter out documents with insufficient content
    print("\n🔍 Filtering documents...")
    filtered_docs = []
    for doc in all_documents:
        word_count = len(doc['content'].split())
        if word_count >= 20:  # Minimum 20 words
            doc['word_count'] = word_count
            filtered_docs.append(doc)
    
    stats["total_words"] = sum(d['word_count'] for d in filtered_docs)
    
    print(f"  ✅ Kept {len(filtered_docs):,} documents (filtered {len(all_documents) - len(filtered_docs):,})")
    
    # Create train/test split
    print("\n✂️  Creating train/test split...")
    import random
    random.shuffle(filtered_docs)
    
    split_idx = int(len(filtered_docs) * 0.85)
    train_docs = filtered_docs[:split_idx]
    test_docs = filtered_docs[split_idx:]
    
    # Save combined dataset
    train_file = os.path.join(output_dir, "canada_wilderness_train.json")
    test_file = os.path.join(output_dir, "canada_wilderness_test.json")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_docs, f, indent=2, ensure_ascii=False)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_docs, f, indent=2, ensure_ascii=False)
    
    # Save statistics
    stats_data = {
        "total_documents": len(filtered_docs),
        "train_documents": len(train_docs),
        "test_documents": len(test_docs),
        "trails_count": stats["trails"],
        "parks_count": stats["parks"],
        "total_words": stats["total_words"],
        "avg_words_per_document": stats["total_words"] / len(filtered_docs) if filtered_docs else 0
        }
    
    stats_file = os.path.join(output_dir, "canada_wilderness_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("📊 FINAL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total Documents: {len(filtered_docs):,}")
    print(f"  - Trails: {stats['trails']:,}")
    print(f"  - Parks: {stats['parks']}")
    print(f"\nTrain Set: {len(train_docs):,} documents")
    print(f"Test Set: {len(test_docs):,} documents")
    print(f"\nTotal Words: {stats['total_words']:,}")
    print(f"Avg Words/Doc: {stats_data['avg_words_per_document']:.0f}")
    print(f"\n📄 Files saved:")
    print(f"  - {train_file}")
    print(f"  - {test_file}")
    print(f"  - {stats_file}")
    print(f"{'='*60}")

    return train_docs, test_docs, stats_data

if __name__ == "__main__":
    train, test, stats = combine_datasets()