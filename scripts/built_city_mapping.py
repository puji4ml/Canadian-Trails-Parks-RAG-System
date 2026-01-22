import csv
import io
import json
import os
import re
import urllib.request
import zipfile
from collections import defaultdict

GEONAMES_BASE = "http://download.geonames.org/export/dump"  # GeoNames dump directory [web:59]

# Choose one:
# - cities15000: small (major cities only)
# - cities5000: medium
# - cities1000: big (good coverage for Canada)
CITIES_ZIP = "cities1000.zip"  # listed in GeoNames dump directory [web:59]

ADMIN1_FILE = "admin1CodesASCII.txt"  # contains CA.xx provinces/territories names [web:73]

OUT_PATH = os.path.join("./data/", "city_to_province.json")


def normalize_name(s: str) -> str:
    s = s.strip().lower()
    # Normalize punctuation/whitespace (keeps things like "st. john's" searchable)
    s = re.sub(r"\s+", " ", s)
    return s


def download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def load_admin1_ca() -> dict:
    """
    Build mapping:
      admin1_code (e.g., "CA.08") -> province/territory name (e.g., "Ontario")
    from admin1CodesASCII.txt. [web:73]
    """
    url = f"{GEONAMES_BASE}/{ADMIN1_FILE}"
    raw = download_bytes(url).decode("utf-8", errors="replace")

    admin1_map = {}
    for line in raw.splitlines():
        # Format: code<TAB>name<TAB>nameAscii<TAB>geonameId [web:73]
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        code, name = parts[0], parts[1]
        if code.startswith("CA."):
            admin1_map[code] = name
    return admin1_map


def load_cities_rows() -> list:
    """
    Download and read cities*.zip -> cities*.txt
    cities files use the GeoNames "geoname" tab-delimited schema. [web:59]
    """
    url = f"{GEONAMES_BASE}/{CITIES_ZIP}"
    zbytes = download_bytes(url)

    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        # cities1000.zip contains cities1000.txt, etc.
        txt_name = CITIES_ZIP.replace(".zip", ".txt")
        with zf.open(txt_name) as f:
            raw = f.read().decode("utf-8", errors="replace")

    rows = []
    reader = csv.reader(io.StringIO(raw), delimiter="\t")
    for parts in reader:
        # GeoNames geoname columns (positions used here):
        #  1 name, 2 asciiname, 3 alternatenames, 8 feature class, 9 feature code,
        # 10 country code, 11 cc2, 12 admin1 code, ... [web:59]
        if len(parts) < 13:
            continue
        country_code = parts[8]  # note: index 8 is "feature class" in some docs; safer to use 8? -> use documented positions below
        # To avoid confusion with indexing, re-map by known column order:
        geonameid = parts[0]
        name = parts[1]
        asciiname = parts[2]
        alternatenames = parts[3]
        feature_class = parts[6]
        feature_code = parts[7]
        country = parts[8]
        admin1 = parts[10]  # admin1 code field (per cities*.txt schema) [web:59]

        # Filter to Canada only
        if country != "CA":
            continue

        rows.append({
            "name": name,
            "asciiname": asciiname,
            "alternatenames": alternatenames,
            "admin1": admin1,  # e.g., "08" for Ontario; combined with CA. [web:59]
            "feature_class": feature_class,
            "feature_code": feature_code,
        })
    return rows


def build_city_to_province() -> dict:
    admin1_map = load_admin1_ca()  # "CA.08" -> "Ontario" [web:73]
    cities = load_cities_rows()    # city records from cities1000.txt [web:59]

    city_to_prov = {}
    collisions = defaultdict(set)

    for r in cities:
        admin1_code = f"CA.{r['admin1']}"
        prov = admin1_map.get(admin1_code)
        if not prov:
            continue

        # Add primary names + a few alternates
        candidates = {r["name"], r["asciiname"]}
        for alt in (r["alternatenames"] or "").split(","):
            alt = alt.strip()
            # avoid huge/garbage alternates
            if 2 <= len(alt) <= 60:
                candidates.add(alt)

        for c in candidates:
            key = normalize_name(c)
            if not key:
                continue
            # Track duplicates: same city name can exist in multiple provinces
            collisions[key].add(prov)

    # Resolve collisions conservatively:
    # if a name maps to multiple provinces, we DO NOT include it (avoid wrong filtering).
    for name_key, provs in collisions.items():
        if len(provs) == 1:
            city_to_prov[name_key] = list(provs)[0]

    return city_to_prov


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    mapping = build_city_to_province()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(mapping):,} city name keys to {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
