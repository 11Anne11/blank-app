import json
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests

from .config import BAG_BASE_URL, RAW_DIR


LOCATION_SEARCH_URL = "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free"
REQUEST_ATTEMPTS = 3


def _request_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fetch a JSON object with bounded retries for transient PDOK failures."""
    last_error: Exception | None = None
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=30,
                headers={"User-Agent": "NederlandseWoningWaardering/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("PDOK gaf geen JSON-object terug.")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < REQUEST_ATTEMPTS:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"PDOK-verzoek mislukt na {REQUEST_ATTEMPTS} pogingen: {last_error}") from last_error


def iterate_collection_items(
    collection_id: str,
    limit: int = 1000,
    bbox: Optional[List[float]] = None,
    max_pages: Optional[int] = None,
    skip_pages: int = 0,
) -> Iterator[Dict[str, Any]]:
    url = f"{BAG_BASE_URL}/collections/{collection_id}/items"
    params = {"f": "json", "limit": limit}
    if bbox is not None:
        params["bbox"] = ",".join(str(value) for value in bbox)
    page_count = 0
    skipped = 0
    while True:
        payload = _request_json(url, params=params)
        if skipped >= skip_pages:
            for feature in payload.get("features", []):
                yield feature
        else:
            skipped += 1
        page_count += 1
        if max_pages is not None and page_count >= max_pages:
            break
        next_href = next((link.get("href") for link in payload.get("links", []) if link.get("rel") == "next"), None)
        if not next_href:
            break
        url = next_href
        params = None


def normalize_postcode(postcode: str) -> str:
    cleaned = postcode.replace(" ", "").upper()
    if len(cleaned) < 6:
        raise ValueError("Postcode moet minimaal 6 tekens bevatten (bijv. 6371AB).")
    return cleaned[:4] + cleaned[4:]


def fetch_address_by_postcode(postcode: str, house_number: int) -> Optional[Dict[str, Any]]:
    """Resolve one exact base address through PDOK's address search service."""
    normalized = normalize_postcode(postcode)
    payload = _request_json(
        LOCATION_SEARCH_URL,
        {
            "q": "*",
            "fq": f"type:adres AND postcode:{normalized} AND huisnummer:{int(house_number)}",
            "rows": 10,
            "fl": "*",
        },
    )
    documents = payload.get("response", {}).get("docs", [])
    exact_number = str(int(house_number))
    document = next(
        (item for item in documents if str(item.get("huis_nlt", "")).upper() == exact_number),
        None,
    )
    if not document:
        return None
    coordinates = re.search(
        r"POINT\(([-\d.]+)\s+([-\d.]+)\)", str(document.get("centroide_ll", ""))
    )
    geometry = (
        {"type": "Point", "coordinates": [float(coordinates.group(1)), float(coordinates.group(2))]}
        if coordinates
        else None
    )
    return {
        "bag_id": document.get("adresseerbaarobject_id"),
        "postcode": document.get("postcode"),
        "huisnummer": document.get("huisnummer"),
        "huisletter": None,
        "huisnummertoevoeging": None,
        "straat": document.get("straatnaam"),
        "woonplaats": document.get("woonplaatsnaam"),
        "provincie": document.get("provincienaam") or "onbekend",
        "geometry": geometry,
        "properties": document,
    }


def fetch_verblijfsobject_for_address(address: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    bag_id = address.get("bag_id")
    if not bag_id:
        return None

    payload = _request_json(
        f"{BAG_BASE_URL}/collections/verblijfsobject/items",
        {"f": "json", "limit": 1, "identificatie": str(bag_id)},
    )
    features = payload.get("features", [])
    if not features:
        return None
    feature = features[0]
    props = feature.get("properties", {})
    return {
        "id": props.get("identificatie"),
        "status": props.get("status"),
        "gebruiksdoel": props.get("gebruiksdoel"),
        "oppervlakte": props.get("oppervlakte"),
        "postcode": props.get("postcode"),
        "bouwjaar": props.get("bouwjaar"),
        "pand_href": props.get("pand.href", [None])[0] if isinstance(props.get("pand.href", []), list) else props.get("pand.href"),
        "geometry": feature.get("geometry"),
        "properties": props,
    }


@lru_cache(maxsize=10000)
def fetch_building_for_pand_url(pand_href: Optional[str]) -> Optional[Dict[str, Any]]:
    if not pand_href:
        return None
    payload = _request_json(pand_href, {"f": "json"})
    props = payload.get("properties", {})
    return {
        "identificatie": props.get("identificatie"),
        "bouwjaar": props.get("bouwjaar"),
        "aantal_verblijfsobjecten": props.get("aantal_verblijfsobjecten"),
        "gebruiksdoel": props.get("gebruiksdoel"),
        "status": props.get("status"),
        "geometry": payload.get("geometry"),
    }


def collect_numeric_region_records(province: str, target_count: int = 5000) -> List[Dict[str, Any]]:
    from .config import PROVINCE_BBOXES

    bbox = PROVINCE_BBOXES.get(province)
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    bboxes = [bbox]
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        tile_width = (max_lon - min_lon) / 3
        tile_height = (max_lat - min_lat) / 3
        bboxes = [[min_lon + x * tile_width, min_lat + y * tile_height,
                   min_lon + (x + 1) * tile_width, min_lat + (y + 1) * tile_height]
                  for x in range(3) for y in range(3)]
        random.SystemRandom().shuffle(bboxes)
    for tile_bbox in bboxes:
        for feature in iterate_collection_items(
            "verblijfsobject", limit=1000, bbox=tile_bbox, max_pages=25
        ):
            props = feature.get("properties", {})
            if props.get("gebruiksdoel") != "woonfunctie":
                continue
            if province != "Heel Nederland" and props.get("provincie_naam") != province:
                continue
            bag_id = props.get("identificatie")
            if bag_id in seen_ids:
                continue
            seen_ids.add(bag_id)
            record = {
            "bag_id": bag_id,
            "postcode": props.get("postcode"),
            "huisnummer": props.get("huisnummer"),
            "huisletter": props.get("huisletter"),
            "huisnummertoevoeging": props.get("toevoeging"),
            "straat": props.get("openbare_ruimte_naam"),
            "woonplaats": props.get("woonplaats_naam"),
            "provincie": props.get("provincie_naam") or province,
            "geometry": feature.get("geometry"),
            "oppervlakte_m2": props.get("oppervlakte"),
            "huisprijs_schatting": None,
            "bouwjaar": props.get("bouwjaar"),
            "gebruiksdoel": props.get("gebruiksdoel"),
            "status": props.get("status"),
            "aantal_verblijfsobjecten": 1,
            "pand_href": props.get("pand.href", [None])[0] if isinstance(props.get("pand.href", []), list) else props.get("pand.href"),
        }
            records.append(record)
            if len(records) >= target_count:
                return records
    return records


def persist_raw_records(label: str, payload: List[Dict[str, Any]]) -> Path:
    raw_dir = RAW_DIR / label
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_path = raw_dir / f"{label}.json"
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return save_path
