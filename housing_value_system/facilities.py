from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Any, Dict, Iterable, List, Optional

import requests


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_FALLBACK_URL = "https://overpass.kumi.systems/api/interpreter"
FACILITY_TYPES = {
    "school": '["amenity"="school"]',
    "ziekenhuis": '["amenity"="hospital"]',
    "supermarkt": '["shop"="supermarket"]',
    "huisarts": '["amenity"="doctors"]',
    "apotheek": '["amenity"="pharmacy"]',
    "kinderopvang": '["amenity"="kindergarten"]',
    "station": '["railway"="station"]',
    "sport": '["leisure"="sports_centre"]',
}


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0088
    lat_delta = radians(lat2 - lat1)
    lon_delta = radians(lon2 - lon1)
    a = sin(lat_delta / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(lon_delta / 2) ** 2
    return earth_radius_km * 2 * asin(sqrt(a))


def _element_coordinates(element: Dict[str, Any]) -> Optional[tuple[float, float]]:
    if element.get("type") == "node":
        return element.get("lat"), element.get("lon")
    center = element.get("center") or {}
    if "lat" in center and "lon" in center:
        return center["lat"], center["lon"]
    return None


def fetch_nearby_facilities(latitude: float, longitude: float, radius_km: float = 5.0) -> List[Dict[str, Any]]:
    return fetch_facilities_for_points([(latitude, longitude)], radius_km)[0]


def fetch_facilities_for_points(
    points: List[tuple[float, float]], radius_km: float = 5.0
) -> List[List[Dict[str, Any]]]:
    if not points:
        return []
    if len(points) > 50:
        raise ValueError("Voorzieningenverrijking ondersteunt maximaal 50 woningen per batch.")
    if not (0.1 <= radius_km <= 50):
        raise ValueError("De straal moet tussen 0,1 en 50 kilometer liggen.")

    point_queries = []
    for latitude, longitude in points:
        selectors = [
            f"nwr{selector}(around:{radius_km * 1000},{latitude},{longitude});"
            for selector in FACILITY_TYPES.values()
        ]
        point_queries.append("".join(selectors))
    query = f"[out:json][timeout:30];({' '.join(point_queries)});out center tags;"
    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": "NederlandseWoningWaardering/1.0"},
            timeout=12,
        )
        response.raise_for_status()
    except requests.RequestException:
        return [_fetch_facilities_for_point(point[0], point[1], radius_km) for point in points]
    elements = response.json().get("elements", [])

    facilities: List[Dict[str, Any]] = []
    seen = set()
    selector_by_type = {value: key for key, value in FACILITY_TYPES.items()}
    for element in elements:
        coordinates = _element_coordinates(element)
        if not coordinates:
            continue
        tags = element.get("tags") or {}
        kind = None
        for selector, label in selector_by_type.items():
            attribute, value = selector.strip("[]").split("=")
            if tags.get(attribute.strip('"')) == value.strip('"'):
                kind = label
                break
        if not kind:
            continue
        key = (kind, element.get("type"), element.get("id"))
        if key in seen:
            continue
        seen.add(key)
        facilities.append(
            {
                "type": kind,
                "naam": tags.get("name") or "Naam onbekend",
                "afstand_km": 0.0,
                "latitude": coordinates[0],
                "longitude": coordinates[1],
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
            }
        )
    results: List[List[Dict[str, Any]]] = []
    for latitude, longitude in points:
        nearby = [
            dict(item, afstand_km=round(_distance_km(latitude, longitude, item["latitude"], item["longitude"]), 3))
            for item in facilities
            if _distance_km(latitude, longitude, item["latitude"], item["longitude"]) <= radius_km
        ]
        results.append(sorted(nearby, key=lambda item: item["afstand_km"]))
    return results


def _fetch_facilities_for_point(latitude: float, longitude: float, radius_km: float) -> List[Dict[str, Any]]:
    selectors = [
        f"nwr{selector}(around:{radius_km * 1000},{latitude},{longitude});"
        for selector in FACILITY_TYPES.values()
    ]
    query = f"[out:json][timeout:20];({' '.join(selectors)});out center tags;"
    response = requests.post(
        OVERPASS_FALLBACK_URL,
        data={"data": query},
        headers={"User-Agent": "NederlandseWoningWaardering/1.0"},
        timeout=25,
    )
    response.raise_for_status()
    elements = response.json().get("elements", [])
    facilities = []
    selector_by_type = {value: key for key, value in FACILITY_TYPES.items()}
    for element in elements:
        coordinates = _element_coordinates(element)
        if not coordinates:
            continue
        tags = element.get("tags") or {}
        kind = None
        for selector, label in selector_by_type.items():
            attribute, value = selector.strip("[]").split("=")
            if tags.get(attribute.strip('"')) == value.strip('"'):
                kind = label
                break
        if kind:
            facilities.append(
                {
                    "type": kind,
                    "naam": tags.get("name") or "Naam onbekend",
                    "afstand_km": round(_distance_km(latitude, longitude, coordinates[0], coordinates[1]), 3),
                    "latitude": coordinates[0],
                    "longitude": coordinates[1],
                    "osm_type": element.get("type"),
                    "osm_id": element.get("id"),
                }
            )
    return sorted(facilities, key=lambda item: item["afstand_km"])


def nearest_facility_features(facilities: Iterable[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}
    for kind in FACILITY_TYPES:
        distances = [item["afstand_km"] for item in facilities if item["type"] == kind]
        result[f"afstand_{kind}_km"] = min(distances) if distances else None
    return result
