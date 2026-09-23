from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR
from .bag_client import collect_numeric_region_records, fetch_building_for_pand_url, persist_raw_records
from .facilities import fetch_facilities_for_points, nearest_facility_features
from .public_enrichment import enrich_with_public_sources


ADDRESS_MATCH_COLUMNS = [
    "postcode",
    "huisnummer",
    "huisletter",
    "huisnummertoevoeging",
]


def _ensure_directory(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_collection_dataframe(records: Sequence[Dict[str, Any]], facility_radius_km: float | None = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    building_cache: Dict[str, Dict[str, Any]] = {}
    pand_urls = {
        record.get("pand_href")
        for record in records
        if record.get("pand_href")
    }
    # Pand-lookups are independent network calls; bounded concurrency avoids
    # turning a large dataset into thousands of sequential requests.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_building_for_pand_url, url): url
            for url in pand_urls
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                building_cache[url] = future.result() or {}
            except Exception:
                building_cache[url] = {}
    points = [
        (float(record["geometry"]["coordinates"][1]), float(record["geometry"]["coordinates"][0]))
        for record in records
        if record.get("geometry") and len(record["geometry"].get("coordinates", [])) >= 2
    ]
    facilities_by_point = []
    facility_error = None
    if points and facility_radius_km is not None:
        try:
            facilities_by_point = fetch_facilities_for_points(points, facility_radius_km)
        except (RuntimeError, ValueError) as exc:
            facilities_by_point = [[] for _ in points]
            facility_error = str(exc)
    point_index = 0
    for record in records:
        bag_id = record.get("bag_id")
        if not bag_id:
            continue
        address = record
        pand_href = address.get("pand_href")
        if pand_href and pand_href not in building_cache:
            building_cache[pand_href] = fetch_building_for_pand_url(pand_href) or {}
        building_meta = building_cache.get(pand_href, {})
        coordinates = (address.get("geometry") or {}).get("coordinates", [None, None])
        facility_features: Dict[str, Any] = {}
        if len(coordinates) >= 2 and coordinates[0] is not None and coordinates[1] is not None and point_index < len(facilities_by_point):
            facility_features = nearest_facility_features(facilities_by_point[point_index])
            point_index += 1
        elif facility_radius_km is not None:
            facility_features = {f"afstand_{kind}_km": None for kind in ["school", "ziekenhuis", "supermarkt", "huisarts", "apotheek", "kinderopvang", "station", "sport"]}
        row = {
            "bag_id": bag_id,
            "postcode": address.get("postcode"),
            "huisnummer": address.get("huisnummer"),
            "huisletter": address.get("huisletter"),
            "huisnummertoevoeging": address.get("huisnummertoevoeging"),
            "straat": address.get("straat"),
            "woonplaats": address.get("woonplaats"),
            "provincie": address.get("provincie"),
            "gebruiksdoel": address.get("gebruiksdoel", "woonfunctie"),
            "status": address.get("status", "In gebruik"),
            "oppervlakte_m2": address.get("oppervlakte_m2"),
            "huisprijs": address.get("huisprijs"),
            "huisprijs_schatting": address.get("huisprijs_schatting"),
            "bouwjaar": address.get("bouwjaar") or building_meta.get("bouwjaar"),
            "aantal_verblijfsobjecten": building_meta.get("aantal_verblijfsobjecten", address.get("aantal_verblijfsobjecten", 1)),
            "pand_id": building_meta.get("identificatie"),
            "pand_status": building_meta.get("status"),
            "pand_gebruiksdoel": building_meta.get("gebruiksdoel"),
            "latitude": (address.get("geometry") or {}).get("coordinates", [None, None])[1],
            "longitude": (address.get("geometry") or {}).get("coordinates", [None, None])[0],
            "collected_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        postcode = str(address.get("postcode") or "").replace(" ", "").upper()
        row["postcode4"] = postcode[:4] if len(postcode) >= 4 else None
        row["postcode4_num"] = pd.to_numeric(row["postcode4"], errors="coerce")
        row["oppervlakte_bekend"] = int(pd.notna(row["oppervlakte_m2"]))
        row["bouwjaar_bekend"] = int(pd.notna(row["bouwjaar"]))
        row.update(facility_features)
        rows.append(row)
    df = pd.DataFrame(rows)
    if facility_error:
        df.attrs["facility_enrichment_error"] = facility_error
    return df


def clean_validation_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["bag_id"]).copy()
    df["postcode"] = df["postcode"].astype(str).str.replace(" ", "").str.upper()
    df["huisnummer"] = pd.to_numeric(df["huisnummer"], errors="coerce").fillna(0).astype(int)
    df["oppervlakte_m2"] = pd.to_numeric(df["oppervlakte_m2"], errors="coerce")
    df["bouwjaar"] = pd.to_numeric(df["bouwjaar"], errors="coerce")
    df["aantal_verblijfsobjecten"] = pd.to_numeric(df["aantal_verblijfsobjecten"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    for column in ["postcode4_num", "oppervlakte_bekend", "bouwjaar_bekend"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    numeric_defaults = {
        "bouwjaar": 1980.0,
        "oppervlakte_m2": 100.0,
        "aantal_verblijfsobjecten": 1.0,
    }
    for column, default in numeric_defaults.items():
        median = df[column].median()
        df[column] = df[column].fillna(median if pd.notna(median) else default)

    df["age_years"] = (datetime.utcnow().year - df["bouwjaar"]).clip(lower=1)
    df["log_oppervlakte"] = np.log1p(df["oppervlakte_m2"])
    df["status_is_active"] = df["status"].fillna("").str.contains("in gebruik|actief|gerealiseerd", case=False, regex=True).astype(int)
    postcode_median = df["postcode4_num"].median()
    df["postcode4_num"] = df["postcode4_num"].fillna(
        postcode_median if pd.notna(postcode_median) else 0
    )
    df["bouwjaar_bekend"] = df["bouwjaar_bekend"].fillna(0).astype(int)
    df["oppervlakte_bekend"] = df["oppervlakte_bekend"].fillna(0).astype(int)
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated(subset=["bag_id"]).sum())
    poisoned = int((df["status"].fillna("").str.contains("ingetrokken|niet in gebruik", case=False, regex=True)).sum()) if "status" in df.columns else 0
    return {
        "rows": int(len(df)),
        "missing_values": missing,
        "duplicate_records": duplicates,
        "retired_objects": poisoned,
        "valid_postcodes": int(df["postcode"].str.len().ge(6).sum()) if "postcode" in df.columns else 0,
    }


def ensure_valid_collection_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "collected_at" not in df.columns:
        return df
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True, errors="coerce")
    df = df[df["collected_at"].notna()].copy()
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = ensure_valid_collection_timestamp(df)
    df = clean_validation_dataset(df)
    df["area_per_unit"] = df["oppervlakte_m2"] / np.maximum(df["aantal_verblijfsobjecten"], 1)
    # BAG is an authoritative source for building characteristics, not for sales
    # prices.  Never create a price target from these same features: that would
    # make validation scores look good without teaching the model market value.
    df = df.drop(columns=["huisprijs_schatting"], errors="ignore")
    if "oppervlakte_m2_funda" in df.columns:
        df["oppervlakte_m2_funda"] = pd.to_numeric(df["oppervlakte_m2_funda"], errors="coerce")
        df["oppervlakteverschil_funda_bag"] = (
            df["oppervlakte_m2_funda"] - df["oppervlakte_m2"]
        )
    if "perceeloppervlakte_m2" in df.columns:
        df["perceeloppervlakte_m2"] = pd.to_numeric(
            df["perceeloppervlakte_m2"], errors="coerce"
        )
        df["perceel_beschikbaar"] = df["perceeloppervlakte_m2"].notna().astype(int)
    if "energielabel" in df.columns:
        df["energielabel"] = df["energielabel"].fillna("onbekend").astype(str)
        df["energielabel_score"] = (
            df["energielabel"].str.extract(r"([A-G])", expand=False)
            .map({"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1})
            .fillna(0)
        )
    df["prijs_bron"] = "BAG (geen prijs)"
    return df


def collect_regional_dataset(province: str, count: int = 5000, facility_radius_km: float | None = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    facility_notice = None
    if facility_radius_km is not None and count > 50:
        facility_notice = (
            "Voorzieningen zijn overgeslagen: maximaal 50 woningen per batch "
            "om de openbare kaartservice niet te overbelasten."
        )
        facility_radius_km = None
    records = collect_numeric_region_records(province, target_count=int(count))
    persist_raw_records(f"{province.lower().replace(' ', '_')}_bag", records)
    df = build_collection_dataframe(records, facility_radius_km)
    enrichment_error = df.attrs.get("facility_enrichment_error")
    df = feature_engineering(df)
    df = enrich_with_public_sources(df)
    validation = validate_dataset(df)
    if facility_notice:
        validation["facility_notice"] = facility_notice
    if enrichment_error:
        validation["facility_enrichment_error"] = enrichment_error
    if df.empty:
        raise ValueError(
            f"Geen woningen gevonden voor {province}. Probeer de knop opnieuw."
        )
    processed_path = _ensure_directory(PROCESSED_DIR) / f"{province.lower().replace(' ', '_')}_dataset.csv"
    df.to_csv(processed_path, index=False)
    return df, validation


def build_ml_ready_dataset(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ["provincie", "woonplaats", "gebruiksdoel"]
    for column in categorical_cols:
        if column in df.columns:
            df[column] = df[column].fillna("onbekend")
    return df


def _normalise_address_match_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return normalized exact-address columns without changing the caller's data."""
    result = frame.copy()
    result["postcode"] = (
        result["postcode"].astype("string").str.replace(" ", "").str.upper()
    )
    result["huisnummer"] = pd.to_numeric(result["huisnummer"], errors="coerce").astype("Int64")
    for column in ("huisletter", "huisnummertoevoeging"):
        result[column] = (
            result[column].astype("string").fillna("").str.replace(" ", "").str.upper()
        )
    return result


def merge_exact_funda_listings(
    dataset: pd.DataFrame, listings: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Attach only unambiguous Funda listings to BAG records.

    Postcode plus number is insufficient in the Netherlands: a house letter or
    addition can identify a distinct dwelling.  Listings whose address could not
    be parsed completely, or whose full address has conflicting prices, are kept
    out rather than assigned to the wrong property.
    """
    if dataset.empty:
        return dataset.copy(), {"matched": 0, "skipped_unparsed": 0, "skipped_conflicting": 0}
    missing_dataset = set(ADDRESS_MATCH_COLUMNS) - set(dataset.columns)
    missing_listings = set(ADDRESS_MATCH_COLUMNS + ["huisprijs", "adres_volledig_geparseerd"]) - set(listings.columns)
    if missing_dataset or missing_listings:
        raise ValueError(
            "Exacte Funda-koppeling mist adresvelden: "
            f"dataset={sorted(missing_dataset)}, listings={sorted(missing_listings)}."
        )

    left = _normalise_address_match_columns(dataset)
    right = _normalise_address_match_columns(listings)
    right["huisprijs"] = pd.to_numeric(right["huisprijs"], errors="coerce")
    candidates = right.loc[
        right["adres_volledig_geparseerd"].fillna(False)
        & right["huisprijs"].gt(0)
        & right["postcode"].notna()
        & right["huisnummer"].notna()
    ].copy()
    skipped_unparsed = int(
        right["huisprijs"].gt(0).sum() - len(candidates)
    )
    conflicting = candidates.duplicated(ADDRESS_MATCH_COLUMNS, keep=False)
    skipped_conflicting = int(conflicting.sum())
    candidates = candidates.loc[~conflicting].drop_duplicates(ADDRESS_MATCH_COLUMNS)

    listing_columns = [
        column
        for column in candidates.columns
        if column not in {*ADDRESS_MATCH_COLUMNS, "prijs_bron"}
    ]
    merged = left.merge(
        candidates[ADDRESS_MATCH_COLUMNS + listing_columns],
        on=ADDRESS_MATCH_COLUMNS,
        how="left",
        suffixes=("", "_funda"),
        indicator="_funda_match",
    )
    matched = merged["_funda_match"].eq("both")
    for column in listing_columns:
        funda_column = f"{column}_funda"
        if funda_column in merged.columns and column in dataset.columns:
            merged[column] = merged[funda_column].combine_first(merged[column])
            merged = merged.drop(columns=[funda_column])
    merged.loc[matched, "prijs_bron"] = "Funda advertentieprijs (exact adres)"
    merged = merged.drop(columns=["_funda_match"])
    return merged, {
        "matched": int(matched.sum()),
        "skipped_unparsed": skipped_unparsed,
        "skipped_conflicting": skipped_conflicting,
    }
