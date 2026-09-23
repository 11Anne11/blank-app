"""Local public-source enrichment for the regional housing dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import ParseError, iterparse

import pandas as pd

from .config import RAW_DIR


PUBLIC_DIR = RAW_DIR / "limburg_public"
ENERGY_XML = PUBLIC_DIR / "energielabels" / "extracted" / "v20260901_v4_xml.xml"
ENERGY_STATUS = ENERGY_XML.with_suffix(".status.json")
LIVE_XML_LIMIT_BYTES = 1_000_000_000
ENABLE_LIVE_ENERGY_XML = False


def _text(element: Any, suffix: str) -> str | None:
    for child in list(element):
        if str(child.tag).split("}")[-1].lower() == suffix.lower():
            return (child.text or "").strip() or None
    return None


def _energy_lookup(
    postcodes: set[str], house_numbers: set[int]
) -> tuple[dict[tuple[str, int], str], str]:
    if not ENABLE_LIVE_ENERGY_XML:
        return {}, "live_xml_uitgeschakeld"
    if not ENERGY_XML.exists() or not postcodes:
        return {}, "niet_beschikbaar"
    signature = {
        "size": ENERGY_XML.stat().st_size,
        "modified": ENERGY_XML.stat().st_mtime_ns,
    }
    if ENERGY_STATUS.exists():
        try:
            with ENERGY_STATUS.open("r", encoding="utf-8") as handle:
                cached_status = json.load(handle)
            if cached_status.get("signature") == signature:
                return {}, str(cached_status.get("status", "onbekend"))
        except (OSError, json.JSONDecodeError):
            pass
    if signature["size"] > LIVE_XML_LIMIT_BYTES:
        return {}, "te_groot_voor_live_inlezen"
    result: dict[tuple[str, int], str] = {}
    try:
        for _, element in iterparse(ENERGY_XML, events=("end",)):
            if str(element.tag).split("}")[-1].lower() != "pandcertificaat":
                continue
            postcode = (_text(element, "Postcode") or "").replace(" ", "").upper()
            house_number = pd.to_numeric(_text(element, "Huisnummer"), errors="coerce")
            label = _text(element, "Energielabel")
            if postcode in postcodes and pd.notna(house_number) and label:
                number = int(house_number)
                if number in house_numbers:
                    result[(postcode, number)] = label.upper()
            element.clear()
    except (OSError, ParseError):
        try:
            with ENERGY_STATUS.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"signature": signature, "status": "ongeldig_xml"},
                    handle,
                )
        except OSError:
            pass
        return result, "ongeldig_xml"
    return result, "gelezen"


def _read_cbs_rows(table_id: str) -> list[dict[str, Any]]:
    candidates = [
        PUBLIC_DIR / f"cbs_{table_id.lower()}" / "TypedDataSet.json",
        PUBLIC_DIR / "cbs" / f"cbs_{table_id.lower()}" / "TypedDataSet.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            rows = payload.get("value", []) if isinstance(payload, dict) else []
            return rows if isinstance(rows, list) else []
    return []


def _source_features(frame: pd.DataFrame) -> dict[str, Any]:
    cbs_rows = _read_cbs_rows("86165NED")
    energy_usable = ENABLE_LIVE_ENERGY_XML and ENERGY_XML.exists() and ENERGY_XML.stat().st_size <= LIVE_XML_LIMIT_BYTES
    if ENERGY_STATUS.exists():
        try:
            with ENERGY_STATUS.open("r", encoding="utf-8") as handle:
                energy_usable = json.load(handle).get("status") == "gelezen"
        except (OSError, json.JSONDecodeError):
            energy_usable = False
    return {
        "bron_bag_bekend": 1,
        "bron_cbs_beschikbaar": int(bool(cbs_rows)),
        "bron_cbs_records": len(cbs_rows),
        "bron_cbs_inkomen_beschikbaar": int(bool(_read_cbs_rows("86161NED"))),
        "bron_energielabel_beschikbaar": int(energy_usable),
        "bron_osm_beschikbaar": int(any((PUBLIC_DIR / "osm").rglob("*.gpkg"))),
        "bron_bgt_beschikbaar": int(any((PUBLIC_DIR / "pdok_bgt").rglob("*.gml"))),
        "bron_brt_beschikbaar": int(any((PUBLIC_DIR / "pdok_brt").rglob("*.geojson"))),
    }


def enrich_with_public_sources(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach exact energy labels and auditable source-availability features."""
    if frame.empty:
        return frame
    result = frame.copy()
    if "postcode" not in result.columns:
        result["postcode"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if "huisnummer" not in result.columns:
        result["huisnummer"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    postcode = result["postcode"].astype("string").str.replace(" ", "").str.upper()
    numbers = pd.to_numeric(result["huisnummer"], errors="coerce")
    lookup, energy_status = _energy_lookup(
        set(postcode.dropna()),
        set(numbers.dropna().astype(int)),
    )
    result["energielabel"] = [
        lookup.get((code, int(number)) if pd.notna(number) else (code, -1), existing)
        for code, number, existing in zip(postcode, numbers, result.get("energielabel", pd.Series(index=result.index, dtype="object")))
    ]
    result["energielabel_bron"] = result["energielabel"].notna().map(
        {True: "EP-Online", False: energy_status}
    )
    for column, value in _source_features(result).items():
        result[column] = value
    return result
