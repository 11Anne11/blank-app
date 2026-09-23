"""Cacheable downloads for free public housing and area data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .config import RAW_DIR


PUBLIC_DATA_DIR = RAW_DIR / "limburg_public"
MANIFEST_PATH = PUBLIC_DATA_DIR / "manifest.json"


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"files": {}}
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"files": {}}
    return manifest if isinstance(manifest, dict) else {"files": {}}


def _save_manifest(manifest: dict[str, Any]) -> None:
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    temporary.replace(MANIFEST_PATH)


def _download(url: str, destination: Path, timeout: int = 60) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)
    return destination.stat().st_size


def _download_odata_dataset(url: str, destination: Path, timeout: int = 120) -> int:
    """Download a CBS OData collection in ID ranges below the row limit."""
    rows: list[dict[str, Any]] = []
    page_size = 10000
    start_id = 0
    while True:
        response = requests.get(
            url,
            params={
                "$filter": f"ID ge {start_id} and ID lt {start_id + page_size}",
                "$top": page_size,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        page = payload.get("value", [])
        if not isinstance(page, list):
            raise ValueError("CBS gaf geen lijst met gegevens terug.")
        rows.extend(page)
        if not page:
            break
        start_id += page_size

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "odata.metadata": f"{url}/$metadata",
                "value": rows,
            },
            handle,
            ensure_ascii=False,
        )
    temporary.replace(destination)
    return destination.stat().st_size


def download_cbs_odata(table_id: str, *, force: bool = False) -> dict[str, Any]:
    """Download a CBS StatLine table and its metadata into the local cache.

    CBS table identifiers are intentionally configurable because CBS retires and
    replaces regional tables over time. The raw files are kept unchanged so
    later feature engineering can be reproduced.
    """

    normalized_id = table_id.strip().upper()
    if not normalized_id.isalnum():
        raise ValueError("Een CBS-tabelnummer mag alleen letters en cijfers bevatten.")

    base_url = f"https://opendata.cbs.nl/ODataApi/OData/{normalized_id}"
    target_dir = PUBLIC_DATA_DIR / f"cbs_{normalized_id.lower()}"
    metadata_path = target_dir / "metadata.json"
    data_path = target_dir / "TypedDataSet.json"
    manifest = _load_manifest()
    record = manifest.setdefault("files", {}).setdefault(normalized_id, {})

    if force or not metadata_path.exists():
        metadata_size = _download(f"{base_url}/DataProperties", metadata_path)
        record["metadata_bytes"] = metadata_size
    if force or not data_path.exists():
        data_size = _download_odata_dataset(f"{base_url}/TypedDataSet", data_path)
        record["data_bytes"] = data_size

    record.update(
        {
            "source": "CBS StatLine OData",
            "table_id": normalized_id,
            "url": base_url,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "metadata_path": str(metadata_path),
            "data_path": str(data_path),
        }
    )
    _save_manifest(manifest)
    return record


def public_source_catalog() -> list[dict[str, str]]:
    return [
        {
            "bron": "BAG/PDOK",
            "status": "actief",
            "gebruik": "Adres, verblijfsobject, woonoppervlakte, bouwjaar en geometrie",
        },
        {
            "bron": "CBS StatLine",
            "status": "downloadbaar",
            "gebruik": "Woningvoorraad en buurtcontext; tabelnummer is configureerbaar",
        },
        {
            "bron": "PDOK BGT/BRT",
            "status": "downloadbaar",
            "gebruik": "Gebouwen, terreinen, wegen en omgeving",
        },
        {
            "bron": "OpenStreetMap",
            "status": "actief",
            "gebruik": "Voorzieningen en afstanden; alleen openbare kaartdata",
        },
        {
            "bron": "EP-Online/RVO",
            "status": "afhankelijk van downloadtoegang",
            "gebruik": "Energielabels wanneer een openbare export beschikbaar is",
        },
    ]
