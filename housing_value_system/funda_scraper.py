"""Opt-in Funda importer based on the supplied Selenium scraper."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait


PROVINCE_SLUGS = {
    "Heel Nederland": "heel-nederland",
    "Drenthe": "drenthe",
    "Flevoland": "flevoland",
    "Friesland": "friesland",
    "Gelderland": "gelderland",
    "Groningen": "groningen",
    "Limburg": "limburg",
    "Noord-Brabant": "noord-brabant",
    "Noord-Holland": "noord-holland",
    "Overijssel": "overijssel",
    "Utrecht": "utrecht",
    "Zeeland": "zeeland",
    "Zuid-Holland": "zuid-holland",
}

FUNDA_RESULT_COLUMNS = [
    "funda_url", "funda_adres", "postcode", "huisnummer", "huisletter",
    "huisnummertoevoeging", "adres_volledig_geparseerd", "woonplaats",
    "huisprijs", "oppervlakte_m2_funda", "overige_inpandige_ruimte_m2",
    "externe_bergruimte_m2", "perceeloppervlakte_m2", "inhoud_m3",
    "gebouwgebonden_buitenruimte_m2",
    "vraagprijs_per_m2", "bouwjaar_funda", "kamers", "slaapkamers",
    "badkamers", "woonlagen", "energielabel", "woningtype", "soort_bouw",
    "soort_dak", "isolatie", "verwarming", "warm_water", "tuin",
    "ligging_tuin", "parkeergelegenheid", "prijs_bron", "prijs_type",
    "verkoopdatum", "woningstatus",
]


def _load_checkpoint(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=FUNDA_RESULT_COLUMNS)
    checkpoint = Path(path)
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        return pd.DataFrame(columns=FUNDA_RESULT_COLUMNS)
    try:
        frame = pd.read_csv(checkpoint)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=FUNDA_RESULT_COLUMNS)
    for column in FUNDA_RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[FUNDA_RESULT_COLUMNS].drop_duplicates(subset=["funda_url"])


def _save_checkpoint(frame: pd.DataFrame, path: str | Path | None) -> None:
    if path is None:
        return
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(checkpoint)


def _search_url(province: str, page: int = 1, listing_mode: str = "available") -> str:
    slug = PROVINCE_SLUGS[province]
    if listing_mode == "sold_last_year":
        url = (
            f"https://www.funda.nl/zoeken/verkocht?selected_area=provincie-{slug}"
            "&sold_since=365"
        )
    else:
        url = f"https://www.funda.nl/zoeken/koop?selected_area=provincie-{slug}"
    if page > 1:
        url += f"&page={page}"
    return url


def _clean(value: str | None) -> str | None:
    return " ".join(value.replace("\xa0", " ").split()).strip() if value else None


def _number(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[\d.]+(?:,\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group().replace(".", "").replace(",", "."))
    except ValueError:
        return None


def _price(value: str | None) -> float | None:
    digits = re.sub(r"[^\d]", "", value or "")
    return float(digits) if digits else None


def _address_key(value: str | None) -> tuple[str, int, str, str] | None:
    """Parse all BAG-relevant address parts from a Funda display address.

    A number alone is not a Dutch address key: ``12A`` and ``12-1`` can be
    distinct dwellings at the same postcode. Returning ``None`` is safer than
    guessing when the display text does not contain a complete address.
    """
    text = value or ""
    postcode_match = re.search(r"\b(\d{4}\s?[A-Z]{2})\b", text, re.I)
    if not postcode_match:
        return None
    before_postcode = text[: postcode_match.start()].rstrip(" ,")
    number_match = re.search(
        r"\b(?P<number>\d+)\s*(?P<letter>[A-Z])?(?:\s*[-/]\s*(?P<addition>[A-Z0-9]+))?\s*$",
        before_postcode,
        re.I,
    )
    if not number_match:
        return None
    return (
        re.sub(r"\s+", "", postcode_match.group(1)).upper(),
        int(number_match.group("number")),
        (number_match.group("letter") or "").upper(),
        (number_match.group("addition") or "").upper(),
    )


def _label_value(text: str, labels: tuple[str, ...]) -> str | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"(?:{label_pattern})\s*:?\s*([\d.,]+\s*(?:m²|m2|kamers?|slaapkamers?)?)",
        text,
        re.I,
    )
    return match.group(1).strip() if match else None


_FUNDA_LABELS = (
    "Vraagprijs per m²", "Vraagprijs", "Status", "Soort woonhuis",
    "Soort bouw", "Bouwjaar", "Soort dak", "Overige inpandige ruimte",
    "Externe bergruimte", "Perceel", "Inhoud", "Aantal kamers",
    "Aantal badkamers", "Badkamervoorzieningen", "Aantal woonlagen",
    "Energielabel", "Isolatie", "Verwarming", "Warm water", "Tuin",
    "Ligging tuin", "Schuur/berging", "Soort parkeergelegenheid",
)

_FUNDA_SECTION_LABELS = (
    "Overdracht", "Bouw", "Oppervlakten en inhoud", "Indeling", "Energie",
    "Kadastrale gegevens", "Buitenruimte", "Bergruimte", "Parkeergelegenheid",
    "Bekijk alle kenmerken", "Populariteit",
)


def _funda_value(text: str, label: str) -> str | None:
    others = [item for item in (*_FUNDA_LABELS, *_FUNDA_SECTION_LABELS) if item != label]
    stop = "|".join(re.escape(item) for item in sorted(others, key=len, reverse=True))
    match = re.search(
        rf"{re.escape(label)}\s*:?\s*(.*?)(?={stop}|$)",
        text,
        re.I,
    )
    return _clean(match.group(1)) if match else None


def _compact_measure(text: str, label: str, unit: str) -> str | None:
    match = re.search(
        rf"{re.escape(label)}\s*(\d+(?:[.,]\d+)?)\s*{re.escape(unit)}",
        text,
        re.I,
    )
    return match.group(1) if match else None


def _characteristics(soup: BeautifulSoup) -> dict[str, str]:
    result: dict[str, str] = {}
    for block in soup.select("dl.object-kenmerken-list"):
        for dt, dd in zip(block.find_all("dt"), block.find_all("dd")):
            name = _clean(dt.get_text(" ", strip=True))
            value = _clean(dd.get_text(" ", strip=True))
            if name and value:
                result[name] = value
    if not result:
        for dt in soup.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd:
                name = _clean(dt.get_text(" ", strip=True))
                value = _clean(dd.get_text(" ", strip=True))
                if name and value:
                    result[name] = value
    return result


def _characteristic_value(chars: dict[str, str], *labels: str) -> str | None:
    for label in labels:
        value = chars.get(label)
        if value:
            return value
    return None


def _listing_urls(driver: webdriver.Chrome, url: str, navigate: bool = True) -> list[str]:
    if navigate:
        driver.get(url)
    WebDriverWait(driver, 15).until(lambda browser: browser.find_element(By.TAG_NAME, "body"))
    # Match the supplied working scraper: allow Funda's result page to finish
    # rendering after the browser check has been completed.
    time.sleep(1)
    # Do not reload or navigate while the user is solving a CAPTCHA.
    # Funda can take several minutes to release the search page.
    deadline = time.monotonic() + 600
    last_status = 0.0
    while time.monotonic() < deadline:
        soup = BeautifulSoup(driver.page_source, "html.parser")
        urls = []
        selectors = [
            "a[href*='/detail/koop/']",
            "a[href*='/koop/'][href*='woning']",
            "a[href*='/koop/']",
        ]
        for selector in selectors:
            for link in soup.select(selector):
                href = urljoin("https://www.funda.nl", link.get("href", "")).split("?")[0]
                if "/detail/koop/" in href and href not in urls:
                    urls.append(href)
            if urls:
                break
        if not urls:
            urls = re.findall(
                r"https?://www\.funda\.nl/detail/koop/[^\"'\\\s]+",
                driver.page_source.replace("\\/", "/"),
            )
            urls = list(dict.fromkeys(urls))
        if urls:
            return urls
        if time.monotonic() - last_status >= 15:
            print("Wachten op Funda: vul de CAPTCHA/toegangscontrole in Chrome in.")
            last_status = time.monotonic()
        time.sleep(2)
    raise RuntimeError(
        "Funda gaf binnen 10 minuten geen woninglinks terug. "
        "Los eerst de CAPTCHA/toegangscontrole op in Chrome en probeer opnieuw."
    )


def _extract_listing(driver: webdriver.Chrome, url: str) -> dict[str, Any] | None:
    driver.get(url)
    WebDriverWait(driver, 15).until(lambda browser: browser.find_element(By.TAG_NAME, "body"))
    WebDriverWait(driver, 10).until(
        lambda browser: (
            "application/ld+json" in browser.page_source
            or "object-kenmerken" in browser.page_source
            or "object-header-price" in browser.page_source
        )
    )
    soup = BeautifulSoup(driver.page_source, "html.parser")
    chars = _characteristics(soup)
    product: dict[str, Any] = {}
    address: dict[str, Any] = {}
    def walk(value: Any) -> None:
        nonlocal product, address
        if isinstance(value, dict):
            kind = value.get("@type")
            if kind == "Product" or (
                isinstance(kind, list) and "Product" in kind
            ):
                product = value
            if isinstance(value.get("address"), dict):
                address = value["address"]
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(script.string or script.get_text())
        except (TypeError, json.JSONDecodeError):
            continue
        walk(payload)
    offers = product.get("offers", {})
    price = offers.get("price") if isinstance(offers, dict) else None
    if price is None:
        node = soup.select_one("[data-test='object-header-price'], .object-header__price")
        price = _price(node.get_text(" ", strip=True) if node else None)
    if price is None:
        price_match = re.search(r"(?:€\s*)?([\d.]{4,})\s*(?:k\.k\.|kosten koper|v\.o\.n\.)?", soup.get_text(" ", strip=True), re.I)
        price = _price(price_match.group(1)) if price_match else None
    if isinstance(price, str):
        price = _price(price)
    title_node = soup.select_one("h1, [data-test='object-header-title']")
    title = _clean(product.get("name") or (title_node.get_text(" ", strip=True) if title_node else None))
    if price is None and not chars:
        return None
    page_text = soup.get_text(" ", strip=True)
    address_key = _address_key(f"{title or ''} {address.get('postalCode', '')}")
    postal_code = _clean(address.get("postalCode"))
    if not postal_code:
        postal_match = re.search(r"\b\d{4}\s?[A-Z]{2}\b", f"{title} {page_text}", re.I)
        postal_code = re.sub(r"\s+", "", postal_match.group()).upper() if postal_match else None
    normalized_postcode = re.sub(r"\s+", "", postal_code).upper() if postal_code else None
    address_is_complete = bool(
        address_key and normalized_postcode and address_key[0] == normalized_postcode
    )
    living_text = (
        _characteristic_value(chars, "Gebruiksoppervlakte wonen", "Woonoppervlakte", "Wonen")
        or _compact_measure(page_text, "Wonen", r"m²")
        or _label_value(page_text, ("Gebruiksoppervlakte wonen", "Woonoppervlakte", "Wonen"))
    )
    lot_text = (
        _characteristic_value(chars, "Perceeloppervlakte", "Perceel")
        or _compact_measure(page_text, "Perceel", r"m²")
        or _label_value(page_text, ("Perceeloppervlakte", "Perceel"))
    )
    rooms_text = _characteristic_value(chars, "Aantal kamers") or _label_value(page_text, ("Aantal kamers",))
    bedrooms_text = _characteristic_value(chars, "Aantal slaapkamers") or _label_value(page_text, ("Aantal slaapkamers",))
    funda_values = {label: _funda_value(page_text, label) for label in _FUNDA_LABELS}
    # The structured dt/dd values are more reliable than the flattened page
    # text, which also contains the description and recommendation widgets.
    rooms_text = _characteristic_value(chars, "Aantal kamers") or funda_values["Aantal kamers"] or rooms_text
    bedrooms_match = re.search(r"\((\d+)\s+slaapkamers?\)", rooms_text or "", re.I)
    return {
        "funda_url": url,
        "funda_adres": title,
        "postcode": postal_code or (address_key[0] if address_key else None),
        "huisnummer": address_key[1] if address_key else None,
        "huisletter": address_key[2] if address_key else None,
        "huisnummertoevoeging": address_key[3] if address_key else None,
        "adres_volledig_geparseerd": address_is_complete,
        "woonplaats": _clean(address.get("addressLocality")),
        "huisprijs": _price(str(price)) if price is not None else None,
        "oppervlakte_m2_funda": _number(living_text),
        "overige_inpandige_ruimte_m2": _number(_characteristic_value(chars, "Overige inpandige ruimte") or _compact_measure(page_text, "Overige inpandige ruimte", r"m²") or funda_values["Overige inpandige ruimte"]),
        "externe_bergruimte_m2": _number(_characteristic_value(chars, "Externe bergruimte") or _compact_measure(page_text, "Externe bergruimte", r"m²") or funda_values["Externe bergruimte"]),
        "gebouwgebonden_buitenruimte_m2": _number(_characteristic_value(chars, "Gebouwgebonden buitenruimte")),
        "perceeloppervlakte_m2": _number(_characteristic_value(chars, "Perceeloppervlakte", "Perceel") or funda_values["Perceel"]),
        "inhoud_m3": _number(_characteristic_value(chars, "Inhoud") or funda_values["Inhoud"]),
        "vraagprijs_per_m2": _number(_characteristic_value(chars, "Vraagprijs per m²") or funda_values["Vraagprijs per m²"]),
        "bouwjaar_funda": _number(_characteristic_value(chars, "Bouwjaar") or funda_values["Bouwjaar"]),
        "kamers": _number(rooms_text),
        "slaapkamers": float(bedrooms_match.group(1)) if bedrooms_match else _number(bedrooms_text),
        "badkamers": _number(_characteristic_value(chars, "Aantal badkamers") or funda_values["Aantal badkamers"]),
        "woonlagen": _number(_characteristic_value(chars, "Aantal woonlagen") or funda_values["Aantal woonlagen"]),
        "energielabel": _characteristic_value(chars, "Energielabel") or funda_values["Energielabel"],
        "woningtype": _characteristic_value(chars, "Soort woonhuis", "Type woning") or funda_values["Soort woonhuis"],
        "soort_bouw": _characteristic_value(chars, "Soort bouw") or funda_values["Soort bouw"],
        "soort_dak": _characteristic_value(chars, "Soort dak") or funda_values["Soort dak"],
        "isolatie": _characteristic_value(chars, "Isolatie") or funda_values["Isolatie"],
        "verwarming": _characteristic_value(chars, "Verwarming") or funda_values["Verwarming"],
        "warm_water": _characteristic_value(chars, "Warm water") or funda_values["Warm water"],
        "tuin": _characteristic_value(chars, "Tuin") or funda_values["Tuin"],
        "ligging_tuin": _characteristic_value(chars, "Ligging tuin") or funda_values["Ligging tuin"],
        "parkeergelegenheid": _characteristic_value(chars, "Soort parkeergelegenheid") or funda_values["Soort parkeergelegenheid"],
        "verkoopdatum": None,
        "prijs_bron": "Funda",
        "prijs_type": "advertentieprijs",
    }


def start_funda_browser(province: str, listing_mode: str = "available") -> webdriver.Chrome:
    slug = PROVINCE_SLUGS.get(province)
    if not slug:
        raise ValueError(f"Onbekende provincie: {province}.")
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--lang=nl-NL")
    options.add_experimental_option("detach", True)
    options.page_load_strategy = "eager"
    driver = webdriver.Chrome(options=options)
    driver.get(_search_url(province, listing_mode=listing_mode))
    WebDriverWait(driver, 15).until(lambda browser: browser.find_element(By.TAG_NAME, "body"))
    return driver


def scrape_funda_listings(
    province: str,
    target: int = 25,
    pages: int = 1,
    driver: webdriver.Chrome | None = None,
    listing_mode: str = "available",
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    slug = PROVINCE_SLUGS.get(province)
    if not slug:
        raise ValueError(f"Onbekende provincie: {province}.")
    if not 1 <= target <= 100000 or not 1 <= pages <= 10000:
        raise ValueError("Funda-import ondersteunt 1-100.000 woningen en 1-10.000 pagina's.")
    owns_driver = driver is None
    manual_driver = driver is not None
    if driver is None:
        driver = start_funda_browser(province, listing_mode=listing_mode)
    saved = _load_checkpoint(checkpoint_path)
    rows: list[dict[str, Any]] = saved.to_dict("records")
    seen_urls = set(saved["funda_url"].dropna().astype(str))
    if len(rows) >= target:
        return saved.head(target)
    try:
        for page in range(1, pages + 1):
            url = _search_url(province, page, listing_mode)
            # The manual CAPTCHA is solved in this browser profile. Reopen the
            # search URL afterwards, just like the supplied working scraper.
            try:
                listing_urls = _listing_urls(
                    driver,
                    url,
                    navigate=not (manual_driver and page == 1),
                )
            except (TimeoutException, WebDriverException) as exc:
                print(f"Funda-pagina overgeslagen door browserfout: pagina {page} ({type(exc).__name__})")
                if manual_driver:
                    raise RuntimeError(
                        "De handmatig geopende Funda-browser verloor de verbinding. "
                        "Laat Chrome open en open daarna opnieuw een CAPTCHA-sessie."
                    ) from exc
                try:
                    driver.quit()
                except WebDriverException:
                    pass
                driver = start_funda_browser(province, listing_mode=listing_mode)
                owns_driver = True
                continue
            for listing_url in listing_urls:
                if len(rows) >= target:
                    break
                if listing_url in seen_urls:
                    continue
                try:
                    row = _extract_listing(driver, listing_url)
                except (TimeoutException, WebDriverException) as exc:
                    print(f"Funda-woning overgeslagen door browserfout: {listing_url} ({type(exc).__name__})")
                    continue
                if row:
                    rows.append(row)
                    seen_urls.add(listing_url)
                    _save_checkpoint(pd.DataFrame(rows, columns=FUNDA_RESULT_COLUMNS), checkpoint_path)
    finally:
        if owns_driver:
            try:
                driver.quit()
            except WebDriverException:
                pass
    return pd.DataFrame(rows, columns=FUNDA_RESULT_COLUMNS).drop_duplicates(subset=["funda_url"])


def scrape_funda_listings_parallel(
    province: str,
    target: int = 25,
    pages: int = 1,
    workers: int = 1,
    listing_mode: str = "available",
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """Scrape disjoint page ranges with separate Chrome sessions."""
    if workers <= 1:
        return scrape_funda_listings(
            province,
            target,
            pages,
            listing_mode=listing_mode,
            checkpoint_path=checkpoint_path,
        )
    if workers > min(4, pages):
        raise ValueError("Het aantal parallelle browsers mag niet groter zijn dan 4 of het aantal pagina's.")
    page_groups = [[] for _ in range(workers)]
    for index, page in enumerate(range(1, pages + 1)):
        page_groups[index % workers].append(page)
    target_per_worker = (target + workers - 1) // workers

    def run_group(group: list[int]) -> pd.DataFrame:
        driver = start_funda_browser(province, listing_mode=listing_mode)
        try:
            worker_checkpoint = None
            if checkpoint_path is not None:
                checkpoint = Path(checkpoint_path)
                worker_checkpoint = checkpoint.with_name(
                    f"{checkpoint.stem}_pages_{group[0]}{checkpoint.suffix}"
                )
            saved = _load_checkpoint(worker_checkpoint)
            rows: list[dict[str, Any]] = saved.to_dict("records")
            seen_urls = set(saved["funda_url"].dropna().astype(str))
            for page in group:
                url = _search_url(province, page, listing_mode)
                try:
                    listing_urls = _listing_urls(driver, url, navigate=True)
                except (TimeoutException, WebDriverException) as exc:
                    print(f"Funda-pagina overgeslagen door browserfout: pagina {page} ({type(exc).__name__})")
                    try:
                        driver.quit()
                    except WebDriverException:
                        pass
                    driver = start_funda_browser(province, listing_mode=listing_mode)
                    continue
                for listing_url in listing_urls:
                    if len(rows) >= target_per_worker:
                        break
                    if listing_url in seen_urls:
                        continue
                    try:
                        row = _extract_listing(driver, listing_url)
                    except (TimeoutException, WebDriverException) as exc:
                        print(f"Funda-woning overgeslagen door browserfout: {listing_url} ({type(exc).__name__})")
                        continue
                    if row:
                        rows.append(row)
                        seen_urls.add(listing_url)
                        _save_checkpoint(pd.DataFrame(rows, columns=FUNDA_RESULT_COLUMNS), worker_checkpoint)
            return pd.DataFrame(rows, columns=FUNDA_RESULT_COLUMNS)
        finally:
            try:
                driver.quit()
            except WebDriverException:
                pass

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_group, group) for group in page_groups if group]
        for future in as_completed(futures):
            frames.append(future.result())
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["funda_url"]).head(target)
    _save_checkpoint(result, checkpoint_path)
    return result


def scrape_funda_listings_both(
    province: str,
    target: int = 25,
    pages: int = 1,
    workers: int = 1,
    checkpoint_dir: str | Path | None = None,
    driver: webdriver.Chrome | None = None,
) -> pd.DataFrame:
    """Collect available and sold listings, splitting the requested target."""
    available_target = (target + 1) // 2
    sold_target = target - available_target
    scraper = scrape_funda_listings_parallel if workers > 1 else scrape_funda_listings
    common = {"province": province, "pages": pages, "listing_mode": "available"}
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    available_checkpoint = checkpoint_root / "beschikbaar.csv" if checkpoint_root else None
    sold_checkpoint = checkpoint_root / "verkocht.csv" if checkpoint_root else None
    if workers > 1:
        available = scraper(target=available_target, workers=workers, checkpoint_path=available_checkpoint, **common)
        sold = scraper(
            province,
            target=sold_target,
            pages=pages,
            workers=workers,
            listing_mode="sold_last_year",
            checkpoint_path=sold_checkpoint,
        )
    else:
        available = scraper(
            target=available_target,
            checkpoint_path=available_checkpoint,
            driver=driver,
            **common,
        )
        sold = scraper(
            province,
            target=sold_target,
            pages=pages,
            listing_mode="sold_last_year",
            checkpoint_path=sold_checkpoint,
        )
    if not available.empty:
        available["woningstatus"] = "beschikbaar"
        available["prijs_type"] = "vraagprijs"
    if not sold.empty:
        sold["woningstatus"] = "verkocht_laatste_12_maanden"
        # Funda exposes the advertised amount, not a verified transaction price.
        sold["prijs_type"] = "laatst_geadverteerde_prijs"
    frames = [frame for frame in (available, sold) if not frame.empty]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["funda_url"]) if frames else pd.DataFrame()
