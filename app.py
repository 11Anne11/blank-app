from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from pandas.errors import EmptyDataError

from housing_value_system.bag_client import fetch_address_by_postcode, fetch_building_for_pand_url, fetch_verblijfsobject_for_address
from housing_value_system.config import MODEL_DIR, PROVINCES
from housing_value_system.data_pipeline import collect_regional_dataset, merge_exact_funda_listings
from housing_value_system.facilities import fetch_nearby_facilities, nearest_facility_features
from housing_value_system.modeling import is_current_model_artifact, model_input_features, predict_house_value, train_model_from_dataset
from housing_value_system.public_data import download_cbs_odata, public_source_catalog
from housing_value_system.public_enrichment import enrich_with_public_sources


CURRENT_YEAR = datetime.now().year
TRAINING_TARGET_LABELS = {
    "huisprijs": "Funda-advertentieprijs",
    "verkoopprijs": "Verkoopprijs",
    "woz_waarde": "WOZ-waarde",
    "waarde_index": "Eigen geverifieerde waarde-index",
}


def load_funda_scraper():
    try:
        from housing_value_system.funda_scraper import (
            scrape_funda_listings,
            scrape_funda_listings_both,
            scrape_funda_listings_parallel,
            start_funda_browser,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Funda-import is niet beschikbaar. Installeer beautifulsoup4 en selenium."
        ) from exc
    return (
        scrape_funda_listings,
        scrape_funda_listings_both,
        scrape_funda_listings_parallel,
        start_funda_browser,
    )


st.set_page_config(
    page_title="Woonwaarde | Nederlandse woningwaardering",
    page_icon=":material/home:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    :root { --navy: #17324d; --ink: #203040; --muted: #687786; --sand: #f7f4ef; --gold: #c9944a; }
    .stApp { background: var(--sand); color: var(--ink); }
    .block-container { max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: var(--navy); }
    h1, h2 { font-family: 'Playfair Display', Georgia, serif; }
    h1 { font-size: 3.2rem !important; letter-spacing: -0.04em; margin-bottom: .2rem; }
    .eyebrow { color: var(--gold); font-size: .78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .intro { color: var(--muted); font-size: 1.08rem; max-width: 690px; margin-bottom: 1.8rem; }
    .section-label { color: var(--navy); font-size: .8rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin: .3rem 0 .7rem; }
    div[data-testid="stForm"], div[data-testid="stExpander"], .result-card {
        background: #fff; border: 1px solid #e7e1d8; border-radius: 16px; padding: 1.35rem;
        box-shadow: 0 8px 24px rgba(38, 49, 59, .05);
    }
    div[data-testid="stForm"] { padding: 1.2rem 1.45rem 1.45rem; }
    .result-card { border-top: 4px solid var(--gold); }
    .result-label { color: var(--muted); font-size: .9rem; margin-bottom: .35rem; }
    .result-price { color: var(--navy); font-family: 'Playfair Display', Georgia, serif; font-size: 2.7rem; font-weight: 700; line-height: 1.05; }
    .result-range { color: var(--muted); margin-top: .7rem; }
    .stButton > button[kind="primary"] { background: var(--navy); border: 0; border-radius: 10px; font-weight: 600; padding: .7rem 1.4rem; }
    .stButton > button[kind="primary"]:hover { background: #285274; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #e7e1d8; border-radius: 12px; padding: .8rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def make_location_features(
    address: dict, apartment: dict | None, building: dict | None
) -> dict:
    """Build only the location-derived model inputs from authoritative BAG data."""
    geometry = (apartment or {}).get("geometry") or (address or {}).get("geometry") or {}
    coordinates = geometry.get("coordinates") or [None, None]
    postcode = str((address or {}).get("postcode") or (apartment or {}).get("postcode") or "")
    return {
        "postcode": (address or {}).get("postcode", ""),
        "huisnummer": (address or {}).get("huisnummer", 0),
        "huisletter": (address or {}).get("huisletter", ""),
        "provincie": (address or {}).get("provincie", "onbekend"),
        "woonplaats": (address or {}).get("woonplaats", "onbekend"),
        "gebruiksdoel": (apartment or {}).get("gebruiksdoel", "woonfunctie"),
        "aantal_verblijfsobjecten": float((building or {}).get("aantal_verblijfsobjecten", 1) or 1),
        "postcode4": postcode.replace(" ", "").upper()[:4],
        "postcode4_num": float(pd.to_numeric(postcode.replace(" ", "")[:4], errors="coerce") or 0),
        "latitude": coordinates[1] if len(coordinates) >= 2 else None,
        "longitude": coordinates[0] if len(coordinates) >= 2 else None,
        "status": (apartment or {}).get("status", ""),
        "status_is_active": int("in gebruik" in str((apartment or {}).get("status", "")).lower()),
        "pand_status": (building or {}).get("status", ""),
        "pand_gebruiksdoel": (building or {}).get("gebruiksdoel", ""),
    }


def available_training_targets(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in TRAINING_TARGET_LABELS
        if column in frame.columns
        and pd.to_numeric(frame[column], errors="coerce").gt(0).any()
    ]


def read_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".json"):
        payload = json.load(uploaded_file)
        if isinstance(payload, dict) and "records" in payload:
            payload = payload["records"]
        if not isinstance(payload, list):
            raise ValueError("JSON moet een lijst records bevatten of een object met de sleutel 'records'.")
        return pd.DataFrame(payload)
    return pd.read_csv(uploaded_file)


def find_funda_listing(postcode: str, house_number: int) -> pd.Series | None:
    path = Path("data/processed/funda_resultaten.csv")
    if not path.exists():
        return None
    try:
        listings = pd.read_csv(path)
    except (OSError, EmptyDataError, pd.errors.ParserError):
        return None
    required = {"postcode", "huisnummer", "huisprijs"}
    if not required.issubset(listings.columns):
        return None
    listings = listings.copy()
    listings["postcode"] = listings["postcode"].astype("string").str.replace(" ", "").str.upper()
    listings["huisnummer"] = pd.to_numeric(listings["huisnummer"], errors="coerce")
    matches = listings[
        (listings["postcode"] == postcode.replace(" ", "").upper())
        & (listings["huisnummer"] == int(house_number))
        & pd.to_numeric(listings["huisprijs"], errors="coerce").notna()
    ]
    return matches.iloc[0] if not matches.empty else None


def funda_number(values: dict, key: str, default: float = 0) -> float:
    value = values.get(key)
    return float(value) if value is not None and pd.notna(value) else default


st.markdown('<div class="eyebrow">Woonwaarde</div>', unsafe_allow_html=True)
st.title("Wat is je woning waard?")
st.markdown(
    '<div class="intro">Een rustige, onderbouwde inschatting van de waarde van een Nederlandse woning. '
    'Vul de belangrijkste kenmerken in en krijg direct een prijsindicatie.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Over Woonwaarde")
    st.write("Een prijsindicatie op basis van woningkenmerken, BAG-locatiegegevens en een gevalideerd model.")
    artifact = MODEL_DIR / "best_housing_model.pkl"
    if artifact.exists() and is_current_model_artifact(artifact):
        st.success("Waarderingsmodel beschikbaar (inclusief oudere modellen)")
        learned_features = model_input_features(artifact)
        if "woningtype" not in learned_features:
            st.info(
                "Dit is het oude model. Het gebruikt alleen BAG-basisgegevens, locatie "
                "en woonoppervlakte. De extra velden en vinkjes worden pas gebruikt "
                "nadat je een nieuw model traint."
            )
    elif artifact.exists():
        st.warning("Verouderd model: train opnieuw met echte prijsrecords")
    else:
        st.warning("Nog geen waarderingsmodel geladen")

selected_tab = st.tabs(["Woningwaardering", "Dataset verzamelen", "Modeltraining"])

with selected_tab[0]:
    st.markdown('<div class="section-label">Woninggegevens</div>', unsafe_allow_html=True)
    st.caption("Postcode en huisnummer zijn nodig om de locatie betrouwbaar uit de BAG op te halen. Overige details mag je zelf aanvullen.")
    with st.form("valuation_form"):
        st.markdown("**Locatie**")
        location_left, location_right = st.columns(2)
        with location_left:
            model_postcode = st.text_input("Postcode", placeholder="bijv. 6371AB", key="model_postcode")
        with location_right:
            model_house_number = st.number_input(
                "Huisnummer", min_value=0, value=0, step=1, key="model_house_number"
            )
        st.markdown("**De basis**")
        basic_left, basic_mid, basic_right = st.columns(3)
        with basic_left:
            model_area = st.number_input("Woonoppervlakte (m²)", min_value=0.0, value=100.0, step=1.0, key="model_area")
            model_lot = st.number_input("Perceeloppervlakte (m²)", min_value=0.0, value=0.0, step=1.0, key="model_lot")
            model_year = st.number_input("Bouwjaar", min_value=0, max_value=2100, value=1980, step=1, key="model_year")
        with basic_mid:
            model_volume = st.number_input("Inhoud (m³)", min_value=0.0, value=0.0, step=1.0, key="model_volume")
            model_rooms = st.number_input("Aantal kamers", min_value=0, max_value=50, value=4, step=1, key="model_rooms")
            model_bedrooms = st.number_input("Aantal slaapkamers", min_value=0, max_value=50, value=2, step=1, key="model_bedrooms")
        with basic_right:
            model_bathrooms = st.number_input("Aantal badkamers", min_value=0, max_value=20, value=1, step=1, key="model_bathrooms")
            model_floors = st.number_input("Aantal woonlagen", min_value=0, max_value=20, value=1, step=1, key="model_floors")
            label_options = ["onbekend", "A+++", "A++", "A+", "A", "B", "C", "D", "E", "F", "G"]
            model_label = st.selectbox("Energielabel", label_options, key="model_label")

        st.markdown("**Type en afwerking**")
        type_left, type_mid, type_right = st.columns(3)
        with type_left:
            model_type = st.selectbox("Woningtype", ["Onbekend", "Eengezinswoning", "Tussenwoning", "Hoekwoning", "2-onder-1-kapwoning", "Vrijstaande woning", "Appartement", "Bovenwoning", "Benedenwoning", "Maisonnette", "Seniorenwoning"], key="model_type")
            model_build_type = st.selectbox("Soort bouw", ["Onbekend", "Bestaande bouw", "Nieuwbouw"], key="model_build_type")
            model_roof = st.selectbox("Soort dak", ["Onbekend", "Zadeldak", "Plat dak", "Lessenaarsdak", "Mansardedak", "Schildak"], key="model_roof")
        with type_mid:
            model_other_area = st.number_input("Overige inpandige ruimte (m²)", min_value=0.0, value=0.0, step=1.0, key="model_other_area")
            model_storage_area = st.number_input("Externe bergruimte (m²)", min_value=0.0, value=0.0, step=1.0, key="model_storage_area")
            model_attached_outdoor_area = st.number_input("Gebouwgebonden buitenruimte (m²)", min_value=0.0, value=0.0, step=1.0, key="model_attached_outdoor_area")
            model_insulation = st.selectbox("Isolatie", ["Onbekend", "Geen isolatie", "Dakisolatie", "Muurisolatie", "Vloerisolatie", "Dubbel glas", "Volledig geïsoleerd"], key="model_insulation")
        with type_right:
            model_heating = st.selectbox("Verwarming", ["Onbekend", "Cv-ketel", "Stadsverwarming", "Vloerverwarming", "Houtkachel", "Warmtepomp", "Elektrische verwarming"], key="model_heating")
            model_hot_water = st.selectbox("Warm water", ["Onbekend", "Cv-ketel", "Boiler", "Zonneboiler", "Stadsverwarming", "Warmtepomp"], key="model_hot_water")
            model_parking = st.selectbox("Parkeren", ["Onbekend", "Openbaar parkeren", "Parkeervergunningen", "Op eigen terrein", "Carport", "Garage", "Geen parkeergelegenheid"], key="model_parking")

        with st.expander("Tuin en buitenruimte", expanded=False):
            garden_left, garden_right = st.columns(2)
            with garden_left:
                model_garden = st.selectbox("Tuin", ["Onbekend", "Geen tuin", "Achtertuin", "Voortuin", "Zijtuin", "Balkon", "Dakterras", "Tuin rondom", "Achtertuin en voortuin"], key="model_garden")
            with garden_right:
                model_garden_location = st.selectbox("Ligging", ["Onbekend", "Noorden", "Noordoosten", "Oosten", "Zuidoosten", "Zuiden", "Zuidwesten", "Westen", "Noordwesten"], key="model_garden_location")

        with st.expander("Welke gegevens weet je?", expanded=False):
            st.caption("Laat een vinkje staan als je het kenmerk weet. Haal het vinkje weg als je het niet weet.")
            known_left, known_mid, known_right = st.columns(3)
            with known_left:
                known_area = st.checkbox("Woonoppervlakte", value=True, key="known_area")
                known_lot = st.checkbox("Perceeloppervlakte", value=True, key="known_lot")
                known_year = st.checkbox("Bouwjaar", value=True, key="known_year")
                known_volume = st.checkbox("Inhoud", value=True, key="known_volume")
                known_rooms = st.checkbox("Aantal kamers", value=True, key="known_rooms")
                known_bedrooms = st.checkbox("Aantal slaapkamers", value=True, key="known_bedrooms")
                known_bathrooms = st.checkbox("Aantal badkamers", value=True, key="known_bathrooms")
            with known_mid:
                known_floors = st.checkbox("Aantal woonlagen", value=True, key="known_floors")
                known_label = st.checkbox("Energielabel", value=True, key="known_label")
                known_type = st.checkbox("Woningtype", value=True, key="known_type")
                known_build_type = st.checkbox("Soort bouw", value=True, key="known_build_type")
                known_roof = st.checkbox("Soort dak", value=True, key="known_roof")
                known_other_area = st.checkbox("Overige inpandige ruimte", value=True, key="known_other_area")
                known_storage_area = st.checkbox("Externe bergruimte", value=True, key="known_storage_area")
            with known_right:
                known_attached_outdoor_area = st.checkbox("Gebouwgebonden buitenruimte", value=True, key="known_attached_outdoor_area")
                known_insulation = st.checkbox("Isolatie", value=True, key="known_insulation")
                known_heating = st.checkbox("Verwarming", value=True, key="known_heating")
                known_hot_water = st.checkbox("Warm water", value=True, key="known_hot_water")
                known_parking = st.checkbox("Parkeren", value=True, key="known_parking")
                known_garden = st.checkbox("Tuin", value=True, key="known_garden")
                known_garden_location = st.checkbox("Ligging tuin", value=True, key="known_garden_location")

        calculate = st.form_submit_button("Bereken de woningwaarde", type="primary", width="stretch")

    if calculate:
        artifact = MODEL_DIR / "best_housing_model.pkl"
        if not artifact.exists():
            st.error("Er is nog geen getraind model. Train eerst een model in de tab Modeltraining.")
        elif not is_current_model_artifact(artifact):
            st.error("Het modelbestand kan niet veilig worden geladen. Train een nieuw model.")
        elif not model_postcode.strip() or int(model_house_number) <= 0:
            st.error("Vul een geldige postcode en een huisnummer in voor een locatiegebonden waardering.")
        elif known_area and float(model_area) <= 0:
            st.error("Vul een woonoppervlakte groter dan 0 m² in, of haal het vinkje bij Woonoppervlakte weg.")
        else:
            try:
                address = fetch_address_by_postcode(model_postcode, int(model_house_number))
                if not address:
                    raise ValueError("Adres niet gevonden in de BAG. Controleer postcode en huisnummer.")
                apartment = fetch_verblijfsobject_for_address(address)
                if not apartment:
                    raise ValueError("Voor dit adres is geen verblijfsobject gevonden in de BAG.")
                building = fetch_building_for_pand_url(apartment.get("pand_href"))
                location_features = make_location_features(address, apartment, building)
                manual_area = float(model_area) if known_area and float(model_area) > 0 else np.nan
                manual_year = float(model_year) if known_year and float(model_year) > 0 else np.nan
                manual_age = max(0.0, float(CURRENT_YEAR) - manual_year) if pd.notna(manual_year) else np.nan
                manual_row = pd.Series({
                    **location_features,
                    "oppervlakte_m2": manual_area,
                    "bouwjaar": manual_year,
                    "age_years": manual_age,
                    "log_oppervlakte": float(np.log1p(manual_area)) if pd.notna(manual_area) else np.nan,
                    "area_per_unit": manual_area / max(float(location_features["aantal_verblijfsobjecten"]), 1.0),
                    "oppervlakte_bekend": int(known_area and pd.notna(manual_area)),
                    "bouwjaar_bekend": int(known_year and pd.notna(manual_year)),
                    "oppervlakte_m2_funda": manual_area,
                    "overige_inpandige_ruimte_m2": float(model_other_area) if known_other_area and float(model_other_area) > 0 else np.nan,
                    "externe_bergruimte_m2": float(model_storage_area) if known_storage_area and float(model_storage_area) > 0 else np.nan,
                    "gebouwgebonden_buitenruimte_m2": float(model_attached_outdoor_area) if known_attached_outdoor_area and float(model_attached_outdoor_area) > 0 else np.nan,
                    "perceeloppervlakte_m2": float(model_lot) if known_lot and float(model_lot) > 0 else np.nan,
                    "inhoud_m3": float(model_volume) if known_volume and float(model_volume) > 0 else np.nan,
                    "vraagprijs_per_m2": np.nan,
                    "bouwjaar_funda": manual_year,
                    "kamers": float(model_rooms) if known_rooms and float(model_rooms) > 0 else np.nan,
                    "slaapkamers": float(model_bedrooms) if known_bedrooms and float(model_bedrooms) > 0 else np.nan,
                    "badkamers": float(model_bathrooms) if known_bathrooms and float(model_bathrooms) > 0 else np.nan,
                    "woonlagen": float(model_floors) if known_floors and float(model_floors) > 0 else np.nan,
                    "energielabel": model_label if known_label else "onbekend",
                    "woningtype": model_type if known_type else "onbekend",
                    "soort_bouw": model_build_type if known_build_type else "onbekend",
                    "soort_dak": model_roof if known_roof else "onbekend",
                    "isolatie": model_insulation if known_insulation else "onbekend",
                    "verwarming": model_heating if known_heating else "onbekend",
                    "warm_water": model_hot_water if known_hot_water else "onbekend",
                    "tuin": model_garden if known_garden else "onbekend",
                    "ligging_tuin": model_garden_location if known_garden_location else "onbekend",
                    "parkeergelegenheid": model_parking if known_parking else "onbekend",
                    "energielabel_score": {"A+++": 7, "A++": 7, "A+": 7, "A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}.get(model_label, 0) if known_label else 0,
                    "perceel_beschikbaar": int(known_lot and float(model_lot) > 0),
                })
                prediction = predict_house_value(str(artifact), manual_row)
            except Exception as exc:
                st.error(f"De waarde kon niet worden berekend: {exc}")
            else:
                price = f"€ {prediction['prediction']:,.0f}".replace(",", ".")
                lower = f"€ {prediction['lower_bound']:,.0f}".replace(",", ".")
                upper = f"€ {prediction['upper_bound']:,.0f}".replace(",", ".")
                st.markdown(
                    f'<div class="result-card"><div class="result-label">Geschatte marktwaarde</div>'
                    f'<div class="result-price">{price}</div>'
                    f'<div class="result-range">Indicatieve bandbreedte: {lower} – {upper}</div></div>',
                    unsafe_allow_html=True,
                )
                st.caption("De bandbreedte is gebaseerd op de validatiefouten van het model, niet op een vaste ±10%. De uitkomst is geen taxatierapport.")
with selected_tab[1]:
    st.subheader("Verzamel een ML-ready dataset")
    with st.expander("Gratis openbare gebiedsbronnen", expanded=False):
        st.write("Bronnen worden lokaal gecachet en overschrijven geen BAG- of Funda-velden.")
        st.dataframe(pd.DataFrame(public_source_catalog()), hide_index=True, width="stretch")
        cbs_table = st.text_input(
            "CBS StatLine-tabelnummer",
            value="86165NED",
            help="Gebruik een bestaand CBS-tabelnummer. Metadata en TypedDataSet worden lokaal bewaard.",
        )
        cbs_force = st.checkbox("CBS-bestand opnieuw downloaden", value=False)
        if st.button("CBS-tabel voor Limburg downloaden"):
            try:
                with st.spinner("CBS metadata en ruwe tabel downloaden..."):
                    cbs_record = download_cbs_odata(cbs_table, force=cbs_force)
                st.success(
                    f"CBS-tabel {cbs_record['table_id']} is lokaal opgeslagen "
                    f"({cbs_record.get('data_bytes', 0):,} bytes)."
                )
                st.caption(
                    "De ruwe bestanden staan onder data/raw/limburg_public. "
                    "Ze worden pas na expliciete feature-engineering voor training gebruikt."
                )
            except (ValueError, OSError, requests.RequestException) as exc:
                st.error(f"CBS-download mislukt ({type(exc).__name__}): {exc}")
    province = st.selectbox("Provincie", PROVINCES, index=12)
    count = st.number_input("Aantal woningen", min_value=10, max_value=100000, value=500, step=10)
    include_dataset_facilities = st.checkbox("Voorzieningen toevoegen aan dataset", value=False)
    include_funda_prices = st.checkbox("Funda-advertentieprijzen toevoegen (optioneel)", value=False)
    funda_listing_mode = st.selectbox(
        "Type Funda-woningen",
        ["Beschikbare en verkochte woningen", "Alleen beschikbare woningen", "Alleen verkocht in de afgelopen 12 maanden"],
        disabled=not include_funda_prices,
        help="De gecombineerde optie haalt beide groepen op. Dit zijn advertentieprijzen; een als verkocht gemarkeerde advertentie bevat geen geverifieerde transactieprijs.",
    )
    funda_mode = (
        "both"
        if funda_listing_mode.startswith("Beschikbare en")
        else "sold_last_year"
        if funda_listing_mode.startswith("Alleen verkocht")
        else "available"
    )
    if include_funda_prices and funda_mode != "available":
        st.caption("De prijs blijft gelabeld als advertentieprijs; de app voert geen speculatieve indexcorrectie uit.")
    funda_pages = st.number_input(
        "Minimaal aantal Funda-pagina's",
        min_value=1,
        max_value=10000,
        value=1,
        step=1,
        disabled=not include_funda_prices,
        help="Funda toont vaak ongeveer 10 woningen per pagina. Je kunt maximaal 10.000 pagina's opgeven.",
    )
    funda_workers = st.number_input(
        "Gelijktijdige Funda-browsers",
        min_value=1,
        max_value=4,
        value=1,
        step=1,
        disabled=not include_funda_prices or funda_pages < 2,
        help="Elke browser krijgt andere pagina's. Bij meer dan 1 browser kan Funda per browser een CAPTCHA vragen.",
    )
    funda_target = st.number_input(
        "Aantal Funda-woningen",
        min_value=5,
        max_value=100000,
        value=10,
        step=5,
        disabled=not include_funda_prices,
        help="Elke detailpagina wordt apart geladen. Ongeveer 10 woningen per pagina; grote imports kunnen zeer lang duren.",
    )
    if include_funda_prices:
        required_funda_pages = max(int(funda_pages), math.ceil(int(funda_target) / 10))
        st.caption(f"Voor {int(funda_target)} Funda-woningen worden maximaal {required_funda_pages} pagina's verwerkt.")
    if "funda_driver" not in st.session_state:
        st.session_state.funda_driver = None
    if "funda_last_results" not in st.session_state:
        st.session_state.funda_last_results = pd.DataFrame()
    if include_funda_prices:
        required_funda_pages = max(int(funda_pages), math.ceil(int(funda_target) / 10))
        if st.button("1. Funda-pagina handmatig openen"):
            try:
                *_, start_funda_browser = load_funda_scraper()
                if st.session_state.funda_driver is not None:
                    st.session_state.funda_driver.quit()
                st.session_state.funda_driver = start_funda_browser(
                    province,
                    listing_mode="available" if funda_mode == "both" else funda_mode,
                )
                st.success("Chrome is geopend. Vul de CAPTCHA handmatig in en laat de resultatenpagina open.")
            except Exception as exc:
                st.error(f"Funda-browser openen mislukt: {exc}")
        if funda_mode == "both":
            st.caption("De gecombineerde import opent aparte sessies voor beschikbare en verkochte advertenties. Los een eventuele CAPTCHA in elk venster op.")
        else:
            st.caption("Open Chrome eerst met de knop hierboven, vul de CAPTCHA in en klik daarna op Dataset bouwen.")
        if not st.session_state.funda_last_results.empty:
            st.subheader("Laatste opgehaalde Funda-informatie")
            st.dataframe(st.session_state.funda_last_results, width="stretch")
    dataset_facility_radius = st.number_input("Voorzieningen binnen (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.5, key="dataset_facility_radius", disabled=not include_dataset_facilities)
    if st.button("Dataset bouwen"):
        with st.spinner("Echte BAG-woningen ophalen en m² verrijken..."):
            manual_driver = None
            build_completed = False
            try:
                radius = float(dataset_facility_radius) if include_dataset_facilities else None
                funda_matches = 0
                funda_merge_stats = {"matched": 0, "skipped_unparsed": 0, "skipped_conflicting": 0}
                funda = pd.DataFrame()
                if include_funda_prices:
                    (
                        scrape_funda_listings,
                        scrape_funda_listings_both,
                        scrape_funda_listings_parallel,
                        _,
                    ) = load_funda_scraper()
                    manual_driver = st.session_state.funda_driver
                    if int(funda_workers) > 1:
                        st.info("Er worden meerdere Chrome-vensters geopend; elk venster verwerkt andere Funda-pagina's. Funda kan per venster een CAPTCHA tonen.")
                        if manual_driver is not None:
                            manual_driver.quit()
                            manual_driver = None
                    elif funda_mode == "both":
                        st.info("De geopende Chrome-sessie wordt eerst voor beschikbare woningen gebruikt. Daarna opent de verkochte zoekopdracht een aparte sessie.")
                    else:
                        st.info("Chrome wordt nu geopend voor Funda. Los eerst de CAPTCHA op; daarna gaat de datasetbouw verder.")
                    if funda_mode == "both":
                        funda = scrape_funda_listings_both(
                            province,
                            target=int(funda_target),
                            pages=required_funda_pages,
                            workers=int(funda_workers),
                            checkpoint_dir="data/processed/funda_checkpoints",
                            driver=manual_driver,
                        )
                    else:
                        funda_scraper = scrape_funda_listings_parallel if int(funda_workers) > 1 else scrape_funda_listings
                        funda = funda_scraper(
                            province,
                            target=int(funda_target),
                            pages=required_funda_pages,
                            listing_mode=funda_mode,
                            checkpoint_path=f"data/processed/funda_checkpoints/{funda_mode}.csv",
                            **(
                                {"workers": int(funda_workers)}
                                if int(funda_workers) > 1
                                else {"driver": manual_driver}
                            ),
                        )
                    st.session_state.funda_last_results = funda.copy()
                    if not funda.empty:
                        funda.to_csv("data/processed/funda_resultaten.csv", index=False)
                dataset, validation = collect_regional_dataset(province, count, radius)
                if include_funda_prices:
                    if not funda.empty:
                        dataset, funda_merge_stats = merge_exact_funda_listings(dataset, funda)
                        funda_matches = funda_merge_stats["matched"]
                        dataset.to_csv(
                            Path("data/processed") / f"{province.lower().replace(' ', '_')}_dataset.csv",
                            index=False,
                        )
                    else:
                        st.warning("Funda leverde geen woningen op; de BAG-dataset blijft wel beschikbaar.")
                build_completed = True
            except Exception as exc:
                st.error(
                    f"Dataset bouwen mislukt ({type(exc).__name__}): "
                    f"{str(exc) or repr(exc)}"
                )
            finally:
                if manual_driver is not None:
                    try:
                        manual_driver.quit()
                    except Exception:
                        pass
                st.session_state.funda_driver = None
            if build_completed:
                if dataset.empty or "oppervlakte_m2" not in dataset.columns:
                    st.warning("De BAG-query leverde geen woningen op. Probeer Dataset bouwen opnieuw.")
                else:
                    st.success("Dataset gebouwd.")
                    st.metric("Aantal records", len(dataset))
                    st.metric("Records met BAG-oppervlakte", int(dataset["oppervlakte_m2"].notna().sum()))
                    estimated_price_count = (
                        int(dataset["huisprijs_schatting"].notna().sum())
                        if "huisprijs_schatting" in dataset.columns
                        else 0
                    )
                    st.metric("Records met huisprijsschatting", estimated_price_count)
                    if include_funda_prices:
                        st.metric("Records met echte Funda-prijs", funda_matches)
                        if not funda.empty:
                            st.subheader("Opgehaalde Funda-informatie")
                            st.dataframe(funda, width="stretch")
                            st.download_button(
                                "Download Funda-resultaten als CSV",
                                funda.to_csv(index=False).encode("utf-8"),
                                file_name="funda_resultaten.csv",
                                mime="text/csv",
                            )
                    st.metric("Records met voorzieningen", int(dataset["afstand_school_km"].notna().sum()) if "afstand_school_km" in dataset.columns else 0)
                    st.metric("Provinciecontrole", int((dataset["provincie"] == province).sum()))
                    if include_dataset_facilities and int(count) > 50:
                        st.warning("Voorzieningen zijn niet toegevoegd: kies maximaal 50 woningen per batch om de kaartservice niet te overbelasten.")
                    st.caption("Het model kan trainen op `huisprijs_schatting`. Dit is een berekende doelvariabele, geen geregistreerde verkoopprijs.")
                    st.caption(
                        "Extra trainingskenmerken: BAG-bouwjaar en leeftijd, aantal verblijfsobjecten, "
                        "postcodegebied, huisnummer, locatiecoordinaten, gebruiksdoel, statussen en "
                        "datakwaliteitsvlaggen. Voorzieningsafstanden worden alleen toegevoegd als je "
                        "de optie inschakelt."
                    )
                    st.json(validation)
                    st.dataframe(dataset.head(10), width="stretch")
                    st.download_button(
                    "Download dataset als CSV",
                    dataset.to_csv(index=False).encode("utf-8"),
                    file_name=f"{province.lower().replace(' ', '_')}_dataset.csv",
                    mime="text/csv",
                )
                    st.download_button(
                    "Download dataset als JSON",
                    dataset.to_json(orient="records", force_ascii=False, indent=2, date_format="iso").encode("utf-8"),
                    file_name=f"{province.lower().replace(' ', '_')}_dataset.json",
                    mime="application/json",
                )

with selected_tab[2]:
    st.subheader("Train en vergelijk modellen")
    uploaded = st.file_uploader("Upload een dataset (CSV of JSON)", type=["csv", "json"])
    if uploaded is not None:
        try:
            df = read_uploaded_dataset(uploaded)
            df = enrich_with_public_sources(df)
            price_columns = [column for column in ["huisprijs", "verkoopprijs", "woz_waarde", "huisprijs_schatting", "waarde_index"] if column in df.columns]
            if price_columns:
                target = st.selectbox("Prijsdoel voor training", price_columns)
                st.caption(f"Training gebruikt {len(df)} records en het doelveld `{target}`.")
                if st.button("Model trainen met upload", key="train_uploaded"):
                    metrics = train_model_from_dataset(df, target)
                    st.success("Model getraind en opgeslagen.")
                    st.json(metrics)
            else:
                st.error("Geen prijsdoel gevonden. Voeg een kolom huisprijs, verkoopprijs, woz_waarde, huisprijs_schatting of waarde_index toe.")
        except Exception as exc:
            st.error(f"Dataset laden of trainen mislukt: {exc}")
    else:
        data_files = [path for path in Path("data/processed").glob("*.csv") if path.stat().st_size > 0]
        if data_files:
            try:
                dataset = pd.read_csv(data_files[0])
                dataset = enrich_with_public_sources(dataset)
            except EmptyDataError:
                dataset = pd.DataFrame()
            if dataset.empty:
                st.info("De opgeslagen dataset is leeg. Bouw eerst een nieuwe dataset.")
                st.stop()
            try:
                target_candidates = [
                    column
                    for column in ["huisprijs", "verkoopprijs", "woz_waarde", "huisprijs_schatting", "waarde_index"]
                    if column in dataset.columns and pd.to_numeric(dataset[column], errors="coerce").notna().any()
                ]
                if not target_candidates:
                    st.info("De dataset bevat nog geen geldige prijsrecords voor training.")
                else:
                    target = st.selectbox("Prijsdoel voor training", target_candidates, key="stored_target")
                    st.caption(f"Training gebruikt {len(dataset)} records en het doelveld `{target}`.")
                    if st.button("Opgeslagen dataset trainen", key="train_stored"):
                        metrics = train_model_from_dataset(dataset, target)
                        st.success("Model getraind en opgeslagen.")
                        st.json(metrics)
            except Exception as exc:
                st.error(f"Modeltraining mislukt: {exc}")
        else:
            st.info("Er is nog geen dataset opgeslagen. Bouw eerst een dataset in de tab 'Dataset verzamelen'.")
