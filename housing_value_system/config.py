from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

PROVINCES = [
    "Heel Nederland",
    "Groningen",
    "Friesland",
    "Drenthe",
    "Overijssel",
    "Flevoland",
    "Gelderland",
    "Utrecht",
    "Noord-Holland",
    "Zuid-Holland",
    "Zeeland",
    "Noord-Brabant",
    "Limburg",
]

PROVINCE_BBOXES = {
    "Heel Nederland": None,
    "Groningen": [5.8, 52.7, 7.3, 53.6],
    "Friesland": [4.7, 52.7, 6.3, 53.6],
    "Drenthe": [6.2, 52.4, 7.3, 53.3],
    "Overijssel": [5.8, 52.1, 7.0, 52.8],
    "Flevoland": [5.0, 52.2, 6.0, 53.0],
    "Gelderland": [4.8, 51.4, 6.8, 52.3],
    "Utrecht": [4.8, 51.8, 5.6, 52.2],
    "Noord-Holland": [4.2, 52.0, 5.2, 53.5],
    "Zuid-Holland": [3.8, 51.5, 5.2, 52.1],
    "Zeeland": [3.2, 51.2, 4.2, 51.8],
    "Noord-Brabant": [4.2, 51.2, 5.9, 51.9],
    "Limburg": [5.6, 50.7, 6.2, 51.5],
}

BAG_BASE_URL = "https://api.pdok.nl/kadaster/bag/ogc/v2"
