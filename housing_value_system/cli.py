from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .bag_client import fetch_address_by_postcode
from .data_pipeline import collect_regional_dataset
from .modeling import train_model_from_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Nederlandse woningwaardering CLI")
    parser.add_argument("--postcode", type=str, default=None, help="Bijv. 6371AB")
    parser.add_argument("--huisnummer", type=int, default=None, help="Bijv. 10")
    parser.add_argument("--provincie", type=str, default="Limburg", help="Provincie of Heel Nederland")
    parser.add_argument("--aantal", type=int, default=1000, help="Aantal records om te verzamelen")
    args = parser.parse_args()

    if args.postcode and args.huisnummer is not None:
        address = fetch_address_by_postcode(args.postcode, args.huisnummer)
        print(json.dumps(address, ensure_ascii=False, indent=2))
        return

    dataset, validation = collect_regional_dataset(args.provincie, args.aantal)
    metrics = train_model_from_dataset(dataset)
    print(json.dumps({"validation": validation, "model_metrics": metrics["results"], "best_model": metrics["best_model"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
