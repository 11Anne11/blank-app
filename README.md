# Nederlandse Woningwaardering

Een volledig, direct bruikbare Python-applicatie voor Nederlandse woningwaardering met echte BAG-data van PDOK, datasetopbouw, data-quality checks, feature-engineering en modeltraining.

## Snel starten

1. Maak een virtuele omgeving aan:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Installeer dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start de GUI:
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 80

   Open the app on the same network with the computer's IPv4 address, for
   example `http://192.168.1.25`. Windows Firewall must allow inbound TCP
   traffic on port 80. This exposes the app to every device that can reach
   this computer; use a private/trusted network only.
   ```
4. Gebruik de GUI om:
   - een postcode + huisnummer te waarderen;
   - een regio te selecteren en een dataset te verzamelen;
   - een model te trainen en vergelijken.

## Belangrijk

- De primaire bron is de officiële BAG OGC API van PDOK: `https://api.pdok.nl/kadaster/bag/ogc/v2`
- De applicatie bouwt een ML-ready dataset op uit werkelijke BAG-records.
- Herkomst- en quality-controle zijn verwerkt in de data pipeline.
- De app is ontworpen om lokaal te draaien en datasets te persisteren in `data/`.
