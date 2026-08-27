import io
import zipfile

import pytest

from gridshift.ingest.dwd import parse_cdc_zip, parse_mosmix_kmz


def _zip_bytes(filename: str, content: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(filename, content)
    return buffer.getvalue()


def test_parse_cdc_zip() -> None:
    content = "STATIONS_ID;MESS_DATUM;QN_9;TT_TU;RF_TU;eor\n00433;2025010100;3;4.2;88;eor\n"
    archive = _zip_bytes("produkt_tu_stunde_20250101_20250101_00433.txt", content)

    frame = parse_cdc_zip(archive, "air_temperature", "00433")

    assert frame.loc[0, "station_id"] == "00433"
    assert frame.loc[0, "value"] == pytest.approx(4.2)
    assert str(frame["timestamp_utc"].dt.tz) == "UTC"


def test_parse_mosmix_station_kmz() -> None:
    kml = """<?xml version="1.0" encoding="UTF-8"?>
    <kml:kml xmlns:kml="http://www.opengis.net/kml/2.2"
      xmlns:dwd="https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd">
      <kml:Document>
        <dwd:ForecastTimeSteps>
          <dwd:TimeStep>2025-01-01T00:00:00Z</dwd:TimeStep>
          <dwd:TimeStep>2025-01-01T01:00:00Z</dwd:TimeStep>
        </dwd:ForecastTimeSteps>
        <kml:Placemark><kml:ExtendedData>
          <dwd:Forecast dwd:elementName="TTT"><dwd:value>273.15 274.15</dwd:value></dwd:Forecast>
          <dwd:Forecast dwd:elementName="FF"><dwd:value>4.0 5.0</dwd:value></dwd:Forecast>
        </kml:ExtendedData></kml:Placemark>
      </kml:Document>
    </kml:kml>"""
    archive = _zip_bytes("MOSMIX_L_10382.kml", kml)

    frame = parse_mosmix_kmz(archive, "10382")

    assert len(frame) == 2
    assert frame.loc[0, "temperature_c"] == pytest.approx(0.0)
    assert frame.loc[1, "wind_speed_ms"] == pytest.approx(5.0)
