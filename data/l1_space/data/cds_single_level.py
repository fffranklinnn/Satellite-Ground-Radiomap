"""
ERA5 single-level downloader for communication attenuation features.

Variables:
- 2m_temperature (t2m)
- surface_pressure (sp)
- total_column_water_vapour (tcwv)
- total_precipitation (tp)
- total_column_cloud_liquid_water (tclw)
"""

import cdsapi


def main() -> None:
    client = cdsapi.Client()
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "2m_temperature",
            "surface_pressure",
            "total_column_water_vapour",
            "total_precipitation",
            "total_column_cloud_liquid_water",
        ],
        "year": ["2025"],
        "month": ["01"],
        "day": ["01"],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "zip",
    }
    client.retrieve(dataset, request).download()


if __name__ == "__main__":
    main()
