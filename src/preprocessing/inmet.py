import json
import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.logger import log

log("BEGAN INMET PREPROCESSING", begin=True)

# Data paths
raw_inmet_dir = RAW_DATA_DIR / "inmet"
if not raw_inmet_dir.exists():
    log("Could not find raw data")

processed_inmet_dir = PROCESSED_DATA_DIR / "inmet"
if not processed_inmet_dir.exists():
    processed_inmet_dir.mkdir(parents=True)
    log("Creating processed directory")

output_file = processed_inmet_dir / "inmet.parquet"

metadata_path = processed_inmet_dir / "metadata.json"
all_metadata = (
    json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata_path.exists()
    else {}
)

# For each raw data
csv_files = sorted(
    p for p in raw_inmet_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"
)

# CLEAN AND GROUP YEAR RAW DATA

writer = None
processed_count = 0

try:
    for input_file in csv_files:

        header_meta = {}
        header_row_idx = None

        date_column = None

        with open(input_file, mode="r", encoding="latin1") as file:

            # Save metadata and rows to skip
            for idx, line in enumerate(file):
                parts = [cell.strip() for cell in line.rstrip("\r\n").split(";")]

                if not parts or parts[0] == "":
                    continue

                if parts[0].upper().strip() in ["DATA", "DATA (YYYY-MM-DD)"]:
                    date_column = parts[0]
                    header_row_idx = idx
                    break

                if len(parts) >= 2:
                    key = parts[0].rstrip(":").strip()
                    header_meta[key] = parts[1]

        # if no header found the file will be ignored
        if header_row_idx is None:
            log(f"Could not find header in {input_file.name}")

        file_stem = input_file.stem
        all_metadata[file_stem] = header_meta

        # Open file from header found
        df = pd.read_csv(
            input_file,
            sep=";",
            decimal=",",
            encoding="latin1",
            skiprows=header_row_idx,
            na_values=["", "NA", "null", "-9999"],
        )

        df.columns = df.columns.str.strip().str.upper()

        COLUMN_MAP = {
            "DATA": "date",
            "DATA (YYYY-MM-DD)": "date",
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "temp_c",
            "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)": "temp_max_c",
            "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)": "temp_min_c",
            "UMIDADE RELATIVA DO AR, HORARIA (%)": "humidity_pct",
            "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)": "humidity_max_pct",
            "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)": "humidity_min_pct",
            "PRECIPITAÇÃO TOTAL, HORÁRIO (MM)": "precip_mm",
            "PRECIPITACAO TOTAL, HORARIO (MM)": "precip_mm",
            "RADIACAO GLOBAL (KJ/M²)": "radiation_kj_m2",
            "RADIAÇÃO GLOBAL (KJ/M²)": "radiation_kj_m2",
        }

        df = df.rename(columns=COLUMN_MAP)

        # Remove unnecessary columns
        df = df.drop(
            columns=[
                "HORA UTC",
                "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (MB)",
                "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (MB)",
                "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (MB)",
                "TEMPERATURA DO PONTO DE ORVALHO (°C)",
                "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)",
                "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)",
                "VENTO, DIREÇÃO HORARIA (GR) (° (GR))",
                "VENTO, RAJADA MAXIMA (M/S)",
                "VENTO, VELOCIDADE HORARIA (M/S)",
                "UNNAMED: 19",
            ],
            errors="ignore",
        )

        # Fill N/A temperature values with average
        df["temp_c"] = df["temp_c"].fillna(
            (df["temp_max_c"] + df["temp_min_c"]) / 2
        )

        # Fill N/A humidity values with average
        df["humidity_pct"] = df["humidity_pct"].fillna(
            (df["humidity_max_pct"] + df["humidity_min_pct"]) / 2
        )

        # Remove max and min temperature and humidity columns used to fill N/A values
        df = df.drop(
            columns=[
                "temp_max_c",
                "temp_min_c",
                "humidity_max_pct",
                "humidity_min_pct",
            ]
        )

        # Group by month
        dt = pd.to_datetime(df["date"])
        df["MES"] = dt.dt.month
        df["ANO"] = dt.dt.year

        df = df.groupby(["MES", "ANO"], as_index=False).agg(
            {
                "temp_c": "mean",  # Average hourly temperature
                "humidity_pct": "mean",  # Average hourly humidity
                "precip_mm": "sum",  # Total monthly precipitation
                "radiation_kj_m2": "sum",  # Total monthly radiation
            }
        )

        df["LONGITUDE"] = pd.to_numeric(
            str(header_meta.get("LONGITUDE", "")).replace(",", "."), errors="coerce"
        )
        df["LONGITUDE"] = df["LONGITUDE"].astype("float64")

        df["LATITUDE"] = pd.to_numeric(
            str(header_meta.get("LATITUDE", "")).replace(",", "."), errors="coerce"
        )
        df["LATITUDE"] = df["LATITUDE"].astype("float64")

        table = pa.Table.from_pandas(df, preserve_index=False)
        
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        else:
            table = table.cast(writer.schema)

        writer.write_table(table=table)
        processed_count += 1

        # Free DataFrames from memory right away
        del df, table
finally:
    log(f"{processed_count} files successfully compiled")

    if writer is not None:
        writer.close()

    # Saves added metadata to json
    metadata_path.write_text(
        json.dumps(all_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log(f"Metadata saved to .json")
    log("FINISHED INMET PREPROCESSING")