import json
import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Data paths
raw_inmet_dir = RAW_DATA_DIR / "inmet"
processed_inmet_dir = PROCESSED_DATA_DIR / "inmet"
processed_inmet_dir.mkdir(parents=True, exist_ok=True)
output_file = processed_inmet_dir / "inmet.parquet"

# Metadata
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
try:
    for input_file in csv_files:

        header_meta = {}
        header_row_idx = None

        with open(input_file, mode="r", encoding="latin1") as file:

            # Save metadata and rows to skip
            for idx, line in enumerate(file):
                parts = [cell.strip() for cell in line.rstrip("\r\n").split(";")]

                if not parts or parts[0] == "":
                    continue

                if parts[0].upper().strip() == "DATA":
                    header_row_idx = idx
                    break

                if len(parts) >= 2:
                    key = parts[0].rstrip(":").strip()
                    header_meta[key] = parts[1]

        # if no header found the file will be ignored
        if header_row_idx is None:
            continue

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

        # Remove unnecessary columns
        df = df.drop(
            columns=[
                "HORA UTC",
                "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)",
                "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)",
                "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)",
                "TEMPERATURA DO PONTO DE ORVALHO (°C)",
                "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)",
                "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)",
                "VENTO, DIREÇÃO HORARIA (gr) (° (gr))",
                "VENTO, RAJADA MAXIMA (m/s)",
                "VENTO, VELOCIDADE HORARIA (m/s)",
                "Unnamed: 19",
            ],
            errors="ignore",
        )

        # Fill N/A temperature values with average
        df["TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"] = df[
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"
        ].fillna(
            (
                df["TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)"]
                + df["TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)"]
            )
            / 2
        )

        # Fill N/A humidity values with average
        df["UMIDADE RELATIVA DO AR, HORARIA (%)"] = df[
            "UMIDADE RELATIVA DO AR, HORARIA (%)"
        ].fillna(
            (
                df["UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)"]
                + df["UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)"]
            )
            / 2
        )

        # Remove max and min temperature and humidity columns used to fill N/A values
        df = df.drop(
            columns=[
                "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)",
                "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)",
                "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)",
                "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)",
            ]
        )

        # Group by month
        dt = pd.to_datetime(df["Data"])
        df["MES"] = dt.dt.month
        df["ANO"] = dt.dt.year

        df = df.groupby(["MES", "ANO"], as_index=False).agg(
            {
                "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "mean",  # Average hourly temperature
                "UMIDADE RELATIVA DO AR, HORARIA (%)": "mean",  # Average hourly humidity
                "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "sum",  # Total monthly precipitation
                "RADIACAO GLOBAL (Kj/m²)": "sum",  # Total monthly radiation
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

        # Free DataFrames from memory right away
        del df, table
finally:
    if writer is not None:
        writer.close()

# Saves added metadata to json
metadata_path.write_text(
    json.dumps(all_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
)