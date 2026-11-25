import json
import pandas as pd

from src.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

raw_inmet_dir = RAW_DATA_DIR / "inmet"
processed_inmet_dir = PROCESSED_DATA_DIR / "inmet"
processed_inmet_dir.mkdir(parents=True, exist_ok=True)

labels_path = processed_inmet_dir / "labels.json"
all_labels = (
    json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
)

csv_files = sorted(raw_inmet_dir.iterdir())

for input_file in csv_files:
    output_file = processed_inmet_dir / input_file.with_suffix(".parquet").name
    if output_file.exists() and input_file.name in all_labels:
        continue

    header_meta = {}
    header_row_idx = None

    with open(input_file, mode="r", encoding="latin1") as file:

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

    if header_row_idx is None:
        continue

    all_labels[input_file.name] = header_meta

    df = pd.read_csv(
        input_file,
        sep=";",
        decimal=",",
        encoding="latin1",
        skiprows=header_row_idx,
        na_values=["", "NA", "null", "-9999"],
    )

    df.head()

    df.to_parquet(output_file, index=False)

labels_path.write_text(
    json.dumps(all_labels, indent=2, ensure_ascii=False), encoding="utf-8"
)
