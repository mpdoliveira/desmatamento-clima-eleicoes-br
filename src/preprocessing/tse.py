import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.logger import log

log("BEGAN TSE PREPROCESSING", begin=True)

# Data paths
raw_tse_dir = RAW_DATA_DIR / "tse"
if not raw_tse_dir.exists():
    log("Could not find raw data")

processed_tse_dir = PROCESSED_DATA_DIR / "tse"
if not processed_tse_dir.exists():
    processed_tse_dir.mkdir(parents=True)
    log("Creating processed directory")

output_file = processed_tse_dir / "tse.parquet"

# For each raw data
csv_files = sorted(
    p for p in raw_tse_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"
)

# CLEAN AND EXTRACT ELECTED MAYORS

writer = None
processed_count = 0

try:
    for input_file in csv_files:

        # Open raw candidate file
        df = pd.read_csv(
            input_file,
            sep=";",
            encoding="latin1",
            na_values=[
                "",
                "NA",
                "null",
                "#NULO",
                "#NE",
                "NÃO DIVULGÁVEL",
                "-1",
                "-3",
                "-4",
            ],
            low_memory=False,
        )

        df.columns = df.columns.str.strip().str.upper()

        COLUMN_MAP = {
            "ANO_ELEICAO": "election_year",
            "CD_TIPO_ELEICAO": "election_type_code",
            "DT_ELEICAO": "election_date",
            "TP_ABRANGENCIA": "scope_type",
            "SG_UF": "uf",
            "SG_UE": "tse_municipality_code",
            "NM_UE": "municipality_name",
            "CD_CARGO": "office_code",
            "DS_CARGO": "office_name",
            "NR_PARTIDO": "party_number",
            "SG_PARTIDO": "party_acronym",
            "NM_PARTIDO": "party_name",
            "CD_SIT_TOT_TURNO": "turn_result_code",
            "DS_SIT_TOT_TURNO": "turn_result",
        }

        df = df.rename(columns=COLUMN_MAP)

        df = df[
            (df["office_name"].str.strip().str.upper() == "PREFEITO")
            & (df["turn_result"].str.strip().str.upper() == "ELEITO")
        ].copy()

        # Remove unnecessary columns
        df = df.drop(
            columns=[
                c
                for c in df.columns
                if c
                not in [
                    "election_year",
                    "election_date",
                    "municipality_name",
                    "party_acronym",
                    "party_name",
                ]
            ],
            errors="ignore",
        )

        # Sort by election date so supplementary elections override original ones
        df["election_month"] = pd.to_datetime(df["election_date"], format="%d/%m/%Y").dt.month
        df = df.drop(columns=["election_date"])

        # Map party ideological alignment
        PARTY_ALIGNMENT_MAP = {
            # Far-Left
            "PC DO B": "far_left",
            "PSTU": "far_left",
            "PCB": "far_left",
            "PCO": "far_left",
            "UP": "far_left",
            "PSOL": "far_left",
            # Left
            "PT": "left",
            "PCDOB": "left",
            # Center-Left
            "PDT": "center_left",
            "PSB": "center_left",
            "REDE": "center_left",
            "PV": "center_left",
            "PPS": "center_left",
            "PPL": "center_left",
            # Center
            "MDB": "center",
            "PMDB": "center",
            "CIDADANIA": "center",
            "PSDB": "center",
            "PMN": "center",
            "MOBILIZA": "center",
            "PTN": "center",
            "PHS": "center",
            "PROS": "center",
            "SOLIDARIEDADE": "center",
            # Center-Right
            "PSD": "center_right",
            "PODE": "center_right",
            "PODEMOS": "center_right",
            "AVANTE": "center_right",
            "PTDOB": "center_right",
            "PTB": "center_right",
            "PR": "center_right",
            "PRP": "center_right",
            "PTC": "center_right",
            "AGIR": "center_right",
            "PRN": "center_right",
            "PAN": "center_right",
            "PGT": "center_right",
            "PST": "center_right",
            "PMB": "center_right",
            # Right
            "PP": "right",
            "PPB": "right",
            "DEM": "right",
            "PFL": "right",
            "UNIAO": "right",
            "UNIÃO": "right",
            "REPUBLICANOS": "right",
            "PRB": "right",
            "PMR": "right",
            "NOVO": "right",
            "PL": "right",
            "PSC": "right",
            "DC": "right",
            "PSDC": "right",
            "PRD": "right",
            "DEMOCRATA": "right",
            # Far-Right
            "PATRI": "far_right",
            "PATRIOTA": "far_right",
            "PEN": "far_right",
            "PRTB": "far_right",
            "MISSAO": "far_right",
            "MISSÃO": "far_right",
            "PSL": "far_right",
        }

        df["party_acronym"] = df["party_acronym"].str.strip().str.upper()
        df["alignment"] = df["party_acronym"].map(PARTY_ALIGNMENT_MAP).fillna("other")

        df["election_year"] = df["election_year"].astype("int64")

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

    log("FINISHED TSE PREPROCESSING")
