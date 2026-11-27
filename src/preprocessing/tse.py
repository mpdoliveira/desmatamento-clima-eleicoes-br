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
            # Far-Left (0.00 - 1.50 | revolucionário / socialista programático)
            "PSTU": "far_left",
            "PCO": "far_left",
            "PCB": "far_left",
            "PSOL": "far_left",
            "UP": "far_left",             # Nota 2022: 1.63 (limiar far_left/left)

            # Left (1.51 - 3.00)
            "PCDOB": "left",
            "PC DO B": "left",            # Corrigido de far_left para left
            "PT": "left",

            # Center-Left (3.01 - 4.49)
            "PDT": "center_left",
            "PSB": "center_left",
            "PPL": "center_left",         # Origem nacionalista-esquerda (MR-8); incorporado ao PCdoB em 2019

            # Center (4.50 - 5.50 + partidos catch-all / fisiológicos históricos)
            "REDE": "center",             # 4.77 no survey (center_left em 2022+ na federação PSOL-REDE)
            "CIDADANIA": "center",
            "PPS": "center",              # Corrigido de center_left para center (mesma legenda 23)
            "PV": "center",               # 5.29 no survey (center_left em 2022+ na federação PT-PCdoB-PV)
            "MDB": "center",              # Catch-all (Centro em Zucco & Power 2024; 7.01 no pós-2016)
            "PMDB": "center",

            # Center-Right (5.51 - 7.00 + Centrão / transição centro-direita)
            "PTB": "center_right",
            "AVANTE": "center_right",
            "PTDOB": "center_right",
            "PT DO B": "center_right",
            "SOLIDARIEDADE": "center_right",  # Corrigido de center para center_right (6.50)
            "SD": "center_right",
            "SDD": "center_right",
            "MOBILIZA": "center_right",   # Corrigido de center para center_right (6.88)
            "PMN": "center_right",
            "DEMOCRATA": "center_right",  # Corrigido de right para center_right (mesma legenda 35 do PMB)
            "PMB": "center_right",        # 6.90 no survey
            "PHS": "center_right",        # Corrigido de center para center_right (6.96)
            "PSD": "center_right",        # 7.09 no survey; Centro em Zucco & Power (2024)
            "PSDB": "center_right",       # Corrigido de center para center_right (7.11; Centro 2000-2012 -> Direita)
            "PODE": "center_right",       # 7.24 no survey; perfil Centrão após absorver PHS e PSC
            "PODEMOS": "center_right",
            "PTN": "center_right",        # Corrigido de center para center_right (predecessor do Podemos)
            "PAN": "center_right",        # Extinto em 2006 (incorporado ao PTB)
            "PGT": "center_right",        # Extinto em 2003
            "PST": "center_right",        # Extinto em 2003

            # Right (7.01 - 8.50)
            "PRD": "right",               # Fusão PTB (6.10) + Patriota (8.55) em 2023 (média estimada: 7.33)
            "PRTB": "right",              # Corrigido de far_right para right (7.45 no survey)
            "PROS": "right",              # Corrigido de center para right (7.47 no survey)
            "PRP": "right",               # Corrigido de center_right para right (7.59 no survey)
            "REPUBLICANOS": "right",
            "PRB": "right",
            "PMR": "right",
            "PL": "right",                # 7.78 como PR (perfil bolsonarista em 2022+)
            "PR": "right",                # Corrigido de center_right para right (mesma legenda 22)
            "AGIR": "right",              # Corrigido de center_right para right (7.86 como PTC)
            "PTC": "right",
            "PRN": "right",
            "DC": "right",
            "PSDC": "right",
            "NOVO": "right",
            "PP": "right",
            "PPB": "right",
            "PSC": "right",
            "UNIAO": "right",             # Fusão DEM (8.57) + PSL (8.11) em 2022 (média estimada: 8.34)
            "UNIÃO": "right",
            "DEM": "right",               # 8.57 em 2018, mas Direita conservadora em Zucco & Power (2024)
            "PFL": "right",

            # Far-Right (8.51 - 10.00 | direita radical / nacionalista-autoritária)
            "PATRI": "far_right",
            "PATRIOTA": "far_right",
            "PEN": "far_right",
            "PSL": "far_right",           # 8.11 (moda 10) em 2018; usar regra temporal para 2000-2016
            "PRONA": "far_right",         # Extinto em 2006 (fundiu-se ao PL)
            "MISSAO": "far_right",        # Registrado em nov/2025 (apenas ciclos 2026+; entre right e far_right)
            "MISSÃO": "far_right",
        }

        PARTY_SCORE_MAP = {
            "PSTU": 0.51, "PCO": 0.61, "PCB": 0.91, "PSOL": 1.28, "UP": 1.63,
            "PCDOB": 1.92, "PC DO B": 1.92, "PT": 2.97, "PDT": 3.92, "PSB": 4.05,
            "REDE": 4.77, "PPS": 4.92, "CIDADANIA": 4.92, "PV": 5.29,
            "PTB": 6.10, "AVANTE": 6.32, "PTDOB": 6.32, "PT DO B": 6.32,
            "SOLIDARIEDADE": 6.50, "SD": 6.50, "SDD": 6.50,
            "PMN": 6.88, "MOBILIZA": 6.88, "PMB": 6.90, "DEMOCRATA": 6.90, "PHS": 6.96,
            "MDB": 7.01, "PMDB": 7.01, "PSD": 7.09, "PSDB": 7.11,
            "PODE": 7.24, "PODEMOS": 7.24, "PTN": 7.24, "PRD": 7.33,
            "PRTB": 7.45, "PROS": 7.47, "PRP": 7.59,
            "REPUBLICANOS": 7.78, "PRB": 7.78, "PMR": 7.78, "PR": 7.78, "PL": 7.78,
            "PTC": 7.86, "AGIR": 7.86, "PRN": 7.86, "DC": 8.11, "PSDC": 8.11,
            "PSL": 8.11, "NOVO": 8.13, "PP": 8.20, "PPB": 8.20, "PSC": 8.33,
            "UNIAO": 8.34, "UNIÃO": 8.34, "PATRIOTA": 8.55, "PATRI": 8.55, "PEN": 8.55,
            "DEM": 8.57, "PFL": 8.57,
        }
        
        df["party_acronym"] = df["party_acronym"].str.strip().str.upper()
        df["alignment"] = df["party_acronym"].map(PARTY_ALIGNMENT_MAP).fillna("other")
        df["aligment_score"] = df["party_acronym"].map(PARTY_SCORE_MAP)

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
