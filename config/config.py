from pathlib import Path 

# Diretórios base
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

# Garantindo a existência das pastas
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    directory.mkdir(parents = True, exist_ok = True)

# Fonte dos dados
DATASET_URL = ("https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv")
RAW_CSV_PATH = RAW_DATA_DIR / "B2W-Reviews01.csv"

# Reprodutibilidade e splits
RANDOM_STATE = 42
TEST_VAL_SIZE = 0.30
TEST_FROM_TEMP_SIZE = 0.50
MIN_TEXT_LENGTH = 3

# Mapeamento de Classes
LABEL_MAP = {
    "Negativo" : 0,
    "Neutro" : 1,
    "Positivo" : 2
}
CLASS_NAMES = list(LABEL_MAP.keys())

# Hiperparâmetros
BASELINE_PARAM_GRID = {"clf__C": [0.1, 1.0, 10.0]}
CV_FOLDS = 3
MAX_ITER = 2000