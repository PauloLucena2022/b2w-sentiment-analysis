import logging 
import numpy as np
import pandas as pd
from config.config import DATASET_URL, RAW_CSV_PATH

logger = logging.getLogger(__name__)

def load_raw_data(force_download: False):
    # Se já tiver local
    if RAW_CSV_PATH.exists() and force_download:
        logger.info("Carregando dataset do cache local")
        return pd.read_csv(RAW_CSV_PATH, sep = ',', low_memory = False)

    # Baixa o dataset
    logger.info("Baixando dataset de %s ...", DATASET_URL)

    df_raw = pd.read_csv(DATASET_URL, sep = ',', low_memory = False)
    df_raw.to_csv(RAW_CSV_PATH, index = False)

    logger.info("Dataset salvo em cache local (%d linhas, %d colunas).", *df_raw.shape)

    return df_raw
# --------------------------------------------------------------------------------------------
def prepare_base_dataframe(df_raw):
    df = df_raw[['review_title', 'review_text', 'overall_rating']].copy()

    df.dropna(subset = ['review_text'], inplace = True)
    df['overall_rating'] = pd.to_numeric(df['overall_rating'], errors = "coerce")
    df.dropna(subset = ['overall_ratings'], inplace = True)

    df["review_title"] = df["review_title"].fillna("").astype(str).str.strip()
    df["review_text"] = df["review_text"].astype(str).str.strip()

    df['review_text'] = np.where(
        df['review_title'] != '',
        df['review_title'] + ". " + df['review_text'],
        df["review_text"]
    )
    df.drop(columns = ['review_title'], inplace = True)

    logger.info("Base preparada com %d registros válidos.", len(df))
    
    return df