import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

YN_MAP_CAT = {'yes': 1, 'no': 0}
FAMSIZE_MAP = {'GT3': 1, 'LE3': 0}
ONE_HOT_COLS = ['school', 'sex', 'address', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']


def encode_one_hot(df: pd.DataFrame, predicting=False):
    if predicting:
        enc = joblib.load('../models/ONE_HOT_ENCODER')
    else:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(df[ONE_HOT_COLS])
        joblib.dump(enc, '../models/ONE_HOT_ENCODER')

    new_cols = enc.get_feature_names_out(ONE_HOT_COLS).tolist()
    encoded = pd.DataFrame(enc.transform(df[ONE_HOT_COLS]).toarray(), columns=new_cols)

    df = df.join(encoded)
    return df.drop(ONE_HOT_COLS, axis=1).fillna(0)


def preprocess_pipeline(df_master: pd.DataFrame, predicting=False):
    df = df_master.copy()
    df.set_index('StudentID', inplace=True)

    # Transformations des catégories en numéros
    df.replace(YN_MAP_CAT, inplace=True)
    df.replace(FAMSIZE_MAP, inplace=True)

    # One Hot Encoding
    df = encode_one_hot(df, predicting)
    return df


