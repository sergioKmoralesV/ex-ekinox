import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

TO_SUM_W1 = ['Fedu', 'Medu', 'Mjob_services', 'Fjob_teacher', 'activities', 'address_U', 'famrel', 'internet', 'higher',
             'nursery', 'studytime']
TO_SUM_W2 = ['Dalc', 'Fjob_other', 'Mjob_other', 'Walc', 'absences', 'address_R', 'failures', 'goout', 'guardian_other',
             'health', 'paid', 'romantic', 'schoolsup']


def calculate_complexity(data: pd.DataFrame, predicting=False):
    result = []
    for i, row in data.iterrows():
        result.append([i, row[TO_SUM_W1].sum() + row[TO_SUM_W2].sum() * 2])

    result = pd.DataFrame(result)
    result.rename(columns={0: 'StudentID', 1: 'Complexity'}, inplace=True)
    result.set_index('StudentID', inplace=True)

    if predicting:
        enc = joblib.load('../models/ONE_HOT_ENCODER')
    else:
        enc = MinMaxScaler()
        enc.fit(np.array(result['Complexity']).reshape(-1, 1))
        joblib.dump(enc, '../models/MIN_MAX_SCALER')

    result['Complexity'] = enc.transform(np.array(result['Complexity']).reshape(-1, 1)) * 100

    return result
