import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

TO_SUM_W1 = ['Fedu', 'Medu', 'Mjob_services', 'Fjob_teacher', 'activities', 'address_U', 'famrel', 'internet', 'higher',
             'nursery', 'studytime']
TO_SUM_W2 = ['Dalc', 'Fjob_other', 'Mjob_other', 'Walc', 'absences', 'address_R', 'failures', 'goout', 'guardian_other',
             'health', 'paid', 'romantic', 'schoolsup']


def calculate_complexity(data: pd.DataFrame, predicting=False, fields=None):
    result = []
    fields_to_use = [k for k, v in fields.items() if v]
    for i, row in data.iterrows():
        filtered_sum_w1 = [field for field in TO_SUM_W1 if any(to_filter in field for to_filter in fields_to_use)]
        filtered_sum_w2 = [field for field in TO_SUM_W2 if any(to_filter in field for to_filter in fields_to_use)]
        result.append([i, row[filtered_sum_w1].sum() + row[filtered_sum_w2].sum() * 2])

    result = pd.DataFrame(result)
    result.rename(columns={0: 'StudentID', 1: 'Complexity'}, inplace=True)
    result.set_index('StudentID', inplace=True)

    if predicting:
        enc = joblib.load('models/MIN_MAX_SCALER')
    else:
        enc = MinMaxScaler()
        enc.fit(np.array(result['Complexity']).reshape(-1, 1))
        joblib.dump(enc, 'models/MIN_MAX_SCALER')

    result['Complexity'] = enc.transform(np.array(result['Complexity']).reshape(-1, 1)) * 100

    return result
