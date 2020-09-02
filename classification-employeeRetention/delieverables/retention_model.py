#retention_model.py
import numpy as np
import pandas as pd

import pickle as pickle
import sklearn
import sys

class EmployeeRetentionModel:

    def __init__(self, model_dict_location):
        with open(model_dict_location, 'rb') as f:
            model_dict = pickle.load(f)
            self.model = model_dict['final_model']
            self.trained_df = model_dict['trained_df']

    def predict_proba(self, X, clean=True, augment=True):
        if clean == True:
            X = self.clean_data(X)
        if augment == True:
            X = self.engineer_features(X)
        return X, self.model.predict_proba(X)

    def clean_data(self, df):
        df = df[df.department != 'temp'].copy()
        df.loc[:, 'department'] = df.department.replace('information_technology', 'IT')
        df.loc[:, 'salary'] = df.salary.replace({'low':0, 'medium':1, 'high':2})
        df.loc[:, 'filed_complaint'] = df.filed_complaint.fillna(0)
        df.loc[:, 'recently_promoted'] = df.recently_promoted.fillna(0)
        df.loc[:, 'department'] = df.department.fillna('Missing')
        df.loc[:, 'last_evaluation_missing'] = df.last_evaluation.isnull().astype(int)
        df.loc[:, 'last_evaluation'] = df.last_evaluation.fillna(0.72)
        return df

    def engineer_features(self, df):
        trained_df = self.trained_df
        df = df.copy()
        df.loc[:,'underperformer'] = (df.last_evaluation < 0.65).astype(int)
        df.loc[:,'overqualified'] = ((df.satisfaction < 0.2) & (df.last_evaluation >0.7)).astype(int)
        df.loc[:,'overachiever'] = ((df.last_evaluation > 0.8) & (df.satisfaction > 0.7)).astype(int)
        df.loc[:,'burnout'] = ((df.avg_monthly_hrs>240) & (df.satisfaction < 0.2)).astype(int)
        df = pd.get_dummies(df, columns = ['department'])
        _, df = trained_df.align(df, join = 'left', axis = 1)
        for col in df.columns:
            df.loc[:, col] = df[col].astype(trained_df[col].dtypes.name)
        return df

def main(data_location, output_location, model_dict_location, clean=True, augment=True):

    df = pd.read_csv(data_location)
    retention_model = EmployeeRetentionModel(model_dict_location)

    df, pred = retention_model.predict_proba(df, clean = clean, augment = augment)
    df['prediction'] = pred[:,1]

    df.to_csv(output_location, index=None)


if __name__ == '__main__':
    main(*sys.argv[1:])
```    
