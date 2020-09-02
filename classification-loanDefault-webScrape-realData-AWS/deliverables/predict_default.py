import numpy as np
import pandas as pd
import pickle as pickle
import sklearn
import sys

class LoanDefaultModel:
    """A model that predicts loan default probability for loans on LendingClub.com"""

    def __init__(self, model_dict_location):
        with open(model_dict_location, 'rb') as f:
            model_dict = pickle.load(f)
            self.model = model_dict['model']
            self.clean_dict = model_dict['clean_dict']
            self.engineer_dict = model_dict['engineer_dict']
            self.threshold_dict = model_dict['threshold_dict']

    def predict_proba(self, X, clean=True, augment = True):
        X_orig = X.copy()
        if clean:
            X = self.clean_data(X)
        if augment:
            X = self.engineer_features(X)
        return X_orig, X, self.model.predict_proba(X)[:,1]

    def clean_data(self, df):
        clean_dict = self.clean_dict

        df = df.drop(clean_dict['drop_cols_in_open_not_in_master'], axis = 1)
        df = df.drop(clean_dict['drop_cols'], axis = 1)

        df.loc[:, 'grade'] = df.grade.map(clean_dict['grade_map'])
        df.loc[:, 'sub_grade'] = df.sub_grade.apply(lambda x: clean_dict['grade_map'][x[0]]*5 + int(x[1]))
        df.loc[:, 'emp_length'] = df.emp_length.apply(lambda x: clean_dict['el_map'][x] if pd.notnull(x) else 0)
        df.loc[:, 'sec_app_earliest_cr_line'] = pd.to_datetime(df.sec_app_earliest_cr_line, errors='coerce')
        df.loc[:, 'earliest_cr_line'] = pd.to_datetime(df.earliest_cr_line, errors = 'coerce')
        df.loc[:, 'list_d'] = pd.to_datetime(df.list_d, errors = 'coerce')
        df.loc[:, 'home_ownership'] = df.home_ownership.replace(['NONE', 'OTHER'], 'ANY')
        df.loc[:, 'purpose'] = df.purpose.apply(lambda x: x.replace(" ", "_").lower())
        try:
            df.loc[:, 'purpose'] = df.purpose.replace(clean_dict['purpose_map'])
        except:
            print("Error with categorical feature purpose:", sys.exc_info()[0])
            raise
        df.loc[:, 'term'] = df.term.replace({36: ' 36 months', 60: ' 60 months'})
        df.loc[:, 'application_type'] = df.application_type.replace({'INDIVIDUAL':'Individual', 'JOINT':'Joint App'})

        df.loc[:, clean_dict['cat_to_num_cols']] = df[clean_dict['cat_to_num_cols']].apply(lambda s: pd.to_numeric(s, errors = 'coerce'))
        df.loc[:, clean_dict['cat_cols']] = df[clean_dict['cat_cols']].fillna('Missing')
        df.loc[:, 'verified_status_joint'] = df.verified_status_joint.replace({' ': 'Missing'})
        df = df.drop(clean_dict['num_cols_miss_sparse'], axis = 1)
        for col in clean_dict['num_cols_miss']:
            df.loc[:, col+'_mflag'] = df[col].isnull()
        for col in clean_dict['num_cols_miss_1200']:
            df.loc[:, col] = df[col].fillna(1200)
        for col in clean_dict['num_cols_miss_median']:
            df.loc[:, col] = df[col].fillna(clean_dict['num_cols_miss_median_values'][col])

        return df


    def engineer_features(self, df):
        engineer_dict = self.engineer_dict

        df.loc[:, 'credit_history'] = ((df.list_d - df.earliest_cr_line)/ np.timedelta64(1, 'D')).astype(int)
        df = df.drop(['earliest_cr_line','list_d'], axis = 1)
        df.loc[:, 'itlm'] = df.annual_inc/df.installment
        df = pd.get_dummies(df)
        df, _ = df.align(engineer_dict['abt_df'], join = 'right', axis = 1, fill_value = 0)

        return df

def main(data_location, model_location, output_location, clean = True, augment = True):

    df = pd.read_csv(data_location)
    default_model = LoanDefaultModel(model_location)

    orig_df, df, pred = default_model.predict_proba(df, clean = clean, augment = augment)
    orig_df.loc[:, 'predict_proba'] = pred
    orig_df.loc[:, 'gr_than_thres_05'] = (orig_df.predict_proba > default_model.threshold_dict['thres_05'])
    orig_df.loc[:, 'gr_than_thres_08'] = (orig_df.predict_proba > default_model.threshold_dict['thres_08'])
    orig_df.loc[:, 'gr_than_thres_10'] = (orig_df.predict_proba > default_model.threshold_dict['thres_10'])
    
    orig_df[['id','sub_grade', 'int_rate','term','loan_amnt','exp_default_rate', 'predict_proba', 
         'gr_than_thres_05', 'gr_than_thres_08', 'gr_than_thres_10']].to_csv(output_location, index = None)

    return

if __name__ == '__main__':
    main(*sys.argv[1:])
