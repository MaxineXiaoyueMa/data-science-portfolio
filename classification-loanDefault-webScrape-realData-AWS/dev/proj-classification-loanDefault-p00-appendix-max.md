## Data Aggregation:
1. individual file: header = first row, footer = last two rows, columns are consistent
2. 2.6 million observations x 150 features per observation: Cloud for training

## EDA - observations:
1. Sample portfolio - a case study of 300 loans in my portfolio selected by Lending Robot:
  1. Even F grades loans can be fully paid off.
  2. target variable = 'Status', only 'Charged Off' and 'Fully Paid' are considered for training.
  3. 30% default rate wipes out return from all other loans.

2. All loans:
  1. defaults highly correlate with borrower's credit worthiness;
  1. defaults highly correlate with grade given by issuer;
  1. no obvious correlation with

## Data Cleaning:
1. Drop:
  1. Rows:
    1. 'loan_status': keep only 'Fully Paid' and 'Charged Off', as they are the only definite status
    2. 'grade': keep only A-D, per platform new policy, thus user case
    3. 'sub_grade': keep only A-D, per platform new policy, thus user case
    4. 'initial_list_status': keep only 'f', as only interested in fractional loans
  2. Columns:
    1. data leakage: compare with columns non-existent in the open loan data (40)
    1. non-informational: 'id', 'url', 'desc',
    2. nlp: 'emp_title', 'desc', 'title',
    3. zero variance: 'pymnt_plan'(pre dropped in data leagaed step), 'member_id', 'initial_list_status'.
    4. geospatial: 'zip_code', 'addr_state'
    5. redundant:
      1. keep one of ('loan_amnt','funded_amnt','funded_amnt_inv')?
      2. 'purpose', 'title'(?)

2. Structural errors:
  1. cat -> num: 'int_rate', 'revol_util',
  2. cat -> ord: 'grade', 'sub_grade', 'emp_length', 'term',
  3. cat -> date: 'issue_d',  'earliest_cr_line','sec_app_earliest_cr_line',

3. Outliers: **Upon further investigation, no outliers warratns dropping**
  1. Possible outliers based on histogram of numerical features, notice single bar charts with large value on x axis, need further investigation, e.g., 'annual_inc', 'dti', 'revol_bal','total_acc','annual_inc_joint', 'tot_coll_amt','tot_cur_bal', 'open_acc_6m', 'max_bal_bc','total_rev_hi_lim','avg_cur_bal','bc_open_to_buy','bc_util','delinq_amnt','mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'num_il_tl','tax_liens', 'total_bal_ex_mort','total_bc_limit', 'total_il_high_credit_limit', 'all_util'

4. Missing values: num_cols(73)

  1. For simplicity, drop sparse features, these features are mainly secondary applicants credit info, could provide potential information, but limited, especially considering the added dimension and complexity.

  2. Missing NOT at random: due to absence of adverse events (i.e. no delinquency, derogatory) - impute with large number as missing is good, and large number is good 100yr*12mo/yr = 1200): 'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog','mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq',

  3. Missing at Random, median: 'dti', 'revol_util', 'collections_12_mths_ex_med',
  'tot_coll_amt', 'tot_cur_bal',  'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_ty_buy', 'bc_util', 'chargeoff_within_12_mths', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acct', 'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',


  4. Pickle imputing information into dict for testing phase

  5. Compromise:
    Some features have missing values due to more than one reason:
       1. they were not available prior to certain date because the platform has not included them in the loan information.
       2. After they become availabe, missing value reason will depend on the features.

    Therefore, the best practice to deal with these features should be:
       1. Find out the date when these features become available.
       2. Prior to the date when these features become available to loans, flag them as 'NA', and fill them with either mode or mean whichever makes sense to the specific features;
       3. After the date, missing values will be flaged as 'Missing', and filled with extreme, mean, mode, small, or whatever number makes sense to the specific features.

    However, due to the constraint on time, given the amount of training available, I simply flag them as 'Missing' and impute with heuristics and see how our model performs on the biased imputation.

## Feature Engineering:
  1. Domain Knowledge
  2. Heuristics:
    1. Group sparse:
    2. new features:
      1. 'credit_history': 'issue_d' - 'earliest_cr_line' + convert to integer sec_app_earliest_cr_line
      2. debt to income ratio: 'lilti': 'installment' / 'annual_inc'
      3. macro economic indicator:(if performance not good)
  3. Prepare ABT:
    1. dummy
    2. train/test split
    3. drop 'issue_d', 'earliest_cr_line'
    4. save the abt

## Model evaluation:
  1. measuring metrics: f2, recall, roc?
  2. pick models:
  3. logistic, RF, Xgboost, SVM
  4. train time -> AWS
  5. Evaluation on test Data
  6. Forecast on open loan data
    1. vs. 'exp_defulat_rate'

## Model delivery:
1. clean data:
    ```python
    drop_cols_in_open_not_in_master = ['exp_default_rate', 'service_fee_rate', 'accept_d', 'exp_d', 'credit_pull_d', 'review_status_d', 'review_status', 'msa', 'ils_exp_d', 'effective_int_rate', 'disbursement_method', 'mtg_payment', 'housing_payment']

    drop_cols = ['id', 'url', 'emp_title', 'desc', 'title', 'member_id', 'initial_list_status','zip_code', 'addr_state', 'funded_amnt']

    master_cat_cols = ['term', 'home_ownership', 'is_inc_v', 'purpose', 'application_type', 'verified_status_joint']

    cat_to_num_cols = ['bc_open_to_buy','percent_bc_gt_75', 'bc_util', 'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 'num_tl_120dpd_2m', 'mo_sin_old_il_acct', 'annual_inc_joint', 'dti_joint', 'mths_since_rcnt_il', 'il_util', 'revol_bal_joint', 'sec_app_fico_range_low', 'sec_app_fico_range_high', 'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog']

    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    el_map = {'10+ years': 10.0,'2 years': 2.0, '< 1 year': 0.5,'3 years': 3.0, '1 year': 1.0,'5 years': 5.0, '4 years': 4.0, '6 years': 6.0, '7 years': 7.0, '8 years': 8.0, '9 years': 9.0}

    purpose_map = {'credit_card_refinancing':'credit_card', 'home_buying':'house', 'business':'small_business', 'moving_and_relocation':'moving'}

    df = df.drop(clean_dict['cols_in_open_not_in_master'], axis = 1)
    df = df.drop(clean_dict['drop_cols'], axis = 1)
    df.loc[:, 'grade'] = df.grade.map(clean_dict['grade_map'])
    df.loc[:, 'sub_grade'] = df.sub_grade.apply(lambda x: clean_dict['grade_map'][x[0]]*5 + int(x[1]))
    df.loc[:, 'emp_length'] = df.emp_length.apply(lambda x: clean_dict['el_map[x]'] if pd.notnull(x) else 0)
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
    df.loc[:, clean_dict['master_cat_cols']] = df[clean_dict['master_cat_cols']].fillna('Missing')
    df.loc[:, 'verified_status_joint'] = df.verified_status_joint.replace({' ': 'Missing'})
    df = df.drop(clean_dict['num_cols_miss_sparse'], axis = 1)
    for col in clean_dict['num_cols_miss']:
        df.loc[:, col+'_mflag'] = df[col].isnull()
    for col in clean_dict['num_cols_miss_1200']:
        df.loc[:, col] = df[col].fillna(1200)
    for col in clean_dict['num_cols_miss_median']:
        df.loc[:, col] = df[col].fillna(clean_dict['num_cols_miss_median_values'][col])

    ```
Engineer features:
```python
  df.loc[:, 'itlm'] = df.annual_inc/df.installment
  df = pd.get_dummies(df)
  df, _ = df.align(abt_df, join = 'right', axis = 1)

```


## Insights:
  Questions/Conterintuitives require further investigation:
  1. really high income, need to borrow?
  1. high collection amount, not default? ~100K, not default
  3. high delinquent amnt, not indicative
  4. hight tax liens doesn't mean low grade, or indicative of defaults
  5. Term, home_ownerhsip, is_inc_v, purpose, application_type, verified_status_joint are all realted to default rate:
    1. AVOID: longer term
    1. Surprise:
      1. Not verified lower default rate,
      1. Join applicats higher rate, smaller amount

## Lessons learned:
    When data cleaning, use data to make prediction as the standard data format to avoid future manipulation such as:
    1. 'term' is numerical feature in prediction data, but was treated as categorical in training
    2.  'purpose' has different categorical values compared to training data, need conversion

    If training on remote server, all the output file even during traiing should be copied back to local, to avoid analysis inconsistence, or changes made on remote:
    train_df.revol_util.median = 0.45 in old file,
    but data cleaning was changed on remote, thus new value is 45.
