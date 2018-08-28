import featuretools as ft
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from featuretools.primitives import AggregationPrimitive, make_agg_primitive

from featuretools.variable_types import (
    Boolean, Datetime,
    DatetimeTimeIndex,
    Discrete,
    Index,
    Numeric,
    Variable,
    Id
)
from datetime import datetime, timedelta

from collections import Counter

start_date = pd.Timestamp("2016-01-01")
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def proc_train_test(is_test=False):
    df = pd.read_csv('../input/application_test.csv').sort_values('SK_ID_CURR').reset_index().drop(columns = ['index']) if is_test else pd.read_csv('../input/application_train.csv').sort_values('SK_ID_CURR').reset_index().drop(columns = ['index'])

    app_types = {}

    # Iterate through the columns and record the Boolean columns
    for col in df:
        # If column is a number with only two values, encode it as a Boolean
        if (df[col].dtype != 'object') and (len(df[col].unique()) <= 2):
            app_types[col] = ft.variable_types.Boolean

    # Record ordinal variables
    app_types['REGION_RATING_CLIENT'] = ft.variable_types.Ordinal
    app_types['REGION_RATING_CLIENT_W_CITY'] = ft.variable_types.Ordinal

    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['car_to_birth_ratio'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['children_ratio'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['car_to_employ_ratio'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['credit_to_annuity_ratio'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['credit_to_goods_ratio'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['credit_to_income_ratio'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['days_employed_percentage'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['income_credit_percentage'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['income_per_child'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['income_per_person'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['payment_rate'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['phone_to_birth_ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['phone_to_employ_ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_SOURCES_WEIGHTED_SUM'] = df['EXT_SOURCE_1'] * 2 + df['EXT_SOURCE_2'] * 3 + df['EXT_SOURCE_3'] * 4
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())

    # Adding new features
    df['CNT_NON_CHILD'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
    df['CHILD_TO_NON_CHILD_RATIO'] = df['CNT_CHILDREN'] / df['CNT_NON_CHILD']
    df['INCOME_PER_NON_CHILD'] = df['AMT_INCOME_TOTAL'] / df['CNT_NON_CHILD']
    df['CREDIT_PER_PERSON'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
    df['CREDIT_PER_CHILD'] = df['AMT_CREDIT'] / df['CNT_CHILDREN']
    df['CREDIT_PER_NON_CHILD'] = df['AMT_CREDIT'] / df['CNT_NON_CHILD']
    for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
        df['EXTERNAL_SOURCES_{}'.format(function_name)] = eval('np.{}'.format(function_name))(df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    df['SHORT_EMPLOYMENT'] = (df['DAYS_EMPLOYED'] < -2000).astype(int)
    df['YOUNG_AGE'] = (df['DAYS_BIRTH'] < -14000).astype(int)

    gc.collect()
    return df, app_types

def proc_bureau():
    bureau_df = pd.read_csv('../input/bureau.csv').sort_values(['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index().drop(columns = ['index'])

    for col in ['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_UPDATE']:
        bureau_df[col] = pd.to_timedelta(bureau_df[col], 'D')

    # Create the date columns
    bureau_df['bureau_credit_application_date'] = start_date + bureau_df['DAYS_CREDIT']
    bureau_df['bureau_credit_end_date'] = start_date + bureau_df['DAYS_CREDIT_ENDDATE']
    bureau_df['bureau_credit_close_date'] = start_date + bureau_df['DAYS_ENDDATE_FACT']
    bureau_df['bureau_credit_update_date'] = start_date + bureau_df['DAYS_CREDIT_UPDATE']

    # Drop the time offset columns
    bureau_df = bureau_df.drop(columns = ['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_UPDATE'])

    gc.collect()
    return bureau_df

def proc_prev():
    prev_df = pd.read_csv('../input/previous_application.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index().drop(columns = ['index'])

    prev_types= {'NFLAG_LAST_APPL_IN_DAY': ft.variable_types.Boolean,
                 'NFLAG_INSURED_ON_APPROVAL': ft.variable_types.Boolean}

    prev_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

    # Convert to timedeltas in days
    for col in ['DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
        prev_df[col] = pd.to_timedelta(prev_df[col], 'D')

    # Make date columns
    prev_df['previous_decision_date'] = start_date + prev_df['DAYS_DECISION']
    prev_df['previous_drawing_date'] = start_date + prev_df['DAYS_FIRST_DRAWING']
    prev_df['previous_first_due_date'] = start_date + prev_df['DAYS_FIRST_DUE']
    prev_df['previous_last_duefirst_date'] = start_date + prev_df['DAYS_LAST_DUE_1ST_VERSION']
    prev_df['previous_last_due_date'] = start_date + prev_df['DAYS_LAST_DUE']
    prev_df['previous_termination_date'] = start_date + prev_df['DAYS_TERMINATION']

    # Drop the time offset columns
    prev_df = prev_df.drop(columns = ['DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION'])

    # Add feature: value ask / value received percentage
    prev_df['APP_CREDIT_PERC'] = prev_df['AMT_APPLICATION'] / prev_df['AMT_CREDIT']

    gc.collect()
    return prev_df, prev_types

def proc_bureau_balance():
    bureau_balance_df = pd.read_csv('../input/bureau_balance.csv').sort_values('SK_ID_BUREAU').reset_index().drop(columns = ['index'])

    # Convert to timedelta
    bureau_balance_df['MONTHS_BALANCE'] = pd.to_timedelta(bureau_balance_df['MONTHS_BALANCE'], 'M')

    # Make a date column
    bureau_balance_df['bureau_balance_date'] = start_date + bureau_balance_df['MONTHS_BALANCE']
    bureau_balance_df = bureau_balance_df.drop(columns = ['MONTHS_BALANCE'])

    gc.collect()
    return bureau_balance_df

def proc_cash():
    cash_df = pd.read_csv('../input/POS_CASH_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index().drop(columns = ['index'])

    # Convert to timedelta objects
    cash_df['MONTHS_BALANCE'] = pd.to_timedelta(cash_df['MONTHS_BALANCE'], 'M')
    # Make a date column
    cash_df['cash_balance_date'] = start_date + cash_df['MONTHS_BALANCE']
    cash_df = cash_df.drop(columns = ['MONTHS_BALANCE'])

    gc.collect()
    return cash_df

def proc_installments():
    install_df = pd.read_csv('../input/installments_payments.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index().drop(columns = ['index'])

    # Percentage and difference paid in each installment (amount paid and installment value)
    install_df['PAYMENT_PERC'] = install_df['AMT_PAYMENT'] / install_df['AMT_INSTALMENT']
    install_df['PAYMENT_DIFF'] = install_df['AMT_INSTALMENT'] - install_df['AMT_PAYMENT']
    # Days past due and days before due
    install_df['DPD'] = install_df['DAYS_ENTRY_PAYMENT'] - install_df['DAYS_INSTALMENT']
    install_df['DBD'] = install_df['DAYS_INSTALMENT'] - install_df['DAYS_ENTRY_PAYMENT']
    install_df['DPD'] = install_df['DPD'].apply(lambda x: x if x > 0 else 0)
    install_df['DBD'] = install_df['DBD'].apply(lambda x: x if x > 0 else 0)

    # Conver to time delta object
    install_df['DAYS_INSTALMENT'] = pd.to_timedelta(install_df['DAYS_INSTALMENT'], 'D')
    install_df['DAYS_ENTRY_PAYMENT'] = pd.to_timedelta(install_df['DAYS_ENTRY_PAYMENT'], 'D')

    # Create time column and drop
    install_df['installments_due_date'] = start_date + install_df['DAYS_INSTALMENT']
    install_df = install_df.drop(columns = ['DAYS_INSTALMENT'])

    install_df['installments_paid_date'] = start_date + install_df['DAYS_ENTRY_PAYMENT']
    install_df = install_df.drop(columns = ['DAYS_ENTRY_PAYMENT'])

    gc.collect()
    return install_df

def proc_credit():
    credit_df = pd.read_csv('../input/credit_card_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index().drop(columns = ['index'])

    # Convert to timedelta objects
    credit_df['MONTHS_BALANCE'] = pd.to_timedelta(credit_df['MONTHS_BALANCE'], 'M')

    # Make a date column
    credit_df['credit_balance_date'] = start_date + credit_df['MONTHS_BALANCE']
    credit_df = credit_df.drop(columns = ['MONTHS_BALANCE'])

    gc.collect()
    return credit_df

def build_es():
    es = ft.EntitySet(id='clients')

    with timer('Processing train...\n'):
        train_df, train_types = proc_train_test(is_test=False)
        es = es.entity_from_dataframe(entity_id='train',
                                      dataframe=train_df,
                                      index='SK_ID_CURR',
                                      variable_types=train_types)
        del train_df
        gc.collect()

    with timer('Processing test...\n'):
        test_df, test_types = proc_train_test(is_test=True)
        es = es.entity_from_dataframe(entity_id='test',
                                      dataframe=test_df,
                                      index='SK_ID_CURR',
                                      variable_types=test_types)
        del test_df
        gc.collect()

    with timer('Processing bureau...\n'):
        bureau_df = proc_bureau()
        es = es.entity_from_dataframe(entity_id='bureau',
                                      dataframe=bureau_df,
                                      index='SK_ID_BUREAU',
                                      time_index='bureau_credit_application_date')
        del bureau_df
        gc.collect()

        train_bureau_rel = ft.Relationship(es['train']['SK_ID_CURR'],
                                          es['bureau']['SK_ID_CURR'])
        test_bureau_rel = ft.Relationship(es['test']['SK_ID_CURR'],
                                          es['bureau']['SK_ID_CURR'])
        es = es.add_relationship(train_bureau_rel)
        es = es.add_relationship(test_bureau_rel)

    with timer('Processing prev...\n'):
        prev_df, prev_types = proc_prev()
        es = es.entity_from_dataframe(entity_id='prev',
                                      dataframe=prev_df,
                                      index='SK_ID_PREV',
                                      time_index='previous_decision_date',
                                      variable_types=prev_types)
        del prev_df
        gc.collect()
        train_prev_rel = ft.Relationship(es['train']['SK_ID_CURR'],
                                        es['prev']['SK_ID_CURR'])
        test_prev_rel = ft.Relationship(es['test']['SK_ID_CURR'],
                                        es['prev']['SK_ID_CURR'])
        es = es.add_relationship(train_prev_rel)
        es = es.add_relationship(test_prev_rel)

    with timer('Processing bureau_balance...\n'):
        bureau_balance_df = proc_bureau_balance()
        es = es.entity_from_dataframe(entity_id='bureau_balance',
                                      dataframe=bureau_balance_df,
                                      make_index=True,
                                      index='bb_index',
                                      time_index='bureau_balance_date')
        del bureau_balance_df
        gc.collect()
        bureau_bb_rel = ft.Relationship(es['bureau']['SK_ID_BUREAU'],
                                        es['bureau_balance']['SK_ID_BUREAU'])
        es = es.add_relationship(bureau_bb_rel)

    with timer('Processing cash...\n'):
        cash_df = proc_cash()
        es = es.entity_from_dataframe(entity_id='cash',
                                      dataframe=cash_df,
                                      make_index=True,
                                      index='cash_index',
                                      time_index='cash_balance_date')
        del cash_df
        gc.collect()
        prev_cash_rel = ft.Relationship(es['prev']['SK_ID_PREV'],
                                        es['cash']['SK_ID_PREV'])
        es = es.add_relationship(prev_cash_rel)

    with timer('Processing installments...\n'):
        install_df = proc_installments()
        es = es.entity_from_dataframe(entity_id='install',
                                      dataframe=install_df,
                                      make_index=True,
                                      index='install_index',
                                      time_index='installments_paid_date')
        del install_df
        gc.collect()
        prev_install_rel = ft.Relationship(es['prev']['SK_ID_PREV'],
                                           es['install']['SK_ID_PREV'])
        es = es.add_relationship(prev_install_rel)

    with timer('Processing credit...\n'):
        credit_df = proc_credit()
        es = es.entity_from_dataframe(entity_id='credit',
                                      dataframe=credit_df,
                                      make_index=True,
                                      index='credit_index',
                                      time_index='credit_balance_date')
        del credit_df
        gc.collect()
        prev_credit_rel = ft.Relationship(es['prev']['SK_ID_PREV'],
                                          es['credit']['SK_ID_PREV'])
        es = es.add_relationship(prev_credit_rel)
    return es

es = build_es()

def normalized_mode_count(x):
    """
    Return the fraction of total observations that
    are the most common observation. For example,
    in an array of ['A', 'A', 'A', 'B', 'B'], the
    function will return 0.6."""

    if x.mode().shape[0] == 0:
        return np.nan

    # Count occurence of each value
    counts = dict(Counter(x.values))
    # Find the mode
    mode = x.mode().iloc[0]
    # Divide the occurences of mode by the total occurrences
    return counts[mode] / np.sum(list(counts.values()))


NormalizedModeCount = make_agg_primitive(function = normalized_mode_count,
                                         input_types = [Discrete],
                                         return_type = Numeric)

def longest_repetition(x):
    """
    Returns the item with most consecutive occurrences in `x`.
    If there are multiple items with the same number of conseqcutive occurrences,
    it will return the first one. If `x` is empty, returns None.
    """

    x = x.dropna()

    if x.shape[0] < 1:
        return None

    # Set the longest element
    longest_element = current_element = None
    longest_repeats = current_repeats = 0

    # Iterate through the iterable
    for element in x:
        if current_element == element:
            current_repeats += 1
        else:
            current_element = element
            current_repeats = 1
        if current_repeats > longest_repeats:
            longest_repeats = current_repeats
            longest_element = current_element

    return longest_element

LongestSeq = make_agg_primitive(function = longest_repetition,
                                     input_types = [Discrete],
                                     return_type = Discrete)

# Late Payment seed feature
late_payment = ft.Feature(es['install']['installments_due_date']) < ft.Feature(es['install']['installments_paid_date'])

# Rename the feature
late_payment = late_payment.rename("late_payment")

# Create a feed representing whether the loan is past due
past_due = ft.Feature(es['bureau_balance']['STATUS']).isin(['1', '2', '3', '4', '5'])
past_due = past_due.rename("past_due")

# Assign interesting features
es['prev']['NAME_CONTRACT_STATUS'].interesting_values = ['Approved', 'Refused', 'Canceled']

# Run and create the features
feature_matrix, feature_names = ft.dfs(entityset = es, target_entity = 'train',
                                       agg_primitives = ['mean', 'max', 'min', 'trend', 'mode', 'count',
                                                         'sum', 'percent_true', NormalizedModeCount, LongestSeq],
                                       trans_primitives = ['diff', 'cum_sum', 'cum_mean', 'percentile'],
                                       where_primitives = ['mean', 'sum'],
                                       seed_features = [late_payment, past_due],
                                       max_depth = 2, features_only = False, verbose = True,
                                       chunk_size = len(es['train'].df),
                                       ignore_entities = ['test'])

# Run and create the features
feature_matrix_test, feature_names_test = ft.dfs(entityset = es, target_entity = 'test',
                                                agg_primitives = ['mean', 'max', 'min', 'trend', 'mode', 'count',
                                                                    'sum', 'percent_true', NormalizedModeCount, LongestSeq],
                                                trans_primitives = ['diff', 'cum_sum', 'cum_mean', 'percentile'],
                                                where_primitives = ['mean', 'sum'],
                                                seed_features = [late_payment, past_due],
                                                max_depth = 2, features_only = False, verbose = True,
                                                chunk_size = len(es['test'].df),
                                                ignore_entities = ['train'])

print('Final training shape: ', feature_matrix.shape)
print('Final testing shape: ', feature_matrix_test.shape)

# Save the feature matrix to a csv
feature_matrix.to_csv('feature_matrix.csv')
feature_matrix_test.to_csv('feature_matrix_test.csv')
