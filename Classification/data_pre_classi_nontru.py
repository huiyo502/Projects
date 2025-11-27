import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
# from imblearn.over_sampling import SMOTE 
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
import pdb
import math
from typing import Tuple, List

np.random.seed(42)


def apply_categorical_mapping(df):
    """
    Applies categorical to numerical mapping for common features.
    """
    df_temp = df.copy()
    
    # Mapping definitions
    MAPPINGS = {
        'CAVI': {'Y': 1, 'N': 0},
        'SMOKE': {'Y': 1, 'N': 0},
        'Arm': {'2EHRZ/4HR': 0, '2EMRZ/2MR': 1, '2MHRZ/2MHR': 2},
        'SEX': {'F': 0, 'M': 1},
        # Combining less common RACE categories into 'OTHER' (4)
        'RACE': {'ASIAN': 0, 'BLACK': 1, "MIXED RACE OR COLOURED": 3, "NOT REPORTED": 4, "OTHER": 4}
    }

    # Apply mappings and safe type casting
    for col, mapping in MAPPINGS.items():
        if col in df_temp.columns:
            df_temp[col] = df_temp[col].map(mapping).astype('Int64')
            
    # Integers check
    for col in ['Arm', 'SMOKE', 'event']:
        if col in df_temp.columns:
             df_temp[col] = df_temp[col].astype('Int64') 

    return df_temp

def select_event_or_max_time_row(group):
    """
    For per ID, selects the row where event=1 and TIME is max, 
    otherwise the row with the maximum TIME if no event occurred.
    """
    if group['event'].eq(1).any():
        # Select all event=1 rows, then the one with max TIME
        event_rows = group[group['event'] == 1]
        return event_rows.loc[event_rows['TIME'].idxmax()]
    else:
        # Select the row with max TIME
        return group.loc[group['TIME'].idxmax()]

def filter_events(df, scenarios):
    """
    Based on the Remox paper define the event and complete data,
    scenario 1: whole period
    scenario 2: 2 month
    scenario 3: 4 weeks

    """
    df = df.sort_values(['USUBJID', 'TIME'])
    rows = []
    for _, group in df.groupby('USUBJID'):
        if scenarios == 1:
            if len(group) > 1 and any(group['event'] == 1):
                rows.append(group[group['event'] == 1])   
            else:
                rows.append(group.tail(1))
        elif scenarios == 2:    
            if len(group) > 1 and any(group['event'] == 1):
                rows.append(group[group['event'] == 1])
            elif len(group) == 1 and any(group["event"] == 0):
                add_group = group.copy()
                add_group['TIME'] = 56.0
                rows.append(group)
                rows.append(add_group)
            elif max(group.TIME) < 28 and any(group['event'] != 1):
                add_group = group.tail(1).copy()
                add_group['TIME'] = 28.0
                rows.append(add_group)
            else:
                rows.append(group.tail(1))   
        elif scenarios == 3:
            if len(group) > 1 and any(group['event'] == 1):
                rows.append(group[group['event'] == 1])
            elif len(group) == 1 and any(group["event"] == 0):
                add_group = group.copy()
                add_group['TIME'] = 4.0
                rows.append(group)
                rows.append(add_group)
            elif max(group.TIME) < 4 and any(group['event'] != 1):
                add_group = group.tail(1).copy()
                add_group['TIME'] = 4.0
                rows.append(add_group)
            else:
                rows.append(group.tail(1))     
    return pd.concat(rows)    

def add_event(df, scenarios):
    """
    Create event if there is no event in the last time point
    """
    df = df.sort_values(['USUBJID', 'TIME'])
    rows = []
    for _, group in df.groupby('USUBJID'):
        if scenarios == 1 or scenarios == 4:
                rows.append(group)
        elif scenarios == 2:    
            if max(group.TIME) > 56.0 and not any(group.TIME == 56.0):
                group['TIME_difference'] = abs(group['TIME'] - 56)
                add_group = group.loc[group['TIME_difference'].idxmin()].to_frame().T
                add_group['TIME'] = 56.0
                add_group['event'] = 0
                rows.append(add_group)
            else:
                rows.append(group)    

        elif scenarios == 3:    
            if max(group.TIME) > 4.0 and not any(group.TIME == 4.0):
                group['TIME_difference'] = abs(group['TIME'] - 4)
                add_group = group.loc[group['TIME_difference'].idxmin()].to_frame().T
                add_group['TIME'] = 4.0
                add_group['event'] = 0
                rows.append(add_group.drop('TIME_difference', axis=1))
            else:
                rows.append(group)  
                                
    return pd.concat(rows)        


def data_one_record(Path, scenarios):
    """
    Keep only one event for each ID
    """
    df_ori = pd.read_csv(Path, index_col=0)

    df_all = pd.read_csv("../Data/df_covari_v2.csv", index_col=0) 
    
    # Time filtering based on scenarios
    if scenarios == 1:
        df = df_ori
    elif scenarios == 2:
        df = df_ori[df_ori["TIME"] <= 8]
        df_all = df_all[df_all["TIME"] <= 8] 
    elif scenarios == 3:
        df = df_ori[df_ori["TIME"] <= 4]
        df_all = df_all[df_all["TIME"] <= 4]

    # Feature selection and merge
    df = df.drop(columns=["X","MBDY_x"], errors='ignore') 
    df = pd.merge(df, df_all, on=["USUBJID", "Arm", "HT", "WT"], how='inner') 
    df = df[["USUBJID", "Arm", "HT", "WT", "TIME_x", "event_x", "CAVI", "SMOKE"]]
    df.rename(columns = {'TIME_x':'TIME', "event_x":"event"}, inplace = True)
    df_temp = df[["USUBJID","Arm","HT","WT","CAVI","SMOKE","TIME","event"]].copy() 
    
    # Apply categorical mapping 
    df_temp_nona = apply_categorical_mapping(df_temp).dropna() 

    print(f"total unique IDs (before dropna): {df_temp.USUBJID.nunique()}")
    print(f"clean data unique IDs (after dropna): {df_temp_nona.USUBJID.nunique()}")
    
    # Split data using GroupShuffleSplit
    splitter = GroupShuffleSplit(test_size=.20, n_splits=1, random_state = 7) 
    split = splitter.split(df_temp_nona, groups=df_temp_nona['USUBJID'])
    train_inds, test_inds = next(split)
    train = df_temp_nona.iloc[train_inds].copy()
    test = df_temp_nona.iloc[test_inds].copy()

    # Define X and y
    x_train, y_train = train.drop(columns=["USUBJID","event"]), train["event"]
    x_test, y_test = test.drop(columns=["USUBJID","event"]), test["event"]
    
    return df_temp_nona, x_train, y_train, x_test, y_test

def data_all(Path, scenarios):
    """
    Keep only one event for each ID
    """   
    df = pd.read_csv(Path, index_col=0)
    df = add_event(df, scenarios)

    if scenarios == 1:
        pass 
    elif scenarios == 2:
        df = df[df["TIME"] <= 8]
    elif scenarios == 3:
        df = df[df["TIME"] <= 4]

    df_temp = df[["USUBJID","Arm","HT","WT","CAVI","SMOKE","TIME","event", "BBLTTP"]].copy() 
    df_temp.sort_values(['TIME'], inplace=True)
    df_temp = df_temp.sort_values(["USUBJID", 'TIME'])
    
    df_temp_nona = apply_categorical_mapping(df_temp).dropna() 
    
    print(f"total unique IDs (before dropna): {df_temp.USUBJID.nunique()}")
    print(f"clean data unique IDs (after dropna): {df_temp_nona.USUBJID.nunique()}")
  
    # Split data using GroupShuffleSplit
    splitter = GroupShuffleSplit(test_size=.20, n_splits=1, random_state = 7)
    split = splitter.split(df_temp_nona, groups=df_temp_nona['USUBJID'])
    train_inds, test_inds = next(split)
    train = df_temp_nona.iloc[train_inds].copy()
    test = df_temp_nona.iloc[test_inds].copy()

    x_train, y_train = train.drop(columns=["USUBJID","event"]), train["event"]
    x_test, y_test = test.drop(columns=["USUBJID","event"]), test["event"]

    return df_temp_nona, x_train, y_train, x_test, y_test


def data_two_record(Path, scenarios, subscenario, directory_path):
    """
    For each ID, keep the first record and the last event, 
    if there is not event occur, it will be the last time point and event=0
    """     
    df = pd.read_csv(Path, index_col=0)
    df = add_event(df, scenarios)
    
    # --- 1. Data preparation---

    # Define feature columns
    common_cols = ["USUBJID","Arm","HT","WT","CAVI","SMOKE","TIME","event", "SEX", "AGE", "RACE", "BBLTTP", "DVTTP"]
    
    if subscenario in [1, 5, 8]:
        feature_cols = common_cols + ["TTP_028"]
        train_max_time, test_min_time, test_max_time = 29, 28, 57 # Week 4 vs Week 8
        time_to_drop = "TTP_028"
    elif subscenario in [2, 3, 7]:
        feature_cols = common_cols + ["TTP_014"]
        if subscenario == 2:
            train_max_time, test_min_time, test_max_time = 15, 14, 57 # Week 2 vs Week 8
        else: # 3 and 7
            train_max_time, test_min_time, test_max_time = 15, 14, 29 # Week 2 vs Week 4 (or 7 uses 2 vs 8 but with special features)
        time_to_drop = "TTP_014"
    elif subscenario == 4:
        # Apply filter_events and GroupShuffleSplit
        return data_two_record_scenario_4(df, scenarios, directory_path)
    elif subscenario == 6:
        # Apply pivoting
        feature_cols = common_cols + ["TTP_028"] 
        time_to_drop = "TTP_028"
    else:
        raise ValueError(f"Invalid subscenario: {subscenario}")
        
    df_temp = df[feature_cols].copy()
    df_temp = apply_categorical_mapping(df_temp)
    
    if subscenario in [5, 6]:
        # Normalization for TTP_028
        df_temp["TTP_028"] = df_temp["TTP_028"] / 24

    # --- 2. Filter data and create event label (DVTTP >= 42) ---
    
    # Find IDs that have records in the test phase
    valid_ids = df_temp[(df_temp['TIME'] > test_min_time) & (df_temp['TIME'] < test_max_time)]['USUBJID'].unique()
    
    # Train data
    train_df = df_temp[df_temp['USUBJID'].isin(valid_ids) & (df_temp['TIME'] < train_max_time)].copy()
    train_df['event'] = train_df['DVTTP'].apply(lambda x: 1 if x >= 42 else 0)
    
    # Test data
    test_df = df_temp[df_temp['USUBJID'].isin(valid_ids) & (df_temp['TIME'] > test_min_time) & (df_temp['TIME'] < test_max_time)].copy()
    test_df['event'] = test_df['DVTTP'].apply(lambda x: 1 if x >= 42 else 0)

    # --- 3. Select single record per ID (Event or Max Time) ---
    
    # Group and apply selection function
    train = train_df.groupby('USUBJID', group_keys=False).apply(select_event_or_max_time_row).reset_index(drop=True).drop_duplicates()
    test = test_df.groupby('USUBJID', group_keys=False).apply(select_event_or_max_time_row).reset_index(drop=True).drop_duplicates()

    # --- 4. Handle Subscenario 6 and 7 (Pivoting/Feature Injection) ---
    if subscenario in [6, 7]:
        # Helper function to get top N DVTTP records for pivoting
        def select_row_more(group: pd.DataFrame) -> pd.DataFrame:
            return group.nlargest(4, 'TIME')    

        # Use train_df for pivoting (all records up to train_max_time)
        pivot_df = train_df.groupby('USUBJID', group_keys=False).apply(select_row_more).reset_index(drop=True).drop_duplicates()
        pivot_df['flag'] = pivot_df.groupby('USUBJID').cumcount() + 1
        pivot_df = pivot_df.pivot(index='USUBJID', columns='flag', values='DVTTP').reset_index()
        pivot_df.columns = ['USUBJID', 'DVTTP_1', 'DVTTP_2', 'DVTTP_3', 'DVTTP_4']
        pivot_df = pivot_df.dropna(subset=[f'DVTTP_{i}' for i in range(1, 5)])

        # Filter test set to only include IDs present in the pivoted features
        test = pd.merge(test.drop(columns=[time_to_drop]), pivot_df, on='USUBJID', how='inner')
        
        # Re-merge pivot_df with train to align ID sets for the train set as well.
        train = pd.merge(train.drop(columns=[time_to_drop]), pivot_df, on='USUBJID', how='inner')


    # --- 5. Final Print and Return ---
    print(f"Number of unique USUBJID in train: {train['USUBJID'].nunique()}")
    print(f"Number of rows in train: {len(train)}")
    print(f"Number of unique USUBJID in test: {test['USUBJID'].nunique()}")
    print(f"Number of rows in test: {len(test)}")
    
    # Save to CSV
    train.to_csv(f"{directory_path}/scenar{scenarios}_subscenar{subscenario}_train.csv", index=False)
    test.to_csv(f"{directory_path}/scenar{scenarios}_subscenar{subscenario}_test.csv", index=False)
 
    # Define columns to drop: USUBJID, event, DVTTP, and the TTP column used for filtering
    x_train, y_train = train.drop(columns=["USUBJID","event", "DVTTP"], errors='ignore'), train["event"]
    x_test, y_test = test.drop(columns=["USUBJID","event", "DVTTP"], errors='ignore'), test["event"]
    
    # Drop the TIME_to_drop column specifically if it exists and was not used as a feature
    x_train = x_train.drop(columns=[time_to_drop], errors='ignore')
    x_test = x_test.drop(columns=[time_to_drop], errors='ignore')
 
    # Return the full dataframe (before final splitting) and the splits
    return df_temp, x_train, y_train, x_test, y_test

def data_two_record_scenario_4(df, scenarios, directory_path):
    """Specific function for Subscenario 4 logic using filter_events and GroupShuffleSplit."""
    feature_cols = ["USUBJID","Arm","HT","WT","CAVI","SMOKE","TIME","event", "SEX", "AGE", "RACE", "BBLTTP", "DVTTP"]
    df_temp = df[feature_cols].copy()
    df_temp = apply_categorical_mapping(df_temp)
    
    df_temp_filtered = filter_events(df_temp, scenarios).drop_duplicates()
    df_temp_nona = df_temp_filtered.dropna()
    
    print(f"total unique IDs (before dropna): {df_temp.USUBJID.nunique()}")
    print(f"clean data unique IDs (after dropna): {df_temp_nona.USUBJID.nunique()}")

    # Use GroupShuffleSplit for patient split
    splitter = GroupShuffleSplit(test_size=.20, n_splits=1, random_state = 7)
    split = splitter.split(df_temp_nona, groups=df_temp_nona['USUBJID'])
    train_inds, test_inds = next(split)
    train = df_temp_nona.iloc[train_inds].copy()
    test = df_temp_nona.iloc[test_inds].copy()
    

    train.to_csv(f"{directory_path}/scenar{scenarios}_subscenar{4}_train.csv", index=False)
    test.to_csv(f"{directory_path}/scenar{scenarios}_subscenar{4}_test.csv", index=False)
    
    # Define x and y
    x_train, y_train = train.drop(columns=["USUBJID","DVTTP","event"]), train["event"]
    x_test, y_test = test.drop(columns=["USUBJID","DVTTP","event"]), test["event"]

    return df_temp, x_train, y_train, x_test, y_test