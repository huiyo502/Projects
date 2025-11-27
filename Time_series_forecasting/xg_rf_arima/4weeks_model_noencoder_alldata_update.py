import numpy as np
import pandas as pd
import random
import argparse
import os
import sys
from datetime import datetime
import warnings

# Statsmodels / Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# Sklearn / Machine Learning
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, GridSearchCV

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

#  Argument Parsing
parser = argparse.ArgumentParser(description='Script for Adaptive TB Trial Forecasting')
parser.add_argument('--model_list', nargs='+', default=["xgb"],
                    help='List of models to run (e.g., ["xgb"], ["ARIMAX"], ["rf"], ["AMR"]).')
parser.add_argument('--scen', type=int, required=True,
                    help='Scenario: 1 for 14-day cutoff, 2 for 28-day cutoff.')
parser.add_argument('--name', type=str, default="TestRun",
                    help='Experiment name for folder organization.')
parser.add_argument('--treatmentsize', type=int, default=None,
                    help='Number of patients to randomly sample per treatment arm.')
args = parser.parse_args()

#  Setup Output Directory and log
today = datetime.now()
date_str = today.strftime("%m-%d")
model_name_for_file = args.model_list[0]
directory_path = f"../Result/{date_str}/{args.name}/{args.scen}"
os.makedirs(directory_path, exist_ok=True)

output_file_path = os.path.join(directory_path, f'output_mod_{model_name_for_file}.txt')
output_file = open(output_file_path, 'w')


#  Evaluation Function
def evaluate_forecast(df_prediction: pd.DataFrame):
    """
    input: df_prediction
        - 'ID'
        - 'DVTTP_true'
        - 'DVTTP_reconstructed'
    output: dict
        - mae_per_id
        - average_mae_per_id
        - overall_mae
        - rmse
        - mase
    """
    df_eval = df_prediction.dropna(subset=['DVTTP_true', 'DVTTP_reconstructed'])

    if df_eval.empty:
        print("WARNING: All data dropped during NaN check. Cannot calculate metrics.")
        return {
            "mae_per_id": pd.Series(dtype=float),
            "average_mae_per_id": np.nan,
            "overall_mae": np.nan,
            "rmse": np.nan,
            "mase": np.nan
        }

    # per-ID MAE
    mae_per_id = df_eval.groupby('ID').apply(
        lambda x: mean_absolute_error(x['DVTTP_true'], x['DVTTP_reconstructed'])
    )
    average_mae_per_id = mae_per_id.mean()

    # overall MAE / RMSE
    overall_mae = mean_absolute_error(df_eval['DVTTP_true'], df_eval['DVTTP_reconstructed'])
    rmse = np.sqrt(mean_squared_error(df_eval['DVTTP_true'], df_eval['DVTTP_reconstructed']))

    # MASE：naive baseline (random walk: y_t_hat = y_{t-1})
    def _naive_mae(x):
        y_true = x['DVTTP_true'].values
        if len(y_true) < 2:
            return np.nan
        y_pred = y_true[:-1]
        y_tgt = y_true[1:]
        return mean_absolute_error(y_tgt, y_pred)

    naive_mae_per_id = df_eval.groupby('ID').apply(_naive_mae).dropna()
    if len(naive_mae_per_id) > 0:
        baseline_mae = naive_mae_per_id.mean()
        mase = average_mae_per_id / baseline_mae if baseline_mae > 0 else np.nan
    else:
        mase = np.nan

    print("\n--- Forecast Evaluation ---")
    print(f"Per-ID MAE (mean over IDs): {average_mae_per_id:.4f}")
    print(f"Overall MAE:               {overall_mae:.4f}")
    print(f"RMSE:                      {rmse:.4f}")
    print(f"MASE (vs naive rw):        {mase:.4f}")
    print("---------------------------\n")

    return {
        "mae_per_id": mae_per_id,
        "average_mae_per_id": average_mae_per_id,
        "overall_mae": overall_mae,
        "rmse": rmse,
        "mase": mase
    }



#  Model 
def model_run(model_list, directory_path,
              x_train, y_train, x_test, y_test,
              train_groups, train_df, test_df):

    keyword = model_list[0]
    forecast = None

    if keyword == "AMR":
        auto = pm.auto_arima(
            y_train,
            seasonal=False,
            stepwise=True,
            trace=False,
            suppress_warnings=True
        )
        p, d, q = auto.order
        mod = ARIMA(y_train, order=(p, d, q))
        model_fit = mod.fit()
        pred_res = model_fit.forecast(steps=len(y_test))
        forecast = np.asarray(pred_res, dtype=float)

    elif keyword == "ARIMAX":
        # Data preparation
        X_tr = x_train.copy()
        X_te = x_test.copy()
        X_tr = X_tr.apply(pd.to_numeric, errors='coerce').astype('float64')
        X_te = X_te.apply(pd.to_numeric, errors='coerce').astype('float64')

        X_tr.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_te.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_tr.fillna(0.0, inplace=True)
        X_te.fillna(0.0, inplace=True)

        const_cols = [c for c in X_tr.columns if X_tr[c].nunique(dropna=False) <= 1]
        if const_cols:
            X_tr = X_tr.drop(columns=const_cols)
            X_te = X_te.drop(columns=const_cols)

        X_te = X_te.reindex(columns=X_tr.columns)

        # Model setting
        auto = pm.auto_arima(
            y_train,
            X=X_tr,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            trace=False
        )
        p, d, q = auto.order
        
        # Model
        mod = SARIMAX(
            y_train,
            exog=X_tr,
            order=(p, d, q),
            trend='n',
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = mod.fit(disp=False)

        pred_res = model_fit.get_forecast(steps=len(y_test), exog=X_te)
        forecast = pred_res.predicted_mean.to_numpy().astype(float)


    elif keyword in ["rf", "xgb"]:
        # Split by group
        n_groups = len(np.unique(train_groups))
        n_splits = min(5, n_groups) if n_groups > 1 else 2
        if n_splits < 2:
            print("WARNING: Not enough unique patient IDs for GroupKFold. Using n_splits=2.")
            n_splits = 2
        group_cv = GroupKFold(n_splits=n_splits)

        if keyword == "rf":
            param_dist = {
                'n_estimators': [100, 300, 500, 800, 1000],
                'max_depth': [10, 20, 30, 40, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            }
            mdl = RandomForestRegressor(random_state=42, n_estimators=100)
            random_search = RandomizedSearchCV(
                estimator=mdl,
                param_distributions=param_dist,
                n_iter=50,
                cv=group_cv,
                verbose=1,
                random_state=42,
                n_jobs=-1
            )
            random_search.fit(x_train, y_train, groups=train_groups)
            best_model = random_search.best_estimator_

        elif keyword == "xgb":
            model_xgb = xgb.XGBRegressor(
                tree_method='hist',
                objective='reg:squarederror',
                random_state=42
            )
            parameters = {
                "max_depth": [3, 4, 6, 5, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                "n_estimators": [100, 300, 500, 700, 900, 1000],
                "colsample_bytree": [0.3, 0.5, 0.7]
            }
            grid_search = GridSearchCV(
                estimator=model_xgb,
                cv=group_cv,
                param_grid=parameters,
                verbose=1,
                n_jobs=-1
            )
            grid_search.fit(x_train, y_train, groups=train_groups)
            best_model = grid_search.best_estimator_

        if isinstance(x_train, pd.DataFrame) and isinstance(x_test, pd.DataFrame):
            x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

        forecast = best_model.predict(x_test)

    else:
        print(f"Error: Model {keyword} not recognized or implemented.")
        forecast = np.zeros(len(y_test))

    # Scaling  back to original time 
    if forecast is None or len(forecast) != len(y_test):
        print(f"FATAL ERROR: Forecast length mismatch or None. "
              f"Expected {len(y_test)}, got {len(forecast) if forecast is not None else 'None'}.")
        return pd.DataFrame()

    df_prediction = pd.DataFrame({
        'DVTTP_diff': np.asarray(forecast, dtype=float),
        'TIME': test_df['TIME'].values,
        'Arm': test_df['Arm'].values,
        'ID': test_df['ID'].values,
        'BBLTTP': test_df['BBLTTP'].values,
        'DVTTP_true': test_df['DVTTP'].values
    })

    last_values = df_prediction.groupby('ID')['BBLTTP'].transform('last')
    df_prediction['DVTTP_reconstructed'] = last_values + df_prediction.groupby('ID')['DVTTP_diff'].cumsum()
    df_prediction['DVTTP_reconstructed'] = df_prediction['DVTTP_reconstructed'].clip(upper=42, lower=0)

    _ = evaluate_forecast(df_prediction)

    return df_prediction


#  Plot 
def generate_plots(train, test, df_prediction, directory_path, model_name):
    """Generates and saves Plotly visualization of actual vs predicted biomarker time-series."""
    replace_dict = {0: "Control Regimen", 2: "Isoniazid Regimen", 1: "Ethambutol Regimen"}
    train_ = train.copy()
    test_ = test.copy()
    pred_ = df_prediction.copy()

    # make sure there is DVTTP_true
    if "DVTTP_true" not in test_.columns and "DVTTP" in test_.columns:
        test_["DVTTP_true"] = test_["DVTTP"]

    for df_ in (train_, test_, pred_):
        df_["Arm"] = df_["Arm"].replace(replace_dict)
        df_["TIME"] = pd.to_numeric(df_["TIME"], errors="coerce")

    order_map = {"Control Regimen": 0, "Isoniazid Regimen": 1, "Ethambutol Regimen": 2}
    arms_present = pd.Index(pd.concat([train_["Arm"], test_["Arm"], pred_["Arm"]]).dropna().unique())
    arms = sorted(arms_present, key=lambda a: order_map.get(a, 99))
    num_arms = len(arms)

    fig = make_subplots(
        rows=1, cols=num_arms,
        subplot_titles=[f"{arm}" for arm in arms],
        shared_yaxes=True
    )

    colors = {
        'train': '#1f78b4',
        'test': '#d95f02',
        'Mean': '#93c47d',
        'ci': 'rgba(117,112,179,0.20)'
    }

    for i, arm in enumerate(arms):
        show_legend = (i == 0)
        train_arm = train_[train_["Arm"] == arm]
        test_arm = test_[test_["Arm"] == arm]
        pred_arm = pred_[pred_["Arm"] == arm]

        train_mean = (train_arm.groupby('TIME', as_index=False)['DVTTP']
                      .mean().sort_values('TIME'))
        test_mean = (test_arm.groupby('TIME', as_index=False)['DVTTP_true']
                     .mean().sort_values('TIME'))

        if len(pred_arm) > 0:
            pred_stats = (
                pred_arm.groupby('TIME')['DVTTP_reconstructed']
                .agg(mean='mean',
                     median='median',
                     std='std',
                     n='size',
                     q05=lambda s: s.quantile(0.05),
                     q95=lambda s: s.quantile(0.95))
                .reset_index()
                .sort_values('TIME')
            )

            pred_stats['std'] = pred_stats['std'].fillna(0.0)
            pred_stats['n'] = pred_stats['n'].fillna(0).astype(int)
            sem = pred_stats['std'] / np.sqrt(pred_stats['n'].clip(lower=1))
            Z = 1.645
            ci_lower_sem = pred_stats['mean'] - Z * sem
            ci_upper_sem = pred_stats['mean'] + Z * sem
            use_sem = pred_stats['n'] >= 2
            pred_stats['ci_lower'] = np.where(use_sem, ci_lower_sem, pred_stats['q05'])
            pred_stats['ci_upper'] = np.where(use_sem, ci_upper_sem, pred_stats['q95'])

            for c in ['ci_lower', 'ci_upper', 'mean', 'median']:
                if c in pred_stats:
                    pred_stats[c] = pred_stats[c].clip(lower=0, upper=42)

            pred_stats = pred_stats.dropna(subset=['ci_lower', 'ci_upper'])

            x = pred_stats['TIME'].to_numpy()
            upper = pred_stats['ci_upper'].to_numpy()
            lower = pred_stats['ci_lower'].to_numpy()

            if len(x) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=np.r_[x, x[::-1]],
                        y=np.r_[upper, lower[::-1]],
                        fill='toself',
                        fillcolor=colors['ci'],
                        line_color='rgba(255,255,255,0)',
                        hoverinfo='skip',
                        name="90% Prediction Confidence Interval",
                        showlegend=show_legend,
                        legendgroup='pred_band'
                    ),
                    row=1, col=i + 1
                )

            fig.add_trace(
                go.Scatter(
                    x=pred_stats["TIME"],
                    y=pred_stats["median"],
                    mode='lines+markers',
                    name='Median Prediction',
                    line=dict(color=colors['Mean'], width=2),
                    hovertemplate='Time: %{x}<br>Median pred: %{y:.2f}',
                    showlegend=show_legend,
                    legendgroup='median_pred'
                ),
                row=1, col=i + 1
            )

        fig.add_trace(
            go.Scatter(
                x=train_mean["TIME"],
                y=train_mean["DVTTP"],
                mode='lines+markers',
                name='Train mean (Actual)',
                line=dict(color=colors['train'], width=2),
                hovertemplate='Time: %{x}<br>Train: %{y:.2f}',
                showlegend=show_legend,
                legendgroup='train_mean'
            ),
            row=1, col=i + 1
        )

        fig.add_trace(
            go.Scatter(
                x=test_mean["TIME"],
                y=test_mean["DVTTP_true"],
                mode='lines+markers',
                name='Test mean (Actual)',
                line=dict(color=colors['test'], width=2, dash='dash'),
                hovertemplate='Time: %{x}<br>Test: %{y:.2f}',
                showlegend=show_legend,
                legendgroup='test_mean'
            ),
            row=1, col=i + 1
        )

    fig.update_layout(
        title=f"Biomarker Forecasting: {model_name} (Scen {args.scen})",
        yaxis_title="Time to Positivity (days)",
        showlegend=True,
        legend=dict(
            x=0.5, y=-0.2, xanchor="center", yanchor="top",
            orientation="h", bordercolor="Black", borderwidth=1
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=14),
        height=600,
        width=1200,
        hovermode="x unified"
    )
    fig.add_annotation(
        text="Time since randomization (days)",
        xref="paper", yref="paper",
        x=0.5, y=-0.18, showarrow=False, font=dict(size=16)
    )

    for i in range(1, num_arms + 1):
        fig.update_xaxes(
            showgrid=False, showline=True, linewidth=2, linecolor='black',
            range=[0, 56], dtick=7, row=1, col=i
        )
        fig.update_yaxes(
            showgrid=False, showline=True, linewidth=2, linecolor='black',
            range=[0, 45], dtick=5, row=1, col=i
        )

    fname = f"{model_name}_population_forecast.png"
    out_path = os.path.join(directory_path, fname)
    try:
        pio.write_image(fig, out_path)
    except Exception as e:
        print(f"WARNING: Could not save image {out_path}: {e}")


# =========================Run section===============================================================================================================================

try:
    sys.stdout = output_file

    # Load data
    train_ori = pd.read_csv("../Data/scenar1_subscenar1_train.csv", index_col=0)
    test_ori = pd.read_csv("../Data/scenar1_subscenar1_test.csv", index_col=0)
    df_combined = pd.concat([train_ori, test_ori], ignore_index=True)

    df = df_combined.sort_values(by=["ID", "TIME"]).reset_index(drop=True)

    # Subsample by treatmentsize 
    if args.treatmentsize is not None:
        PATIENT_COL = 'ID'
        ARM_COL = 'Arm'
        patients = df_combined[[PATIENT_COL, ARM_COL]].drop_duplicates(PATIENT_COL)
        sample_ids = patients.groupby(ARM_COL)[PATIENT_COL] \
            .apply(lambda s: s.sample(n=min(args.treatmentsize, len(s)), random_state=42)) \
            .reset_index(level=0, drop=True)
        df = df_combined[df_combined[PATIENT_COL].isin(sample_ids)]

    print(f"Total Unique Patients after subsampling: {df['ID'].nunique()}")

    df = df.sort_values(by=["ID", "TIME"]).reset_index(drop=True)
    df['DVTTP'] = df['DVTTP'].clip(upper=42)

    # Feature Engineering
    df['DVTTP_diff'] = df.groupby('ID')['DVTTP'].diff().fillna(0)

    model_name = args.model_list[0]
    lag_range = 2 if args.scen == 1 else 4
    if model_name in ["rf", "xgb"]:
        for lag in range(1, lag_range):
            df[f"lag_{lag}"] = df.groupby('ID')["DVTTP_diff"].shift(lag)
            df = df.dropna(subset=[f"lag_{i}" for i in range(1, lag_range)])
    else:
        pass
#        for lag in range(1, 10):
#            df[f"lag_{lag}"] = df["DVTTP"].shift(lag)
#            df = df.dropna(subset=[f"lag_{i}" for i in range(1, 10)])

    # Train-Test Split
    cutoff = 14 if args.scen == 1 else 28
    train = df[df["TIME"] <= cutoff].copy()
    test_temp = df[df["TIME"] > cutoff].copy()

    last_train_values = train.groupby('ID')['DVTTP'].last().reset_index().rename(columns={"DVTTP": "BBLTTP_last"})
    test = test_temp.merge(last_train_values, on='ID', how='left')
    if 'BBLTTP' in test.columns:
        test.drop('BBLTTP', axis=1, inplace=True)
    test.rename(columns={"BBLTTP_last": "BBLTTP"}, inplace=True)
    train.dropna(inplace=True)
    test.dropna(inplace=True)
    # Prepare X/y
    cols_to_drop_from_X = ["TTP_028", "DVTTP", "DVTTP_diff", "event", "ID"]
    x_train = train.drop(columns=[c for c in cols_to_drop_from_X if c in train.columns])
    y_train = train["DVTTP_diff"]

    x_test = test.drop(columns=[c for c in cols_to_drop_from_X if c in test.columns])
    y_test = test["DVTTP_diff"]

    train_groups = train['ID']

    # Run model
    model_name = args.model_list[0]
    df_prediction = model_run(args.model_list, directory_path,
                              x_train, y_train, x_test, y_test,
                              train_groups, train, test)

    # Save results and plot
    if not df_prediction.empty:
        csv_path = os.path.join(directory_path, f"{model_name}_forecast_data.csv")
        df_prediction.to_csv(csv_path, index=False)
        generate_plots(train, test, df_prediction, directory_path, model_name)

finally:
    sys.stdout = sys.__stdout__
    output_file.close()
    print("Execution complete. Check output folder for results and log file.")
