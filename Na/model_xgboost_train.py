import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import scipy.linalg
scipy.linalg.inv = np.linalg.inv
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import optuna
from optuna.samplers import TPESampler
from pathlib import Path
import warnings
import pickle

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING) 

plt.rcParams['font.family'] = 'Arial' 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = Path("Modeling_Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.2
OPTUNA_TRIALS = 50

TARGET_COLS = ['PCR', 'PC', 'TC']
FEATURE_COLS = ['d002', 'La', 'Lc', 'Pr', 'g', 'Va', 'd', 'xi', 'r', 'D/R']

def plot_combined_scatter(scatter_results_dict, output_dir):
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Ideal Line', alpha=0.6)
    
    colors = {'PCR': '#1f77b4', 'PC': '#ff7f0e', 'TC': '#2ca02c'}
    markers = {'PCR': 'o', 'PC': 's', 'TC': '^'}
    fallback_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fallback_markers = ['o', 's', '^', 'D', 'v', 'p']
    
    y_text_pos = 0.95
    idx = 0
    
    origin_export_data = []

    for target, res in scatter_results_dict.items():
        t_str = str(target)
        c = colors.get(t_str, fallback_colors[idx % len(fallback_colors)])
        m = markers.get(t_str, fallback_markers[idx % len(fallback_markers)])
        idx += 1
        
        scaler = res['plot_scaler']
        y_test_norm = scaler.transform(res['y_test'].reshape(-1, 1)).ravel()
        y_pred_norm = scaler.transform(res['y_pred'].reshape(-1, 1)).ravel()
        
        for i in range(len(y_test_norm)):
            origin_export_data.append({
                'Target': t_str,
                'True_Value_Original': res['y_test'][i],
                'Predicted_Value_Original': res['y_pred'][i],
                'True_Value_Normalized_X': y_test_norm[i],
                'Predicted_Value_Normalized_Y': y_pred_norm[i]
            })

        plt.scatter(y_test_norm, y_pred_norm, c=c, marker=m, label=t_str, alpha=0.7, edgecolors='white', s=80)
        
        info_text = f"{t_str}: $R^2$={res['r2']:.4f} | RMSE={res['rmse']:.4f}"
        plt.text(0.05, y_text_pos, info_text, transform=plt.gca().transAxes, fontsize=12, color=c, fontweight='bold')
        y_text_pos -= 0.05
        
    plt.xlabel('True Value (Normalized)')
    plt.ylabel('Predicted Value (Normalized)')
    plt.title('Test Prediction Accuracy', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(output_dir / "Combined_Scatter.png", bbox_inches='tight', dpi=300)
    plt.close()

    df_origin = pd.DataFrame(origin_export_data)
    df_origin.to_excel(output_dir / "Scatter_Plot_Data_Origin.xlsx", index=False)

def plot_and_export_shap(model, X_data, y_data, target_name, output_dir):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    if isinstance(shap_values, list): 
        shap_values = shap_values[0]
    
    y_range = np.max(y_data) - np.min(y_data)
    if y_range > 0:
        shap_values = shap_values / y_range
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_shap = np.sum(mean_abs_shap) if np.sum(mean_abs_shap) > 0 else 1.0
    importance_ratio = mean_abs_shap / total_shap
    
    feature_scaler = MinMaxScaler()
    X_data_norm = pd.DataFrame(feature_scaler.fit_transform(X_data), columns=X_data.columns, index=X_data.index)
    
    df_imp = pd.DataFrame({
        'Feature': X_data.columns, 
        'Mean_Abs_SHAP': mean_abs_shap,
        'Importance_Ratio': importance_ratio
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    
    excel_path = output_dir / f"SHAP_Data_Origin_{target_name}.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        X_data_norm.to_excel(writer, sheet_name="1_Feature_Values_X", index=False)
        pd.DataFrame(shap_values, columns=X_data.columns).to_excel(writer, sheet_name="2_SHAP_Values_Y", index=False)
        df_imp.to_excel(writer, sheet_name="3_Bar_Importance", index=False)
    
    sorted_features = df_imp['Feature'].tolist()
    sorted_values = df_imp['Mean_Abs_SHAP'].values 
    
    col_indices = [X_data.columns.get_loc(c) for c in sorted_features]
    shap_sorted = shap_values[:, col_indices]
    X_sorted = X_data_norm[sorted_features]
    
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.2)
    
    ax0 = plt.subplot(gs[0])
    shap.summary_plot(shap_sorted, X_sorted, plot_type="dot", show=False, sort=False, feature_names=sorted_features, cmap='RdBu_r')
    ax0.set_title(f"{target_name} SHAP Global Manifold", fontsize=16, fontweight='bold')
    
    ax1 = plt.subplot(gs[1])
    y_pos = np.arange(len(sorted_features))
    ax1.barh(y_pos[::-1], sorted_values, color='#6aa84f', alpha=0.8, height=0.6)
    ax1.set_yticks(y_pos[::-1])
    ax1.set_yticklabels(sorted_features)
    ax1.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax1.set_title("Feature Importance", fontsize=16, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for i, v in enumerate(sorted_values):
        ax1.text(v, y_pos[::-1][i], f" {v:.4f}", va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / f"SHAP_Plot_{target_name}.png", bbox_inches='tight', dpi=300)
    plt.close()

def optimize_hyperparameters(X_train, y_train, groups_train):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 40, 100),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.12, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': 126,
            'n_jobs': -1
        }
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=126)                             
        train_idx, val_idx = next(gss.split(X_train, y_train, groups=groups_train))
        
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        
        return mean_squared_error(y_va, preds)
    

    sampler = TPESampler(seed=126)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    
    best_params = study.best_params
    best_params['random_state'] = 126
    
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)
    return final_model

if __name__ == "__main__":
    DATA_FILE = "Data.xlsx" 
    
    if not Path(DATA_FILE).exists():
        print(f"Error: Dataset '{DATA_FILE}' not found.")
        exit()
        
    df_raw = pd.read_excel(DATA_FILE)
    df_raw.columns = df_raw.columns.astype(str).str.strip()
    
    if 'group_id' not in df_raw.columns:
        print("Error: 'group_id' column is missing for grouped split.")
        exit()
        
    unique_groups = df_raw['group_id'].unique()
    
    train_groups, test_groups = train_test_split(unique_groups, test_size=TEST_SIZE, random_state=126)
    
    train_df = df_raw[df_raw['group_id'].isin(train_groups)].copy()
    test_df = df_raw[df_raw['group_id'].isin(test_groups)].copy()
    
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COLS]
    groups_train = train_df['group_id']
    
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COLS]
    
    X_all = df_raw[FEATURE_COLS]
    
    scatter_results = {}
    
    print("Model training and hyperparameter optimization started.")
    
    for target in TARGET_COLS:
        print(f"Processing target: {target}")
        
        model = optimize_hyperparameters(X_train, y_train[target], groups_train)
        
        preds = model.predict(X_test)
        

        y_test_original = y_test[target].values
        plot_scaler = MinMaxScaler()
        plot_scaler.fit(np.concatenate([y_test_original, preds]).reshape(-1, 1))
        y_test_normalized = plot_scaler.transform(y_test_original.reshape(-1, 1)).ravel()
        preds_normalized = plot_scaler.transform(preds.reshape(-1, 1)).ravel()
        
        test_r2 = r2_score(y_test_normalized, preds_normalized)
        test_rmse = np.sqrt(mean_squared_error(y_test_normalized, preds_normalized))
        
        print(f"[{target}] Test R2: {test_r2:.4f}, Normalized RMSE: {test_rmse:.4f}")
        
        scatter_results[target] = {
            'y_test': y_test_original, 
            'y_pred': preds,           
            'r2': test_r2, 
            'rmse': test_rmse,
            'plot_scaler': plot_scaler
        }
        
        plot_and_export_shap(model, X_all, df_raw[target].values, target, OUTPUT_DIR)
        
        with open(OUTPUT_DIR / f"XGB_Model_{target}.pkl", "wb") as f:
            pickle.dump(model, f)
            
    plot_combined_scatter(scatter_results, OUTPUT_DIR)
    print(f"Execution complete. All Origin-ready data saved directly in {OUTPUT_DIR}")
