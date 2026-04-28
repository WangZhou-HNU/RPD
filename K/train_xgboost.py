import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from pathlib import Path
import warnings
import sys

# Import Optuna for hyperparameter optimization
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress verbose logging
except ImportError:
    print("❌ Error: 'optuna' library is missing. Please run: pip install optuna")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 0. Basic Configuration
# ==========================================
try:
    BASE_DIR = Path(__file__).parent.resolve()
except NameError:
    BASE_DIR = Path.cwd().resolve()

DATA_FILENAME = "Data_Total.xlsx"
DATA_PATH = BASE_DIR / DATA_FILENAME

# Define output directory
OUTPUT_DIR = BASE_DIR / "Training_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotting configuration
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

# ==========================================
# 1. Advanced Plotting Functions
# ==========================================

def plot_composite_shap(shap_values, X_test, target_name):
    """
    Generates a composite figure: 
    Left: SHAP Beeswarm summary plot.
    Right: Feature Importance Bar plot based on mean |SHAP| values.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df_imp = pd.DataFrame({'feature': X_test.columns, 'mean_abs': mean_abs_shap})
    df_imp = df_imp.sort_values('mean_abs', ascending=False).head(15).reset_index(drop=True)
    
    sorted_features = df_imp['feature'].tolist()
    sorted_values = df_imp['mean_abs'].values
    
    col_indices = [X_test.columns.get_loc(c) for c in sorted_features]
    shap_sorted = shap_values[:, col_indices]
    X_sorted = X_test[sorted_features]
    
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.2)
    
    # Subplot 1: SHAP Summary
    ax0 = plt.subplot(gs[0])
    shap.summary_plot(shap_sorted, X_sorted, plot_type="dot", show=False, sort=False, 
                      feature_names=sorted_features, cmap='RdBu_r')
    ax0.set_title(f"{target_name} SHAP Overview", fontsize=16, fontweight='bold')
    
    # Subplot 2: Feature Importance Bar Chart
    ax1 = plt.subplot(gs[1])
    y_pos = np.arange(len(sorted_features))
    ax1.barh(y_pos[::-1], sorted_values, color='#6aa84f', alpha=0.8, height=0.6)
    ax1.set_yticks(y_pos[::-1])
    ax1.set_yticklabels(sorted_features)
    ax1.set_xlabel("Mean |SHAP value|")
    ax1.set_title("Feature Importance", fontsize=16, fontweight='bold')
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for i, v in enumerate(sorted_values):
        ax1.text(v, y_pos[::-1][i], f" {v:.4f}", va='center', fontsize=10)

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"Composite_SHAP_{target_name}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_learning_curve(evals_result, target_name):
    """Plots the training and validation RMSE loss curves."""
    if not evals_result: return
    train_errors = evals_result['validation_0']['rmse']
    test_errors = evals_result['validation_1']['rmse']
    epochs = len(train_errors)
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, train_errors, label='Train Loss', linewidth=2)
    plt.plot(x_axis, test_errors, label='Test Loss', linewidth=2, linestyle='--')
    plt.legend()
    plt.xlabel('Estimators (Epochs)')
    plt.ylabel('RMSE')
    plt.title(f'{target_name} Training Loss Curve')
    plt.grid(True, alpha=0.3)
    
    save_path = OUTPUT_DIR / f"Loss_Curve_{target_name}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_combined_scatter(all_results):
    """Generates a combined scatter plot for multiple targets (Predicted vs. True)."""
    plt.figure(figsize=(8, 8))
    colors = {'PC': '#1f77b4', 'TC': '#ff7f0e', 'PCR': '#2ca02c'}
    markers = {'PC': 'o', 'TC': 's', 'PCR': '^'}
    
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Ideal Line', alpha=0.6)
    y_text_pos = 0.95
    
    for target, res in all_results.items():
        c = colors.get(target, 'blue')
        m = markers.get(target, 'o')
        plt.scatter(res['y_test'], res['y_pred'], c=c, marker=m, label=target, alpha=0.7, edgecolors='white', s=80)
        
        info_text = f"{target}: $R^2$={res['r2']:.4f} | RMSE={res['rmse']:.4f}"
        plt.text(0.05, y_text_pos, info_text, transform=plt.gca().transAxes, 
                 fontsize=12, color=c, fontweight='bold')
        y_text_pos -= 0.05
        
    plt.xlabel('True Value (Normalized)')
    plt.ylabel('Predicted Value (Normalized)')
    plt.title('Combined Prediction Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    save_path = OUTPUT_DIR / "Combined_Scatter.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ==========================================
# 2. Data Export Utility
# ==========================================
def save_raw_data(target_name, shap_values, X_test, df_imp, evals_result, y_test, y_pred, r2, rmse):
    """Saves raw data, metrics, and logs to an Excel file."""
    filename = OUTPUT_DIR / f"RawData_{target_name}.xlsx"
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 1. SHAP Values
        pd.DataFrame(shap_values, columns=X_test.columns).to_excel(writer, sheet_name='SHAP_Values', index=False)
        
        # 2. Feature Values
        pd.DataFrame(X_test.reset_index(drop=True)).to_excel(writer, sheet_name='Feature_Values', index=False)
        
        # 3. Feature Importance
        df_imp.to_excel(writer, sheet_name='Feature_Importance', index=False)
        
        # 4. Metrics
        df_metrics = pd.DataFrame({
            'Metric': ['R2', 'RMSE'],
            'Value': [r2, rmse]
        })
        df_metrics.to_excel(writer, sheet_name='Metrics', index=False)

        # 5. Loss History
        if evals_result:
            df_loss = pd.DataFrame({
                'Epoch': range(len(evals_result['validation_0']['rmse'])),
                'Train_RMSE': evals_result['validation_0']['rmse'],
                'Test_RMSE': evals_result['validation_1']['rmse']
            })
            df_loss.to_excel(writer, sheet_name='Loss_History', index=False)
            
        # 6. Predictions
        df_pred = pd.DataFrame({'True_Value': y_test, 'Predicted_Value': y_pred})
        df_pred.to_excel(writer, sheet_name='Predictions', index=False)

# ==========================================
# 3. Optimization Engine
# ==========================================

def objective(trial, X, y):
    """
    Optuna objective function: Optimizes hyperparameters to maximize CV R2 score.
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 8), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    }
    
    model = xgb.XGBRegressor(**param)
    
    # 3-Fold Cross-Validation
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    return scores.mean()

# ==========================================
# 4. Main Pipeline
# ==========================================
def run_optimization_pipeline():
    print(f"📖 Loading dataset from: {DATA_PATH}")
    # Load 'Dataset'
    try:
        df_total = pd.read_excel(DATA_PATH, sheet_name='Dataset').dropna()
    except ValueError:
        print("⚠️ Warning: 'Dataset' sheet not found, loading the first sheet instead.")
        df_total = pd.read_excel(DATA_PATH).dropna()

    df_total.columns = df_total.columns.str.strip()

    # Automatically identify targets
    targets = ['PC', 'TC', 'PCR']
    available_targets = [t for t in targets if t in df_total.columns]
    
    if not available_targets:
        print("❌ Error: Target columns (PC/TC/PCR) not found in the dataset.")
        return

    # Define Features
    drop_cols = available_targets + [c for c in df_total.columns if 'Unnamed' in c]
    feature_cols = [c for c in df_total.columns if c not in drop_cols]
    
    X = df_total[feature_cols]
    
    # Unified Dataset Split
    print("✂️  Performing unified dataset split (80% Train / 20% Test)...")
    train_idx, test_idx = train_test_split(df_total.index, test_size=0.2, random_state=42)
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]

    combined_results = {}

    for target_name in available_targets:
        print(f"\n🔄 [Processing] {target_name} (Optimizing with Optuna)...")

        # Data Preparation
        y_train_raw = df_total.loc[train_idx, target_name].values.reshape(-1, 1)
        y_test_raw = df_total.loc[test_idx, target_name].values.reshape(-1, 1)
        
        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(y_train_raw).ravel()
        y_test = scaler.transform(y_test_raw).ravel()
        
        # 1. Hyperparameter Search
        print("   🔍 Searching for best hyperparameters (20 trials)...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)
        
        best_params = study.best_params
        print(f"      🏆 Best CV R2: {study.best_value:.4f}")
        
        # 2. Train with Best Parameters
        print("   🏋️  Training final model with best parameters...")
        final_params = best_params.copy()
        final_params.update({'n_jobs': -1, 'random_state': 42, 'verbosity': 0})
        
        model = xgb.XGBRegressor(**final_params)
        model.fit(
            X_train, y_train, 
            eval_set=[(X_train, y_train), (X_test, y_test)], 
            verbose=False
        )
        
        # 3. Evaluation
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"   ✅ Final Test Result: R2={r2:.4f}, RMSE={rmse:.4f}")
        
        combined_results[target_name] = {
            'y_test': y_test, 'y_pred': y_pred, 'r2': r2, 'rmse': rmse
        }
        
        # 4. SHAP Analysis
        model.get_booster().set_param('base_score', '0.5')
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list): shap_values = shap_values[0]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        total_shap = np.sum(mean_abs_shap)

        df_imp = pd.DataFrame({
            'Feature': X_test.columns,
            'Mean_Abs_SHAP': mean_abs_shap,
            'Percentage': (mean_abs_shap / total_shap) * 100
        }).sort_values('Percentage', ascending=False)

        plot_composite_shap(shap_values, X_test, target_name)
        plot_learning_curve(model.evals_result(), target_name)
        
        # Save results
        save_raw_data(target_name, shap_values, X_test, df_imp, model.evals_result(), y_test, y_pred, r2, rmse)

    # 5. Summary Plot
    if combined_results:
        print("\n🎨 Generating combined scatter plot...")
        plot_combined_scatter(combined_results)

if __name__ == "__main__":
    print("="*60)
    print("🚀 Optuna Intelligent Training Engine")
    print("Step 1: Data Loading")
    print("Step 2: Automated Hyperparameter Optimization")
    print("="*60)
    
    run_optimization_pipeline()
    
    print(f"\n✨ All tasks completed! Please check output directory: {OUTPUT_DIR}")