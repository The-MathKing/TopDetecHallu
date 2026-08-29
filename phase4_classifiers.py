import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate(X, y, name=""):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {'lr_roc_auc': [], 'lr_pr_auc': [], 'lr_f1': [],
               'xgb_roc_auc': [], 'xgb_pr_auc': [], 'xgb_f1': []}
               
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        lr_preds = lr.predict_proba(X_test)[:, 1]
        lr_preds_bin = lr.predict(X_test)
        
        metrics['lr_roc_auc'].append(roc_auc_score(y_test, lr_preds))
        metrics['lr_pr_auc'].append(average_precision_score(y_test, lr_preds))
        metrics['lr_f1'].append(f1_score(y_test, lr_preds_bin))
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
        xgb_preds_bin = xgb_model.predict(X_test)
        
        metrics['xgb_roc_auc'].append(roc_auc_score(y_test, xgb_preds))
        metrics['xgb_pr_auc'].append(average_precision_score(y_test, xgb_preds))
        metrics['xgb_f1'].append(f1_score(y_test, xgb_preds_bin))
        
    print(f"\n--- {name} ---")
    print(f"LR  - ROC-AUC: {np.mean(metrics['lr_roc_auc']):.4f} \u00B1 {np.std(metrics['lr_roc_auc']):.4f} | "
          f"PR-AUC: {np.mean(metrics['lr_pr_auc']):.4f} \u00B1 {np.std(metrics['lr_pr_auc']):.4f} | "
          f"F1: {np.mean(metrics['lr_f1']):.4f} \u00B1 {np.std(metrics['lr_f1']):.4f}")
    print(f"XGB - ROC-AUC: {np.mean(metrics['xgb_roc_auc']):.4f} \u00B1 {np.std(metrics['xgb_roc_auc']):.4f} | "
          f"PR-AUC: {np.mean(metrics['xgb_pr_auc']):.4f} \u00B1 {np.std(metrics['xgb_pr_auc']):.4f} | "
          f"F1: {np.mean(metrics['xgb_f1']):.4f} \u00B1 {np.std(metrics['xgb_f1']):.4f}")
    
    return metrics

def run_phase4():
    df = pd.read_csv("phase3_results/train_features.csv")
    print(f"Loaded {len(df)} rows.")
    
    # 0 is grounded, 1 is hallucinated
    df['target'] = (df['label'] == 'hallucinated').astype(int)
    
    # Fill any NaNs with 0 (e.g. if a layer had no features)
    df = df.fillna(0.0)
    
    h0_cols = [c for c in df.columns if 'h0' in c]
    h1_cols = [c for c in df.columns if 'h1' in c]
    
    X_0D = df[h0_cols]
    X_1D = df[h1_cols]
    X_Combined = df[h0_cols + h1_cols]
    y = df['target']
    
    train_and_evaluate(X_0D, y, "0D Features Only (Baseline)")
    train_and_evaluate(X_1D, y, "1D Features Only")
    metrics = train_and_evaluate(X_Combined, y, "Combined (0D + 1D)")
    
    # Save a little report for latex parsing if needed
    with open("phase4_results.txt", "w") as f:
        f.write(f"XGB Combined ROC-AUC: {np.mean(metrics['xgb_roc_auc']):.4f}\n")
        f.write(f"XGB Combined F1: {np.mean(metrics['xgb_f1']):.4f}\n")

if __name__ == "__main__":
    run_phase4()
