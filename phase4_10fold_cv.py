import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

def compute():
    df = pd.read_csv("phase3_results/train_features.csv")
    df['target'] = (df['label'] == 'hallucinated').astype(int)
    
    df_msp = pd.read_csv("phase4_msp_features.csv")
    df = pd.merge(df, df_msp[['example_id', 'label', 'mean_prob', 'min_prob']], on=['example_id', 'label'], how='left')
    
    df_scgpt = pd.read_csv("phase4_selfcheck_features.csv")
    df = pd.merge(df, df_scgpt[['example_id', 'label', 'selfcheck_score']], on=['example_id', 'label'], how='left')
    
    y = df['target']
    X_0d = df[[c for c in df.columns if 'h0' in c]]
    X_comb = df[[c for c in df.columns if 'h0' in c or 'h1' in c]]
    X_1d = df[[c for c in df.columns if 'h1' in c]]
    X_msp = df[['mean_prob', 'min_prob']]
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    auc_0d = []
    auc_1d = []
    auc_comb = []
    auc_msp = []
    
    for train_idx, test_idx in skf.split(X_0d, y):
        # 0D
        lr_0d = LogisticRegression(max_iter=1000)
        lr_0d.fit(X_0d.iloc[train_idx], y.iloc[train_idx])
        preds_0d = lr_0d.predict_proba(X_0d.iloc[test_idx])[:, 1]
        auc_0d.append(roc_auc_score(y.iloc[test_idx], preds_0d))
        
        # 1D
        lr_1d = LogisticRegression(max_iter=1000)
        lr_1d.fit(X_1d.iloc[train_idx], y.iloc[train_idx])
        preds_1d = lr_1d.predict_proba(X_1d.iloc[test_idx])[:, 1]
        auc_1d.append(roc_auc_score(y.iloc[test_idx], preds_1d))
        
        # Comb
        lr_comb = LogisticRegression(max_iter=1000)
        lr_comb.fit(X_comb.iloc[train_idx], y.iloc[train_idx])
        preds_comb = lr_comb.predict_proba(X_comb.iloc[test_idx])[:, 1]
        auc_comb.append(roc_auc_score(y.iloc[test_idx], preds_comb))
        
        # MSP via LR
        lr_msp = LogisticRegression(max_iter=1000)
        lr_msp.fit(X_msp.iloc[train_idx], y.iloc[train_idx])
        preds_msp = lr_msp.predict_proba(X_msp.iloc[test_idx])[:, 1]
        auc_msp.append(roc_auc_score(y.iloc[test_idx], preds_msp))
        
    t_stat_msp, p_val_msp = stats.ttest_rel(auc_0d, auc_msp)
    t_stat_comb, p_val_comb = stats.ttest_rel(auc_0d, auc_comb)
    
    print(f"10-fold CV Results:")
    print(f"0D ROC-AUC:   {np.mean(auc_0d):.4f} +/- {np.std(auc_0d):.4f}")
    print(f"1D ROC-AUC:   {np.mean(auc_1d):.4f} +/- {np.std(auc_1d):.4f}")
    print(f"MSP ROC-AUC:  {np.mean(auc_msp):.4f} +/- {np.std(auc_msp):.4f}")
    print(f"Comb ROC-AUC: {np.mean(auc_comb):.4f} +/- {np.std(auc_comb):.4f}")
    
    print(f"\nPaired t-test p-value (0D vs MSP): {p_val_msp:.5f}")
    print(f"Paired t-test p-value (0D vs Combined): {p_val_comb:.5f}")

if __name__ == "__main__":
    compute()
