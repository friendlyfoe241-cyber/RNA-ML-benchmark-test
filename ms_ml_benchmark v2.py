import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import gzip
import shap
import scipy.stats as stats
from collections import Counter
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, GridSearchCV,
                                      learning_curve)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, brier_score_loss)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import RFE
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


print("="*60)
print("MS BLOOD RNA — PhD LEVEL ML BENCHMARK")
print("="*60)


# ── 1. LOAD GEO DATASET ───────────────────────────────────────
print("\n[1/9] Loading GEO dataset (GSE17048)...")
with gzip.open('GSE17048_series_matrix.txt.gz', 'rt') as f:
    lines = f.readlines()


label_line = lines[33]
labels = label_line.strip().split('\t')[1:]
labels = [l.strip('"') for l in labels]


conditions = []
for label in labels:
    if 'healthy' in label.lower(): conditions.append('HC')
    elif 'RR' in label: conditions.append('RR')
    elif 'PP' in label: conditions.append('PP')
    elif 'SP' in label: conditions.append('SP')
    else: conditions.append('Other')


data_start = 0
for i, line in enumerate(lines):
    if '!series_matrix_table_begin' in line:
        data_start = i + 1
        break


data_rows, gene_ids = [], []
for line in lines[data_start:]:
    if '!series_matrix_table_end' in line: break
    if line.startswith('"ID_REF"'): continue
    parts = line.strip().split('\t')
    gene_id = parts[0].strip('"')
    try:
        values = [float(x) for x in parts[1:]]
        if len(values) == len(conditions):
            gene_ids.append(gene_id)
            data_rows.append(values)
    except: continue


df_geo = pd.DataFrame(data_rows, index=gene_ids,
                       columns=conditions).T.reset_index()
df_geo = df_geo.rename(columns={'index': 'Condition'})
print(f"Loaded: {len(gene_ids)} genes x {len(conditions)} patients")
print(f"Classes: {Counter(conditions)}")


# ── 2. LOAD ITALIAN DATASET (for cross-dataset validation) ────
print("\n[2/9] Loading Italian dataset for cross-dataset validation...")
try:
    df_it = pd.read_csv("RRvsHC_-F2-LogTPM_train_lessFeatures_107rep.csv", sep=';')
    df_it_t = df_it.set_index('Condition').T.reset_index()
    df_it_t = df_it_t.rename(columns={'index': 'Patient'})
    for col in df_it_t.columns[1:]:
        df_it_t[col] = df_it_t[col].astype(str).str.strip().str.replace(',','.').astype(float)
    df_it_t = df_it_t[df_it_t['Patient'].str.startswith(('RR','HC'))]
    df_it_t['Label'] = df_it_t['Patient'].str.startswith('RR').astype(int)
    print(f"Italian dataset: {len(df_it_t)} patients")
except FileNotFoundError:
    print("Italian dataset not found - skipping cross-dataset validation")
    df_it_t = None


# ── 3. FEATURE SELECTION ─────────────────────────────────────
print("\n[3/9] Feature selection (Variance + RFE)...")
gene_cols = [c for c in df_geo.columns if c != 'Condition']
variances = df_geo[gene_cols].var()
top_1000 = variances.nlargest(1000).index.tolist()


df_geo['Label'] = (df_geo['Condition'] != 'HC').astype(int)
X_rfe = df_geo[top_1000].values
y_rfe = df_geo['Label'].values
scaler_rfe = StandardScaler()
X_rfe_scaled = scaler_rfe.fit_transform(X_rfe)
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42),
          n_features_to_select=200, step=50)
rfe.fit(X_rfe_scaled, y_rfe)
selected_genes = [top_1000[i] for i, s in enumerate(rfe.support_) if s]
print(f"Selected {len(selected_genes)} genes")


# ── 4. PREPARE DATA WITH SMOTE ────────────────────────────────
print("\n[4/9] Preparing data with SMOTE...")
X = df_geo[selected_genes].values
y = df_geo['Label'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
print(f"Training: {len(X_train)}, Testing: {len(X_test)}")


# ── 5. HYPERPARAMETER TUNING ──────────────────────────────────
print("\n[5/9] Hyperparameter tuning...")
param_grids = {
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {'n_estimators':[100,200],'max_depth':[None,10,20],
         'min_samples_split':[2,5]}),
    'XGBoost': (
        XGBClassifier(random_state=42, eval_metric='logloss'),
        {'n_estimators':[100,200],'max_depth':[3,5,7],
         'learning_rate':[0.05,0.1],'subsample':[0.8,1.0]}),
    'SVM': (
        SVC(probability=True, random_state=42),
        {'C':[0.1,1,10],'kernel':['rbf','linear'],'gamma':['scale','auto']}),
    'Logistic Regression': (
        LogisticRegression(max_iter=1000, random_state=42),
        {'C':[0.01,0.1,1,10],'solver':['lbfgs','liblinear']}),
    'Neural Network': (
        MLPClassifier(max_iter=1000, random_state=42),
        {'hidden_layer_sizes':[(100,),(100,50),(200,100)],
         'alpha':[0.0001,0.001],'learning_rate':['constant','adaptive']}),
    'KNN': (
        KNeighborsClassifier(),
        {'n_neighbors':[3,5,7,11],'weights':['uniform','distance']}),
    'Gradient Boosting': (
        GradientBoostingClassifier(random_state=42),
        {'n_estimators':[100,200],'learning_rate':[0.05,0.1],
         'max_depth':[3,5]}),
    'Naive Bayes': (
        GaussianNB(),
        {'var_smoothing':[1e-9,1e-8,1e-7]}),
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tuned_models = {}
for name, (model, params) in param_grids.items():
    print(f"  Tuning {name}...")
    grid = GridSearchCV(model, params, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    tuned_models[name] = grid.best_estimator_
    print(f"  Best: {grid.best_params_}")


# ── 6. ENSEMBLE STACKING ─────────────────────────────────────
print("\n[6/9] Building ensemble stacking model...")
estimators = [
    ('rf',  tuned_models['Random Forest']),
    ('xgb', tuned_models['XGBoost']),
    ('svm', tuned_models['SVM']),
    ('lr',  tuned_models['Logistic Regression']),
    ('nn',  tuned_models['Neural Network']),
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5, passthrough=False)
stack.fit(X_train, y_train)
tuned_models['Stacking Ensemble'] = stack
print("Stacking ensemble built!")


# ── 7. BENCHMARK ─────────────────────────────────────────────
print("\n[7/9] Running full benchmark...")


def bootstrap_ci(y_true, y_pred, y_proba, metric_fn,
                 n_bootstrap=1000, ci=95):
    scores = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        try:
            score = (roc_auc_score(y_true[idx], y_proba[idx])
                     if metric_fn == roc_auc_score
                     else metric_fn(y_true[idx], y_pred[idx]))
            scores.append(score)
        except: continue
    lo = np.percentile(scores, (100-ci)/2)
    hi = np.percentile(scores, 100-(100-ci)/2)
    return lo, hi


results = {}
roc_data = {}
all_cv_scores = {}


for name, model in tuned_models.items():
    preds  = model.predict(X_test)
    proba  = model.predict_proba(X_test)[:,1]
    cv_sc  = cross_val_score(model, X_res, y_res, cv=cv,
                              scoring='accuracy')
    acc_ci = bootstrap_ci(y_test, preds, proba, accuracy_score)
    auc_ci = bootstrap_ci(y_test, preds, proba, roc_auc_score)
    all_cv_scores[name] = cv_sc
    results[name] = {
        'Accuracy':    round(accuracy_score(y_test, preds)*100, 2),
        'Precision':   round(precision_score(y_test, preds)*100, 2),
        'Recall':      round(recall_score(y_test, preds)*100, 2),
        'F1':          round(f1_score(y_test, preds)*100, 2),
        'AUC-ROC':     round(roc_auc_score(y_test, proba), 4),
        'CV Acc':      round(cv_sc.mean()*100, 2),
        'CV Std':      round(cv_sc.std()*100, 2),
        'Acc CI':      f"[{acc_ci[0]*100:.1f}-{acc_ci[1]*100:.1f}]",
        'AUC CI':      f"[{auc_ci[0]:.4f}-{auc_ci[1]:.4f}]",
        'Brier':       round(brier_score_loss(y_test, proba), 4),
    }
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_data[name] = (fpr, tpr, results[name]['AUC-ROC'])
    print(f"  {name}: Acc={results[name]['Accuracy']}% "
          f"AUC={results[name]['AUC-ROC']} "
          f"Brier={results[name]['Brier']}")


results_df = pd.DataFrame(results).T.sort_values('AUC-ROC',
                                                   ascending=False)
results_df.to_csv('phd_benchmark_results.csv')
print("\nFull Results:")
print(results_df.to_string())


# Statistical significance
print("\nStatistical significance (Wilcoxon)...")
best_name = results_df.index[0]
best_cv   = all_cv_scores[best_name]
sig_results = {}
for name, scores in all_cv_scores.items():
    if name == best_name: continue
    try:
        _, p = stats.wilcoxon(best_cv, scores)
    except:
        p = 1.0
    sig_results[name] = round(p, 4)
    print(f"  {best_name} vs {name}: p={p:.4f} "
          f"({'SIG' if p<0.05 else 'ns'})")


# ── 8. CROSS-DATASET VALIDATION ──────────────────────────────
print("\n[8/9] Cross-dataset validation...")
if df_it_t is not None:
    # Find common genes between GEO and Italian datasets
    it_genes = [c for c in df_it_t.columns
                if c not in ['Patient','Label']]
    common = [g for g in selected_genes if g in it_genes]
    print(f"Common genes between datasets: {len(common)}")


    cross_results = {}
    if len(common) >= 10:
        X_cross = df_it_t[common].values
        y_cross = df_it_t['Label'].values
        X_cross_scaled = scaler.transform(
            np.hstack([X_cross,
                       np.zeros((len(X_cross),
                                 len(selected_genes)-len(common)))]))
        for name in ['Logistic Regression','SVM','Random Forest',
                     'XGBoost','Neural Network']:
            try:
                model = tuned_models[name]
                preds = model.predict(X_cross_scaled)
                proba = model.predict_proba(X_cross_scaled)[:,1]
                cross_results[name] = {
                    'Cross Accuracy': round(
                        accuracy_score(y_cross, preds)*100, 2),
                    'Cross AUC': round(
                        roc_auc_score(y_cross, proba), 4),
                }
                print(f"  {name}: Acc={cross_results[name]['Cross Accuracy']}% "
                      f"AUC={cross_results[name]['Cross AUC']}")
            except Exception as e:
                print(f"  {name}: failed ({e})")
    else:
        print("  Not enough common genes — skipping cross-dataset test")
else:
    print("  Skipped (Italian dataset not available)")


# ── 9. GRAPHS ─────────────────────────────────────────────────
print("\n[9/9] Generating graphs...")
colors = ['#e74c3c','#2ecc71','#3498db','#f39c12',
          '#9b59b6','#1abc9c','#e67e22','#2c3e50','#e91e63']


# Graph 1 — ROC Curves
plt.figure(figsize=(11,8))
for (name,(fpr,tpr,auc)),col in zip(roc_data.items(), colors):
    lw = 3 if name == 'Stacking Ensemble' else 2
    ls = '-' if name == 'Stacking Ensemble' else '-'
    plt.plot(fpr, tpr, color=col, linewidth=lw,
             label=f'{name} (AUC={auc})',
             linestyle='--' if name=='Stacking Ensemble' else '-')
plt.plot([0,1],[0,1],'k--',linewidth=1,label='Random Classifier')
plt.fill_between([0,1],[0,1],alpha=0.05,color='gray')
plt.xlabel('False Positive Rate',fontsize=12)
plt.ylabel('True Positive Rate',fontsize=12)
plt.title('ROC Curves — PhD Level ML Benchmark\n'
          'MS vs HC (GSE17048, RFE+SMOTE+GridSearchCV+Stacking)',
          fontsize=11)
plt.legend(loc='lower right',fontsize=8)
plt.tight_layout()
plt.savefig('phd_roc_curves.png',dpi=300)
plt.show()


# Graph 2 — Accuracy with CI
names   = results_df.index.tolist()
accs    = results_df['Accuracy'].tolist()
ci_strs = results_df['Acc CI'].tolist()
ci_lo   = [float(s.strip('[]').split('-')[0]) for s in ci_strs]
ci_hi   = [float(s.strip('[]').split('-')[1]) for s in ci_strs]
x = np.arange(len(names))
plt.figure(figsize=(13,7))
bar_cols = ['#e74c3c' if 'Stack' in n else '#3498db' for n in names]
bars = plt.bar(x, accs, color=bar_cols, alpha=0.85, width=0.6)
for i,(lo,hi,acc) in enumerate(zip(ci_lo,ci_hi,accs)):
    plt.errorbar(x[i],acc,yerr=[[acc-lo],[hi-acc]],
                 fmt='none',color='black',capsize=5,linewidth=2)
plt.xlabel('Model',fontsize=11)
plt.ylabel('Accuracy (%)',fontsize=11)
plt.title('Model Accuracy with 95% Confidence Intervals\n'
          '(Bootstrap n=1000, red=Stacking Ensemble)',fontsize=11)
plt.xticks(x,names,rotation=20,ha='right')
plt.ylim(50,118)
for bar,acc in zip(bars,accs):
    plt.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+1.5,
             f'{acc}%',ha='center',va='bottom',
             fontsize=8,fontweight='bold')
plt.tight_layout()
plt.savefig('phd_accuracy_ci.png',dpi=300)
plt.show()


# Graph 3 — Comprehensive Heatmap
metrics_cols = ['Accuracy','Precision','Recall','F1','AUC-ROC','CV Acc']
hm = results_df[metrics_cols].copy()
hm['AUC-ROC'] = hm['AUC-ROC']*100
plt.figure(figsize=(13,7))
im = plt.imshow(hm.values.astype(float),
                cmap='RdYlGn',aspect='auto',vmin=50,vmax=100)
plt.colorbar(im,label='Score')
plt.xticks(range(len(metrics_cols)),
           ['Accuracy','Precision','Recall','F1',
            'AUC x100','CV Acc'],fontsize=9)
plt.yticks(range(len(hm.index)),hm.index,fontsize=9)
plt.title('Comprehensive Performance Heatmap\n'
          '(8 Models + Stacking Ensemble, 6 Metrics)',fontsize=11)
for i in range(len(hm.index)):
    for j in range(len(metrics_cols)):
        plt.text(j,i,f'{hm.values[i,j]:.1f}',
                 ha='center',va='center',
                 fontsize=8,fontweight='bold')
plt.tight_layout()
plt.savefig('phd_heatmap.png',dpi=300)
plt.show()


# Graph 4 — UMAP Visualisation
print("  Building UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
X_umap = reducer.fit_transform(X_scaled)
plt.figure(figsize=(10,7))
scatter_colors = ['#e74c3c' if l==1 else '#3498db' for l in y]
plt.scatter(X_umap[:,0],X_umap[:,1],
            c=scatter_colors,alpha=0.7,s=60,edgecolors='white',linewidth=0.5)
legend_elements = [
    Patch(facecolor='#e74c3c',label='MS Patient'),
    Patch(facecolor='#3498db',label='Healthy Control')
]
plt.legend(handles=legend_elements,fontsize=11)
plt.xlabel('UMAP Dimension 1',fontsize=11)
plt.ylabel('UMAP Dimension 2',fontsize=11)
plt.title('UMAP Dimensionality Reduction\n'
          'Blood RNA Expression: MS vs Healthy Controls (n=144)',
          fontsize=11)
plt.tight_layout()
plt.savefig('phd_umap.png',dpi=300)
plt.show()


# Graph 5 — Learning Curves
print("  Building learning curves...")
best_model = tuned_models[best_name]
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_res, y_res,
    train_sizes=np.linspace(0.1,1.0,10),
    cv=5, scoring='accuracy', n_jobs=-1)
plt.figure(figsize=(10,6))
plt.plot(train_sizes,train_scores.mean(axis=1)*100,
         'o-',color='#e74c3c',label='Training Accuracy',linewidth=2)
plt.fill_between(train_sizes,
                 (train_scores.mean(axis=1)-train_scores.std(axis=1))*100,
                 (train_scores.mean(axis=1)+train_scores.std(axis=1))*100,
                 alpha=0.15,color='#e74c3c')
plt.plot(train_sizes,val_scores.mean(axis=1)*100,
         'o-',color='#3498db',label='Validation Accuracy',linewidth=2)
plt.fill_between(train_sizes,
                 (val_scores.mean(axis=1)-val_scores.std(axis=1))*100,
                 (val_scores.mean(axis=1)+val_scores.std(axis=1))*100,
                 alpha=0.15,color='#3498db')
plt.xlabel('Training Set Size',fontsize=11)
plt.ylabel('Accuracy (%)',fontsize=11)
plt.title(f'Learning Curves — {best_name}\n'
          '(Shaded area = ±1 std dev)',fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('phd_learning_curves.png',dpi=300)
plt.show()


# Graph 6 — Calibration Curves
print("  Building calibration curves...")
plt.figure(figsize=(10,8))
plt.plot([0,1],[0,1],'k--',label='Perfect Calibration',linewidth=1.5)
cal_colors = colors[:len(tuned_models)]
for (name,model),col in zip(tuned_models.items(),cal_colors):
    try:
        proba = model.predict_proba(X_test)[:,1]
        fraction_pos, mean_pred = calibration_curve(
            y_test, proba, n_bins=8)
        plt.plot(mean_pred, fraction_pos, 'o-',
                 color=col, linewidth=2, label=name)
    except: continue
plt.xlabel('Mean Predicted Probability',fontsize=11)
plt.ylabel('Fraction of Positives',fontsize=11)
plt.title('Calibration Curves — All Models\n'
          '(Closer to diagonal = better calibrated)',fontsize=11)
plt.legend(loc='upper left',fontsize=8)
plt.tight_layout()
plt.savefig('phd_calibration.png',dpi=300)
plt.show()


# Graph 7 — Statistical Significance
plt.figure(figsize=(10,5))
sig_names = list(sig_results.keys())
p_vals    = list(sig_results.values())
bar_cols2 = ['#e74c3c' if p<0.05 else '#95a5a6' for p in p_vals]
plt.barh(sig_names,p_vals,color=bar_cols2,alpha=0.85)
plt.axvline(x=0.05,color='black',linestyle='--',
            linewidth=1.5,label='p=0.05 threshold')
plt.xlabel('p-value (Wilcoxon Signed-Rank Test)',fontsize=11)
plt.title(f'Statistical Significance: {best_name} vs All Models\n'
          '(Red = p<0.05, statistically significant)',fontsize=10)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig('phd_significance.png',dpi=300)
plt.show()


# Graph 8 — Brier Scores
plt.figure(figsize=(10,5))
brier_names  = results_df.index.tolist()
brier_scores = results_df['Brier'].tolist()
brier_cols   = ['#e74c3c' if s<0.1 else
                '#f39c12' if s<0.2 else '#95a5a6'
                for s in brier_scores]
bars = plt.bar(brier_names, brier_scores,
               color=brier_cols, alpha=0.85)
plt.axhline(y=0.1,color='green',linestyle='--',
            linewidth=1.5,label='Excellent threshold (0.1)')
plt.axhline(y=0.25,color='orange',linestyle='--',
            linewidth=1.5,label='Acceptable threshold (0.25)')
plt.xlabel('Model',fontsize=11)
plt.ylabel('Brier Score (lower = better)',fontsize=11)
plt.title('Brier Scores — Probability Calibration Quality\n'
          '(Red = excellent, orange = good, grey = poor)',fontsize=11)
plt.xticks(rotation=20,ha='right')
plt.legend(fontsize=9)
for bar,score in zip(bars,brier_scores):
    plt.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.002,
             f'{score}',ha='center',va='bottom',fontsize=9)
plt.tight_layout()
plt.savefig('phd_brier_scores.png',dpi=300)
plt.show()


print("\n"+"="*60)
print("PhD BENCHMARK COMPLETE")
print("="*60)
print(f"Best Model:    {best_name}")
print(f"Best AUC-ROC:  {results_df['AUC-ROC'].iloc[0]}")
print(f"Best Accuracy: {results_df['Accuracy'].iloc[0]}%")
print(f"Best Brier:    {results_df['Brier'].iloc[0]}")
print("\nFiles saved:")
for f in ['phd_roc_curves','phd_accuracy_ci','phd_heatmap',
          'phd_umap','phd_learning_curves',
          'phd_calibration','phd_significance','phd_brier_scores']:
    print(f"  {f}.png")
print("  phd_benchmark_results.csv")

