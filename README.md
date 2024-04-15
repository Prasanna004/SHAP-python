# SHAP-python
#SHAP (SHapley Additive exPlanations)
#sample code to build a model to predict the feature

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
raw_df = pd.read_csv(" ") #load your data in csv format
print(raw_df)
X = raw_df.iloc[:, 2:].copy()
y = raw_df.iloc[:, 1].copy()
y.loc[y == 'unhealthy'] = 1
y.loc[y == 'healthy'] = 0
y = y.astype('int')
from xgboost import XGBClassifier
import shap
raw_params = {'subsample': 0.8, 'scale_pos_weight': 1.452991452991453, 'n_estimators': 280, 'max_depth': 4, 'gamma': 2.033333333333333, 'colsample_bytree': 0.1473684210526316}
raw_model = XGBClassifier(**raw_params)
raw_model.fit(X, y)
explainer_raw = shap.TreeExplainer(raw_model, feature_pertubation='interventional', model_output='probability', data=X)
shap_raw = explainer_raw.shap_values(X)
pathogen = raw_df.pathogen.copy()
#replace the pathogen has per your features in your data
pathogen = pathogen.replace({'feature1 5': 'virus', 'feature1': 'fungus', 'feature1': 'bacteria'})
pathogen = pathogen.str.split(' ', expand=True)
pathogen = pathogen.iloc[:, 0]
pathogen.name = 'pathogen'
print(pathogen.unique())
shap_df = pd.DataFrame(shap_raw, columns=X.columns)
shap_df = pd.concat([pathogen, shap_df], axis = 1)
print(shap_df)
shap_df = shap_df.loc[shap_df.pathogen != 'none', :]
X_filt = X.loc[raw_df.pathogen != 'none', :]
# Loop through all confirmed and get mean absolute shap value
mean_vals = []
genus_names = []
abundance_list = []
for p in shap_df.pathogen.unique():
    values = shap_df.loc[shap_df.pathogen == p, p]
    abundances = X_filt.loc[shap_df.pathogen == p, p]
    mean_vals = mean_vals + list(values)
    abundance_list = abundance_list + list(abundances)
    genus_names = genus_names + [p] * len(values)
plot_df = pd.DataFrame({'Genus': genus_names, 'shap': mean_vals, 'abundance': abundance_list})
print(plot_df)
plot_df.to_csv("outdir", index=False, header=True)
raw_CR = X[['data1','data2','data3','data4','data5','data6','data7','data8','data9','data10','data11','data12','data13']].copy()
raw_CR_params = {'subsample': 0.8, 'scale_pos_weight': 1.452991452991453, 'n_estimators': 240, 'max_depth': 1, 'gamma': 1.711111111111111, 'colsample_bytree': 0.9052631578947369}
raw_CR_model = XGBClassifier(**raw_CR_params)
raw_CR_model.fit(raw_CR, y)
explainer_CR = shap.TreeExplainer(raw_CR_model, feature_pertubation='interventional', model_output='probability', data=raw_CR)
shap_CR = explainer_CR.shap_values(raw_CR)
j = 202 #can modify 
print(f'Actual Classification {y[j]}')
print(raw_CR.index[j])

shap.summary_plot(shap_pre, raw_CR, show=False, plot_size=(4, 5), color_bar_label='Read Count', max_display=25)
plt.show()
