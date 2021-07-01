#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script .

The code is developed and teset with Python 3.7, scikit-learn 0.24.2
'''
import numpy as np
import pandas as pd
import os, argparse, logging, csv, re

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2020~2022, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2021-06-30'

# Parameters
FV_NAMES = ['PCA', 'CAE', 'CVAE', 'Pretrained-BigEarth', 'Pretrained-ImageNet']
FV_FILES = ['fv_pca.zip', 'fv_cae.zip', 'fv_cvae.zip', 'fv_ptbe.zip', 'fv_ptin.zip']
EVENT_NAMES = []
EVENT_FILE = 'tad_filtered.csv'
CV_FOLD = 10


# Read events
def read_events(file, verbose=0):
    events = pd.read_csv(file, index_col=0)
    if verbose>0:
        logging.info(events.shape)
        for c in events.columns:
            logging.info(c + '\t counts: ' + str(events[c].sum()) + '\t prob:' + str(events[c].sum()/events.shape[0]))
    return(events)

# Function to give report for binary classifications
def evaluate_binary(yt, yp, id=None, ythresh=1.):
    from sklearn.metrics import confusion_matrix
    ytb = (yt>=ythresh)*1
    ypb = (yp>=ythresh)*1
    # Derive metrics
    output = {'id':id}
    TN, FP, FN, TP = confusion_matrix(ytb, ypb).ravel()
    output['true_positive'] = np.round(TP,2)
    output['false_positive'] = np.round(FP,2)
    output['false_negative'] = np.round(FN,2)
    output['true_negative'] = np.round(TN,2)
    output['sensitivity'] = np.round(TP/(TP+FN),2)
    output['specificity'] = np.round(TN/(FP+TN),2)
    output['prevalence'] = np.round((TP+FN)/(FN+TP+FP+TN),8)
    output['ppv'] = np.round(TP/(TP+FP),4)
    output['npv'] = np.round(TN/(TN+FN),4)
    output['fpr'] = np.round(FP/(FP+TN),4)
    output['threat_score'] = np.round(TP/(TP+FP+FN),4)
    output['accuracy'] = np.round((TP+TN)/(FN+TP+FP+TN),4)
    output['balanced_accuracy'] = np.round((TP/(TP+FN) + TN/(FP+TN))*0.5,4)
    output['F1'] = np.round(2*TP/(2*TP+FP+FN),4)
    output['MCC'] = np.round((TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)),4)
    output['informedness'] = np.round(output['sensitivity'] + output['specificity'] - 1,4)
    output['markedness'] = np.round(output['ppv'] + output['npv'] -1,4)
    return(output)

# Clean up NAs
def clean_nans(x, y):
    # Merge x and y by index
    data = pd.merge(y, x, left_index=True, right_index=True)
    nrow_original = data.shape[0]
    # Drop NaNs
    data = data.dropna(axis=0, how='any')
    nrow_dropna = data.shape[0]
    # Retrieve original index as dates and reset
    dates = data.index
    data = data.reset_index(drop=True)
    print('Dropping rows containing NaNs: ' + str(nrow_original - nrow_dropna))
    y = data.iloc[:,0]
    x = data.iloc[:,1:]
    return((x, y, dates))

# Create cross validation folds
def create_cv_folds(x, y, kfold=5, randseed=123):
    from sklearn.model_selection import StratifiedKFold, KFold
    # clean up nans
    x, y, dates = clean_nans(x, y)
    # Create CV
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=randseed)
    splits = skf.split(x, y)
    # 
    return((splits, x, y, dates))

# Evaluate one FV-Event pair
def evaluate_fv_event_pair(fv, event, fv_id=None, kfold=5, randseed=123):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, cross_validate
    # Create CV folds
    x, y, dates = clean_nans(x=fv, y=event)
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=randseed)
    clf = LogisticRegression(dual=True,
                             C=1.0,
                             class_weight='balanced', 
                             random_state=0,
                             solver='liblinear',
                             max_iter=10000)
    # Get predictions from CV
    y_pred = cross_val_predict(clf, x, y, cv=skf)
    eval_all = evaluate_binary(y, y_pred, id=fv_id)
    #print(eval_all)
    # Get evaluations for each CV fold
    eval_cv = cross_validate(clf, x, y, cv=skf, return_train_score=True,
                             scoring=['accuracy', 'recall', 'precision', 'roc_auc', 'f1'])
    for k in eval_cv.keys():
        eval_all[k+'_mean'] = np.round(np.mean(eval_cv[k]),4)
        eval_all[k+'_std'] = np.round(np.std(eval_cv[k]),4)
    #print(eval_cv)
    return(eval_all, eval_cv)


### Main Script ###
# Parameters
FV_NAMES = ['PCA', 'CAE', 'CVAE', 'Pretrained-BigEarth', 'Pretrained-ImageNet']
FV_FILES = ['fv_pca.zip', 'fv_cae.zip', 'fv_cvae.zip', 'fv_ptbe.zip', 'fv_ptin.zip']
EVENT_NAMES = ['CS', 'TYW', 'NWPTY', 'FT', 'NE', 'SWF', 'HRD', 'HRH']
EVENT_FILE = 'tad_filtered.csv'
CV_FOLD = 10
DATAPATH = '../data/'

events = read_events(DATAPATH+EVENT_FILE, verbose=1)
results = []
cv_details = {}

# Loop
for i in range(len(FV_NAMES)):
    for j in range(len(EVENT_NAMES)):
        fv = pd.read_csv(DATAPATH+FV_FILES[i], compression='zip', index_col=0)
        fv_name = FV_NAMES[i]
        event_name = EVENT_NAMES[j]
        exp_id = fv_name+'-'+event_name
        print(exp_id)
        eval_all, eval_cv = evaluate_fv_event_pair(fv, events[event_name], fv_id=exp_id, kfold=5)
        results.append(eval_all)
        cv_details[exp_id] = eval_cv

pd.DataFrame(results).to_csv('exp_results.csv', index=False)
