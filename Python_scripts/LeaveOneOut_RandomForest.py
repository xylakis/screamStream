# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 21:09:23 2024

@author: Manos
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import scipy.stats
from datetime import datetime
from RF_defs import HP_tuning

#Affect measures
Metrics = ['Mean','Amplitude']

#Epsilon values
Thresholds = [0.05]

#Validation runs on the classifier
TIMES_TO_RUN_EXPERIMENT = 5

#results holds all averaged folds, results_all holds all folds
results = pd.DataFrame()
results_all = pd.DataFrame()

random_forest_importances = pd.DataFrame()
random_forest_best_HPs = pd.DataFrame()
significance_tests = []

room_features_cols = ['Area_size','Ceiling_height','Light_contrast','Light_levels','Light_temperature','Blocked_path','Empty_room',
                      'Hiding_place','Interior_arrangement','Triggers_present','Battery_present','Note_present','Event','Cutscene']
affect_labels_cols_short = ['Arousal_VOICE','Surprise_FACE','Fear_FACE','Surprise_UTER','Fear_UTER']

for metric in Metrics: 
    for th in Thresholds: 
        for l,label in enumerate(affect_labels_cols_short):
            path = "../Data/Random Forest data files/"+label+"_"+metric+"_inputs_targets.csv"
            df = pd.read_csv(path,index_col=0)
            df_data = df
                
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
                
            Conditions = label+"_"+metric+"_"+str(th)
            print(Conditions,current_time)
            
            acc_test_lab = 0
            acc_train_lab = 0
            f1_test_lab = 0
            f1_train_lab= 0 
            prec_test_lab = 0 
            prec_train_lab = 0 
            rec_test_lab = 0
            rec_train_lab = 0
                
            sum_importances_lab = np.zeros(len(room_features_cols), dtype=np.float64)
                
            best_val_score = 0
            significance_counter = 0 
            
            #Outer Leave-One-Out loop for Hyper-parameter tuning 
            for VALIDATION_streamer in df_data.Channel_id.unique():
        
                #define test participants
                test_streamers = df_data.Channel_id.unique()
                test_streamers = test_streamers[np.where(test_streamers != VALIDATION_streamer)]

                acc_test_pid = 0
                acc_train_pid = 0
                f1_test_pid = 0
                f1_train_pid = 0 
                prec_test_pid = 0 
                prec_train_pid = 0 
                rec_test_pid = 0
                rec_train_pid = 0
                
                sum_importances_pid = np.zeros(len(room_features_cols), dtype=np.float64)
                
                #HP TUNE
                best_hyperparameters,best_hyperparameters_ALL = HP_tuning(df_data,VALIDATION_streamer,label,room_features_cols,best_val_score)
                    
                #4. DEFINE OPTIMAL RF
                optclf = RandomForestClassifier(n_estimators = best_hyperparameters['n_estimators'], criterion=best_hyperparameters['criterion'], 
                                                min_samples_split = best_hyperparameters['min_samples_split'], min_samples_leaf=best_hyperparameters['min_samples_split'], 
                                                max_features = best_hyperparameters['max_features'], max_depth = best_hyperparameters['max_depth'], 
                                                bootstrap=best_hyperparameters['bootstrap'],random_state=None)
                
                
                for t in range(0,TIMES_TO_RUN_EXPERIMENT):
                        
                    acc_test = 0
                    acc_train = 0
                    f1_test = 0
                    f1_train = 0 
                    prec_test = 0 
                    prec_train = 0 
                    rec_test = 0
                    rec_train = 0
                    
                    Dataset_size_train = 0
                    Dataset_size_test = 0
                    
                    correct_preds = 0 
                        
                    sum_importances = np.zeros(len(room_features_cols), dtype=np.float64)
            
                    #Inner Leave-One-Out loop for train-test
                    for s,streamer in enumerate(test_streamers):
                        train_set = df_data[(df_data.Channel_id != VALIDATION_streamer) & (df_data.Channel_id != streamer)]
                            
                        X = train_set[room_features_cols]
                        y = train_set[label]
                            
                        test_set = df_data[(df_data.Channel_id == streamer)]
                        X_test = test_set[room_features_cols]
                        y_test = test_set[label]
                            
                        optclf.fit(X,y)
                        #print(optclf.feature_importances_)
                        importances = optclf.feature_importances_
                    
                        sum_importances += importances
                        
                        Dataset_size_train += len(y)
                        Dataset_size_test += len(y_test)
                        
                        # Evaluate the classifier on the test set
                        accuracy_test_noNorm = accuracy_score(y_test, optclf.predict(X_test),normalize=False)
                        correct_preds += accuracy_test_noNorm
  
                        acc_test_indiv = accuracy_score(y_test, optclf.predict(X_test))
                        prec_test_indiv =  precision_score(y_test, optclf.predict(X_test), average='macro')
                        rec_test_indiv = recall_score(y_test, optclf.predict(X_test), average='macro')
                        f1_test_indiv = f1_score(y_test, optclf.predict(X_test), average='macro')
                    
                        # Evaluate the classifier on the train set
                        acc_train_indiv = accuracy_score(y, optclf.predict(X))
                        prec_train_indiv =  precision_score(y, optclf.predict(X), average='macro')
                        rec_train_indiv = recall_score(y, optclf.predict(X), average='macro')
                        f1_train_indiv = f1_score(y, optclf.predict(X), average='macro')
                        
                        acc_test += acc_test_indiv
                        prec_test +=  prec_test_indiv
                        rec_test += rec_test_indiv
                        f1_test += f1_test_indiv
                    
                        acc_train += acc_train_indiv
                        prec_train +=  prec_train_indiv
                        rec_train += rec_train_indiv
                        f1_train += f1_train_indiv
                    
                        #Binomial Testing
                        binom_testing_high = scipy.stats.binomtest(correct_preds, n=Dataset_size_test, p=0.5)
                        
                        if binom_testing_high.pvalue < 0.05:
                            significance_counter += 1
                            
                            N_ESTIMATORS = best_hyperparameters['n_estimators']
                            CRITERION = best_hyperparameters['criterion']
                            MAX_FEATURES = best_hyperparameters['max_features']
                            MAX_DEPTH = best_hyperparameters['max_depth']
                            BOOTSTRAP = best_hyperparameters['bootstrap']
                            MIN_SAMPLES_SPLIT  = best_hyperparameters['min_samples_split']
                            MIN_SAMPLES_LEAF  = best_hyperparameters['min_samples_split']
                            
                        #store to results all 
                        new_entry = {"Affect_conditions": Conditions,
                                      "Validation":VALIDATION_streamer,
                                      "Streamer":streamer,
                                      "Accuracy_Test":round(acc_test_indiv,3),
                                      "F1_Test":round(f1_test_indiv,3),
                                      "Precision_Test":round(prec_test_indiv,3),
                                      "Recall_Test":round(rec_test_indiv,3),
                                      "Accuracy_Train":round(acc_train_indiv,3),
                                      "F1_Train":round(f1_train_indiv,3),
                                      "Precision_Train":round(prec_train_indiv,3),
                                      "Recall_Train":round(rec_train_indiv,3),
                                      "Dataset_train": len(y),
                                      "Dataset_test": len(y_test),
                                      "Correct_Test_pred": accuracy_test_noNorm,
                                      "Binom_test": True if binom_testing_high.pvalue < 0.05 else False,
                                      "Estimators":best_hyperparameters['n_estimators'],
                                      "Criterion":best_hyperparameters['criterion'],
                                      "Max_Features":best_hyperparameters['max_features'],
                                      "Max_Depth":best_hyperparameters['max_depth'],
                                      "Bootstrap":best_hyperparameters['bootstrap'],
                                      "Min_samples_split":best_hyperparameters['min_samples_split'],
                                      "Min_samples_leaf":best_hyperparameters['min_samples_split']
                                      }
                    
                        holder = pd.DataFrame.from_dict([new_entry])
                        results_all = pd.concat([results_all, holder], axis=0, ignore_index=True)
                
                        
                    acc_test_pid += round(acc_test/len(test_streamers),3)
                    acc_train_pid += round(acc_train/len(test_streamers),3)
                    f1_test_pid += round(f1_test/len(test_streamers),3)
                    f1_train_pid += round(f1_train/len(test_streamers),3)
                    prec_test_pid += round(prec_test/len(test_streamers),3) 
                    prec_train_pid += round(prec_train/len(test_streamers),3) 
                    rec_test_pid += round(rec_test/len(test_streamers),3)
                    rec_train_pid += round(rec_train/len(test_streamers),3)
                    
                    sum_importances_pid += sum_importances/(len(test_streamers))
                        
                    
                acc_test_lab += acc_test_pid/TIMES_TO_RUN_EXPERIMENT
                acc_train_lab += acc_train_pid/TIMES_TO_RUN_EXPERIMENT
                f1_test_lab += f1_test_pid/TIMES_TO_RUN_EXPERIMENT
                f1_train_lab += f1_train_pid/TIMES_TO_RUN_EXPERIMENT
                prec_test_lab += prec_test_pid/TIMES_TO_RUN_EXPERIMENT
                prec_train_lab += prec_train_pid/TIMES_TO_RUN_EXPERIMENT 
                rec_test_lab += rec_test_pid/TIMES_TO_RUN_EXPERIMENT
                rec_train_lab += rec_train_pid/TIMES_TO_RUN_EXPERIMENT
                
                sum_importances_lab += sum_importances_pid/TIMES_TO_RUN_EXPERIMENT
            
            
            print(metric,label,significance_counter)
            
            random_forest_best_HPs[Conditions] = list(best_hyperparameters_ALL.values())
            
            random_forest_importances[Conditions] = sum_importances_lab
            random_forest_importances[Conditions] = round(random_forest_importances[Conditions]/len(df_data.Channel_id.unique()),5)
            
            binom_testing = scipy.stats.binomtest(correct_preds, n=Dataset_size_test, p=0.5)
            
            if binom_testing.pvalue<=0.05:
                binom_check = True
            else:
                binom_check = False
            
            new_entry = {"Affect_conditions": Conditions, 
                          "Accuracy_Test":round(acc_test_lab/len(df_data.Channel_id.unique()),3),
                          "F1_Test":round(f1_test_lab/len(df_data.Channel_id.unique()),3),
                          "Precision_Test":round(prec_test_lab/len(df_data.Channel_id.unique()),3),
                          "Recall_Test":round(rec_test_lab/len(df_data.Channel_id.unique()),3),
                          "Accuracy_Train":round(acc_train_lab/len(df_data.Channel_id.unique()),3),
                          "F1_Train":round(f1_train_lab/len(df_data.Channel_id.unique()),3),
                          "Precision_Train":round(prec_train_lab/len(df_data.Channel_id.unique()),3),
                          "Recall_Train":round(rec_train_lab/len(df_data.Channel_id.unique()),3),
                          "Dataset_train": int(Dataset_size_train),
                          "Dataset_test": int(Dataset_size_test),
                          "Correct_Test_pred": int(correct_preds),
                          "Binom_test":binom_check}
        
            holder = pd.DataFrame.from_dict([new_entry])
            results = pd.concat([results, holder], axis=0, ignore_index=True)
                
            
    
    