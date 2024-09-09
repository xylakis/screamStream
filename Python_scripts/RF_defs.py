# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:02:09 2024

@author: Manolis
"""

from sklearn.ensemble import RandomForestClassifier

def HP_tuning(df_data,VALIDATION_streamer,label,room_features_cols,best_val_score):
    
    #1. Create VALIDATION and TRAINING-TEST set
    val_set = df_data[df_data.Channel_id == VALIDATION_streamer]
    y_val = val_set[label]
    X_val = val_set[room_features_cols]
    
    train_set = df_data[(df_data.Channel_id != VALIDATION_streamer)]
    X = train_set[room_features_cols]
    y = train_set[label]

    # #3. HYPERPARAMETER TUNING FOR RF
    n_estimators = [50,100,200] # number of trees in the random forest
    criterion = ['gini','entropy','log_loss']
    max_features = ['sqrt','log2',None] # number of features in consideration at every split
    max_depth = [3,7,11] # maximum number of levels allowed in each decision tree
    min_samples_split = [2, 6, 10] # minimum sample number to split a node
    min_samples_leaf = [9]
    bootstrap = [True]
        
    param_grid = {'n_estimators': n_estimators, 'criterion':criterion,'max_features': max_features, 'min_samples_split':min_samples_split ,
                  'min_samples_leaf':min_samples_leaf,'max_depth': max_depth, 'bootstrap':bootstrap}
        
    best_score = 0
    
    #for time_val in range(0,TIMES_TO_RUN_VALIDATION): 
    for n_estimators in param_grid['n_estimators']:
        for criterion in param_grid['criterion']:
            for max_depth in param_grid['max_depth']:
                for max_features in param_grid['max_features']:
                    for bootstrap in param_grid['bootstrap']:
                        for min_samples_split in param_grid['min_samples_split']:
                            for min_samples_leaf in param_grid['min_samples_leaf']:
                                # Train model with current hyperparameters
                                rf = RandomForestClassifier(n_estimators=n_estimators,
                                                            criterion=criterion,
                                                            max_depth=max_depth,
                                                            min_samples_split=min_samples_split,
                                                            min_samples_leaf=min_samples_leaf,
                                                            max_features=max_features,
                                                            bootstrap = bootstrap,
                                                            random_state=None)
                                rf.fit(X, y.ravel())
                                #Evaluate performance on training set
                                score = rf.score(X_val, y_val.ravel())
                            
                            # Update best hyperparameters if score is better
                            if score > best_score:
                                #print(score,n_estimators,max_depth,max_features,bootstrap,MIN_SAMPLES_SPLIT,MIN_SAMPLES_LEAF,len(my_features))
                                best_hyperparameters = {'n_estimators': n_estimators,
                                                        'criterion': criterion,
                                                        'max_depth': max_depth,
                                                        'max_features': max_features,
                                                        'bootstrap': bootstrap,
                                                        'min_samples_split': min_samples_split,
                                                        'min_samples_leaf': min_samples_leaf
                                                        }
                                best_score = score
                            
                            if score > best_val_score:
                                best_hyperparameters_ALL = {'n_estimators': n_estimators,
                                                        'criterion': criterion,
                                                        'max_depth': max_depth,
                                                        'max_features': max_features,
                                                        'bootstrap': bootstrap,
                                                        'min_samples_split': min_samples_split,
                                                        'min_samples_leaf': min_samples_leaf
                                                        }
                                best_val_score = score
                            
    return best_hyperparameters,best_hyperparameters_ALL
                            
                            