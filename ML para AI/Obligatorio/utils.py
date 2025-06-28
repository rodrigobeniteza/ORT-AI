import pandas as py
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.image import PatchExtractor
from skimage.transform import resize
from sklearn.base import BaseEstimator # Para tipos de modelos base

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from skimage import feature
from itertools import chain
from sklearn.neural_network import MLPClassifier
   


def optimize_model_and_faces_with_optuna(
    positive_patches, 
    negative_patches,
    model_class,
    param_distributions,
    n_trials=100,
    cv_folds=5,
    test_size=0.2,
    random_state=42,
    n_jobs=-1,
    show_progress=True,
    optimize_faces=True,
    optimize_pca=True,
    fixed_pca_params=None
):
    """
    Comprehensive and model-agnostic Optuna optimization function.
    
    Parameters:
    -----------
    positive_patches : array-like
        Array of positive face patches
    negative_patches : array-like  
        Array of negative background patches
    model_class : class
        The model class to instantiate (e.g., xgb.XGBClassifier, GradientBoostingClassifier)
    param_distributions : dict
        Dictionary defining parameter search spaces using Optuna suggest methods.
        Format: {
            'param_name': {
                'type': 'int'|'float'|'categorical'|'uniform'|'loguniform'|'tuple',
                'low': value,      # for int/float
                'high': value,     # for int/float  
                'step': value,     # optional for int
                'log': bool,       # optional for float
                'choices': list    # for categorical
                # For tuple:
                'length': int,     # length of the tuple
                'elements': {      # param config for each element (e.g., {'type': 'int', ...})
                    ...
                }
            }
        }
    n_trials : int, default=100
        Number of optimization trials
    cv_folds : int, default=5
        Number of cross-validation folds
    test_size : float, default=0.2
        Proportion of data for validation
    random_state : int, default=42
        Random state for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs
    show_progress : bool, default=True
        Whether to show progress bar
    optimize_faces : bool, default=True
        Whether to optimize the number of positive samples
    optimize_pca : bool, default=True
        Whether to optimize PCA parameters
    fixed_pca_params : dict, optional
        Fixed PCA parameters if optimize_pca=False
        
    Returns:
    --------
    dict: Dictionary containing optimization results
    """
    

    def objective(trial):
        """
        Model-agnostic objective function
        """
        model_params = {}
        
        # 1. OPTIMIZE MODEL PARAMETERS (completely flexible)
        for param_name, param_config in param_distributions.items():
            param_type = param_config['type']
            
            if param_type == 'int':
                model_params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high'],
                    step=param_config.get('step', 1)
                )
            elif param_type == 'float':
                if param_config.get('log', False):
                    model_params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=True
                    )
                else:
                    model_params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
            elif param_type == 'categorical':
                model_params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
            elif param_type == 'uniform':
                model_params[param_name] = trial.suggest_uniform(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            elif param_type == 'loguniform':
                model_params[param_name] = trial.suggest_loguniform(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            elif param_type == 'tuple':
                tuple_length = param_config['length']
                tuple_elements = []
                for i in range(tuple_length):
                    elem_type = param_config['elements']['type']
                    elem_name = f"{param_name}_{i}"
                    if elem_type == 'int':
                        elem = trial.suggest_int(
                            elem_name,
                            param_config['elements']['low'],
                            param_config['elements']['high'],
                            step=param_config['elements'].get('step', 1)
                        )
                    elif elem_type == 'float':
                        if param_config['elements'].get('log', False):
                            elem = trial.suggest_float(
                                elem_name,
                                param_config['elements']['low'],
                                param_config['elements']['high'],
                                log=True
                            )
                        else:
                            elem = trial.suggest_float(
                                elem_name,
                                param_config['elements']['low'],
                                param_config['elements']['high']
                            )
                    elif elem_type == 'categorical':
                        elem = trial.suggest_categorical(
                            elem_name,
                            param_config['elements']['choices']
                        )
                    else:
                        raise ValueError(f"Unsupported tuple element type: {elem_type}")
                    tuple_elements.append(elem)
                model_params[param_name] = tuple(tuple_elements)
        
        # --- Handle MLPClassifier special case for hidden_layer_sizes ---
        if model_class == MLPClassifier or (hasattr(model_class, '__name__') and model_class.__name__ == 'MLPClassifier'):
            n_layers = model_params.pop('n_layers', None)
            # Only build hidden_layer_sizes if n_layers is present
            if n_layers is not None:
                hidden_layers = []
                for i in range(n_layers):
                    units = model_params.pop(f'n_units_l{i}', None)
                    if units is not None:
                        hidden_layers.append(units)
                model_params['hidden_layer_sizes'] = tuple(hidden_layers)
            # Remove any stray n_units_l{i} keys
            for key in list(model_params.keys()):
                if key.startswith('n_units_l'):
                    model_params.pop(key)
        
        # Add fixed parameters (like random_state, n_jobs if needed)
        if 'random_state' not in model_params and hasattr(model_class(), 'random_state'):
            model_params['random_state'] = random_state
        if 'n_jobs' not in model_params and hasattr(model_class(), 'n_jobs'):
            model_params['n_jobs'] = n_jobs
            
        # 2. OPTIMIZE NUMBER OF POSITIVE SAMPLES (P) if requested
        if optimize_faces:
            max_positive = len(positive_patches)
            min_positive = min(1000, max_positive // 4)
            
            n_positive_samples = trial.suggest_int(
                'n_positive_samples', 
                min_positive, 
                max_positive, 
                step=500
            )
        else:
            n_positive_samples = len(positive_patches)
        
        # 3. OPTIMIZE PCA PARAMETERS if requested
        if optimize_pca:
            pca_params = {
                'n_components': trial.suggest_int('pca_n_components', 50, 800, step=50),
                'whiten': trial.suggest_categorical('pca_whiten', [True, False])
            }
        else:
            pca_params = fixed_pca_params or {'n_components': 500, 'whiten': True}
        
        # 4. DATA PREPARATION
        try:
            # Select subset of positive patches
            current_positive_patches = positive_patches[:n_positive_samples]
            
            # Extract HOG features
            X_full = np.array([
                feature.hog(im) for im in chain(current_positive_patches, negative_patches)
            ])
            y_full = np.zeros(len(X_full))
            y_full[:len(current_positive_patches)] = 1
            
            # Train-validation split
            X_train_trial, X_val_trial, y_train_trial, y_val_trial = train_test_split(
                X_full, y_full, test_size=test_size, random_state=random_state, stratify=y_full
            )
            
            # Scaling
            scaler_trial = StandardScaler()
            X_train_std_trial = scaler_trial.fit_transform(X_train_trial)
            
            # PCA
            pca_trial = PCA(**pca_params)
            X_train_pca_trial = pca_trial.fit_transform(X_train_std_trial)
            
            # 5. CREATE AND EVALUATE MODEL
            model = model_class(**model_params)
            
            cv_scores = cross_val_score(
                model, X_train_pca_trial, y_train_trial,
                cv=cv_folds, scoring='f1', n_jobs=n_jobs
            )
            
            return cv_scores.mean()
            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return 0.0  # Return poor score for failed trials
    
    # Run optimization
    print(f"Starting model-agnostic optimization with Optuna...")
    print(f"Model: {model_class.__name__}")
    print(f"Parameters to optimize: {list(param_distributions.keys())}")
    if optimize_faces:
        print(f"Also optimizing: Number of faces (P)")
    if optimize_pca:
        print(f"Also optimizing: PCA parameters")
    print(f"Trials: {n_trials}")
    print(f"CV folds: {cv_folds}")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    
    # Get best parameters
    best_params = study.best_params.copy()
    best_score = study.best_value
    
    print(f"\nOptimization completed!")
    print(f"Best F1-score: {best_score:.4f}")
    print(f"Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # 6. TRAIN FINAL MODEL WITH BEST PARAMETERS
    print(f"\nTraining final model with best parameters...")
    
    # Separate different types of parameters
    model_params_final = {}
    best_n_positive = len(positive_patches)
    best_pca_params = fixed_pca_params or {'n_components': 500, 'whiten': True}
    
    for key, value in best_params.items():
        if key == 'n_positive_samples':
            best_n_positive = value
        elif key.startswith('pca_'):
            param_name = key.replace('pca_', '')
            best_pca_params[param_name] = value
        else:
            model_params_final[key] = value
    
    # Add fixed parameters for final model
    if 'random_state' not in model_params_final and hasattr(model_class(), 'random_state'):
        model_params_final['random_state'] = random_state
    if 'n_jobs' not in model_params_final and hasattr(model_class(), 'n_jobs'):
        model_params_final['n_jobs'] = n_jobs
    
    # Prepare final data
    best_positive_patches = positive_patches[:best_n_positive]
    X_best = np.array([
        feature.hog(im) for im in chain(best_positive_patches, negative_patches)
    ])
    y_best = np.zeros(len(X_best))
    y_best[:len(best_positive_patches)] = 1
    
    # Final train-test split
    X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(
        X_best, y_best, test_size=test_size, random_state=random_state, stratify=y_best
    )
    
    # Final scaling
    scaler_best = StandardScaler()
    X_train_std_best = scaler_best.fit_transform(X_train_best)
    X_test_std_best = scaler_best.transform(X_test_best)
    
    # Final PCA
    pca_best = PCA(**best_pca_params)
    X_train_pca_best = pca_best.fit_transform(X_train_std_best)
    X_test_pca_best = pca_best.transform(X_test_std_best)
    
    # Train final model
    final_model = model_class(**model_params_final)
    final_model.fit(X_train_pca_best, y_train_best)
    
    # Evaluate final model
    y_pred_final = final_model.predict(X_test_pca_best)
    final_f1 = f1_score(y_test_best, y_pred_final)

    # Finalize the best model
    X_std_final = scaler_best.fit_transform(X_best)
    X_pca_final = pca_best.fit_transform(X_std_final)
    final_model = model_class(**model_params_final)
    final_model.fix(X_pca_final, y_best)

    
    print(f"Final model F1-score on test set: {final_f1:.4f}")
    print(f"Best number of positive samples (P): {best_n_positive}")
    
    return {
        'study': study,
        'best_params': best_params,
        'best_score': best_score,
        'final_test_score': final_f1,
        'trained_model': final_model,
        'scaler': scaler_best,
        'pca': pca_best,
        'model_class': model_class,
        'best_model': final_model,
        'model_params': model_params_final,
        'pca_params': best_pca_params,
        'n_positive_samples': best_n_positive
    }
