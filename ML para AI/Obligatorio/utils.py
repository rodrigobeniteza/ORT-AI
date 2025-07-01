# import pandas as py
# import numpy as np
# import optuna
# import xgboost as xgb
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# # from sklearn.feature_extraction.image import PatchExtractor
# # from skimage.transform import resize
# from sklearn.base import BaseEstimator # Para tipos de modelos base

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.metrics import f1_score
# from itertools import chain
# from sklearn.neural_network import MLPClassifier
   

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
    fixed_pca_params=None,
    param_transformer=None  # Nueva función para transformar parámetros
):
    """
    Comprehensive and model-agnostic Optuna optimization function.
    1. Model hyperparameters 
    2. Number of positive samples (P = len(positive_patches))
    
    Parameters:
    -----------
    positive_patches : array-like
        Array of positive face patches
    negative_patches : array-like  
        Array of negative background patches
    model_class :  class
        The model class to instantiate (e.g., xgb.XGBClassifier, GradientBoostingClassifier)
    param_distributions : dict
        Dictionary defining parameter search spaces using Optuna suggest methods.
        Format: {
            'param_name': {
                'type': 'int'|'float'|'categorical'|'uniform'|'loguniform',
                'low': value,      # for int/float
                'high': value,     # for int/float  
                'step': value,     # optional for int
                'log': bool,       # optional for float
                'choices': list    # for categorical
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
    param_transformer : callable, optional
        Function to transform optimized parameters before passing to model.
        Should take a dict of parameters and return a transformed dict.
        
    Returns:
    --------
    dict: Dictionary containing:
        - 'study': Optuna study object
        - 'best_params': Best parameters found
        - 'cv_score': Best F1-score achieved in cross-validation
        - 'test_score': Best F1-score achieved in test set
        - 'trained_model': Model trained with best parameters
        - 'scaler': Fitted StandardScaler
        - 'pca': Fitted PCA transformer
        - 'model_class': Model class used
        - 'model_params': best model parameters found
        - 'pca_params': PCA parameters used
        - 'n_positive_samples': best number of positive samples found
        - 'X_train_std': X_train_std_final
        - 'y_train': y_train_final
        - 'X_test_std': X_test_std_final
        - 'y_test': y_test_final
    -----------

    """
    
    import optuna
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import f1_score, classification_report
    from skimage import feature
    from itertools import chain
    import numpy as np
    import inspect
    
    # Handle both class and instance inputs
    if inspect.isclass(model_class):
        model_class_name = model_class.__name__
        model_constructor = model_class
    else:
        model_class_name = model_class.__class__.__name__
        model_constructor = model_class.__class__
        print(f"Warning: You passed a model instance. Using the class {model_class_name} instead.")
    
    def objective(trial):
        """
        Model-agnostic objective function using only cross-validation
        """
        model_params = {}
        
        # 1. OPTIMIZE MODEL PARAMETERS
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
        
        # Transform parameters if transformer is provided
        if param_transformer:
            model_params = param_transformer(model_params)
        
        # Add fixed parameters if they exist in the model
        try:
            dummy_model = model_constructor()
            if hasattr(dummy_model, 'random_state') and 'random_state' not in model_params:
                model_params['random_state'] = random_state
            if hasattr(dummy_model, 'n_jobs') and 'n_jobs' not in model_params:
                model_params['n_jobs'] = n_jobs
        except:
            pass
            
        # 2. OPTIMIZE NUMBER OF POSITIVE SAMPLES (P)
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
        
        # 3. OPTIMIZE PCA PARAMETERS
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
            return 0.0
    
    # Run optimization
    print(f"Starting model-agnostic optimization with Optuna...")
    print(f"Model: {model_class_name}")
    print(f"Using {cv_folds}-fold cross-validation for optimization")
    print(f"Parameters to optimize: {list(param_distributions.keys())}")
    if optimize_faces:
        print(f"Also optimizing: Number of faces (P)")
    if optimize_pca:
        print(f"Also optimizing: PCA parameters")
    print(f"Trials: {n_trials}")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    
    # Get best parameters
    best_params = study.best_params.copy()
    best_score = study.best_value
    
    print(f"\nOptimization completed!")
    print(f"Best CV F1-score: {best_score:.4f}")
    print(f"Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # 6. FINAL EVALUATION ON HELD-OUT TEST SET
    print(f"\nTraining final model and evaluating on held-out test set...")
    
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
    
    # Transform final parameters if transformer is provided
    if param_transformer:
        model_params_final = param_transformer(model_params_final)
    
    # Add fixed parameters for final model
    try:
        dummy_model = model_constructor()
        if hasattr(dummy_model, 'random_state') and 'random_state' not in model_params_final:
            model_params_final['random_state'] = random_state
        if hasattr(dummy_model, 'n_jobs') and 'n_jobs' not in model_params_final:
            model_params_final['n_jobs'] = n_jobs
    except:
        pass
    
    # Prepare final data with best parameters
    final_positive_patches = positive_patches[:best_n_positive]
    X_final = np.array([
        feature.hog(im) for im in chain(final_positive_patches, negative_patches)
    ])
    y_final = np.zeros(len(X_final))
    y_final[:len(final_positive_patches)] = 1
    
    # Split into train/test for final evaluation
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X_final, y_final, test_size=test_size, random_state=random_state, stratify=y_final
    )
    
    # Final preprocessing
    scaler_final = StandardScaler()
    X_train_std_final = scaler_final.fit_transform(X_train_final)
    X_test_std_final = scaler_final.transform(X_test_final)
    
    pca_final = PCA(**best_pca_params)
    X_train_pca_final = pca_final.fit_transform(X_train_std_final)
    X_test_pca_final = pca_final.transform(X_test_std_final)
    
    # Train final model
    best_model = model_constructor(**model_params_final)
    best_model.fit(X_train_pca_final, y_train_final)
    
    # Evaluate on test set
    y_pred_final = best_model.predict(X_test_pca_final)
    final_f1 = f1_score(y_test_final, y_pred_final)

    # Finalize the model
    X_final_std = scaler_final.fit_transform(X_final)
    X_final_pca = pca_final.fit_transform(X_final_std)
    final_model = model_constructor(**model_params_final)
    final_model.fit(X_final_pca, y_final)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Cross-validation F1-score (optimization): {best_score:.4f}")
    print(f"Test set F1-score (final evaluation): {final_f1:.4f}")
    print(f"Best number of positive samples (P): {best_n_positive}")
    print(f"\nDetailed test set results:")
    print(classification_report(y_test_final, y_pred_final))
    
    return {
        'study': study,
        'best_params': best_params,
        'cv_score': best_score,
        'test_score': final_f1,
        'trained_model': final_model,
        'scaler': scaler_final,
        'pca': pca_final,
        'model_class': model_constructor,
        # 'model_params': model_params_final,
        'pca_params': best_pca_params,
        # 'n_positive_samples': best_n_positive,
        'X_train_pca': X_train_pca_final,
        'y_train': y_train_final,
        'X_test_pca': X_test_pca_final,
        'y_test': y_test_final,
        'y_pred_final': y_pred_final

    }

# Función específica para transformar parámetros de MLPClassifier
def mlp_param_transformer(params):
    """
    Transform custom MLP parameters to sklearn MLPClassifier format
    """
    transformed_params = {}
    
    # Extract architecture parameters
    n_layers = params.pop('n_layers', 1)
    
    # Build hidden_layer_sizes tuple
    hidden_layers = []
    for i in range(n_layers):
        layer_key = f'n_units_l{i}'
        if layer_key in params:
            hidden_layers.append(params.pop(layer_key))
    
    # Remove unused layer parameters
    keys_to_remove = [key for key in params.keys() if key.startswith('n_units_l')]
    for key in keys_to_remove:
        params.pop(key)
    
    # Set hidden_layer_sizes
    if hidden_layers:
        transformed_params['hidden_layer_sizes'] = tuple(hidden_layers)
    
    # Add remaining parameters
    transformed_params.update(params)
    
    return transformed_params
    # return results

# Visualization function for results
def plot_optimization_results(study):
    """
    Plot optimization results - Compatible with different Optuna versions
    """
    import matplotlib.pyplot as plt
    import optuna
    
    # Plot optimization history
    try:
        # Try new method (Optuna >= 3.0)
        optuna.visualization.matplotlib.plot_optimization_history(study)
    except TypeError:
        print("Error: plot_optimization_history is not supported in this version of Optuna")
        
    # Plot parameter importances
    try:
        optuna.visualization.matplotlib.plot_param_importances(study)
    except TypeError:
        print("Error: plot_param_importances is not supported in this version of Optuna")
    
    plt.tight_layout()
    plt.show()


def plot_pca_analysis(results, X_train_pca, y_train):
    """
    Plot comprehensive PCA analysis including variance analysis and component visualization
    
    Parameters:
    -----------
    results : dict
        Results from optimize_model_and_faces_with_optuna function
    X_train_pca : array-like, optional
        PCA-transformed training data for visualization
    y_train : array-like, optional
        Training labels for visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract PCA object from results
    pca = results['pca']
    # X_train_pca = results['X_train_pca']
    # y_train = results['y_train']
    
    # Get variance information
    var_explicada = pca.explained_variance_ratio_
    var_acumulada = np.cumsum(var_explicada)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. VARIANCE ANALYSIS (2 plots)
    # Varianza explicada por componente
    ax1 = plt.subplot(2, 3, 1)
    plt.bar(range(1, len(var_explicada)+1), var_explicada, color='skyblue', alpha=0.8)
    plt.title('Varianza explicada por componente', fontsize=14, fontweight='bold')
    plt.xlabel('Componente principal')
    plt.ylabel('Proporción de varianza')
    plt.grid(True, alpha=0.3)
    
    # Varianza acumulada
    ax2 = plt.subplot(2, 3, 2)
    plt.bar(range(1, len(var_acumulada)+1), var_acumulada, color='lightgreen', alpha=0.8)
    plt.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95%')
    plt.title('Varianza acumulada', fontsize=14, fontweight='bold')
    plt.xlabel('Número de componentes')
    plt.ylabel('Proporción acumulada de varianza')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. VARIANCE INFORMATION TEXT
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    
    # Find components needed for different variance thresholds
    components_90 = np.argmax(var_acumulada >= 0.90) + 1 if any(var_acumulada >= 0.90) else len(var_acumulada)
    components_95 = np.argmax(var_acumulada >= 0.95) + 1 if any(var_acumulada >= 0.95) else len(var_acumulada)
    components_99 = np.argmax(var_acumulada >= 0.99) + 1 if any(var_acumulada >= 0.99) else len(var_acumulada)
    
    info_text = f"""
    PCA ANALYSIS SUMMARY
    
    Total components: {len(var_explicada)}
    Components used: {results['pca_params']['n_components']}
    Whiten: {results['pca_params']['whiten']}
    
    VARIANCE EXPLAINED:
    • First component: {var_explicada[0]:.3f} ({var_explicada[0]*100:.1f}%)
    • First 2 components: {var_acumulada[1]:.3f} ({var_acumulada[1]*100:.1f}%)
    • First 3 components: {var_acumulada[2]:.3f} ({var_acumulada[2]*100:.1f}%)
    
    COMPONENTS NEEDED:
    • 90% variance: {components_90} components
    • 95% variance: {components_95} components  
    • 99% variance: {components_99} components
    
    Current variance captured: {var_acumulada[results['pca_params']['n_components']-1]:.3f}
    ({var_acumulada[results['pca_params']['n_components']-1]*100:.1f}%)
    """
    
    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # 4. 2D VISUALIZATION (if data provided)
    # if X_train_pca is not None and y_train is not None:
    ax4 = plt.subplot(2, 3, 4)
    colors = ['tab:red', 'tab:blue']
    labels = ['Background', 'Face']
    
    for class_value in [0, 1]:
        mask = y_train == class_value
        plt.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], 
                    c=colors[class_value], label=labels[class_value], 
                    alpha=0.6, s=20)
    
    plt.xlabel(f'PC1 ({var_explicada[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({var_explicada[1]*100:.1f}% variance)')
    plt.title('PCA - Primeras 2 componentes', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. 3D VISUALIZATION
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    
    for class_value in [0, 1]:
        mask = y_train == class_value
        ax5.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], X_train_pca[mask, 2],
                    c=colors[class_value], label=labels[class_value], 
                    alpha=0.7, s=15)
    
    ax5.set_xlabel(f'PC1 ({var_explicada[0]*100:.1f}%)')
    ax5.set_ylabel(f'PC2 ({var_explicada[1]*100:.1f}%)')
    ax5.set_zlabel(f'PC3 ({var_explicada[2]*100:.1f}%)')
    ax5.set_title('PCA - Primeras 3 componentes', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.view_init(elev=15, azim=45)
    
        
    plt.suptitle(f'PCA Analysis - {results["model_class"].__name__} Optimization', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


def plot_pca_analysis_plotly(results, X_train_pca, y_train):
    """
    Interactive PCA analysis using Plotly (alternative to matplotlib version)
    """
    import numpy as np
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
    except ImportError:
        print("Plotly not available. Please install with: pip install plotly")
        return
    
    # Extract PCA object from results
    pca = results['pca']
    
    # Get variance information
    var_explicada = pca.explained_variance_ratio_
    var_acumulada = np.cumsum(var_explicada)
    
    # Create subplots - Removed one column since we're removing the heatmap
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Varianza explicada por componente', 
            'Varianza acumulada',
            'PCA - 2D Visualization',
            'PCA - 3D Visualization'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter3d"}]]
    )
    
    # 1. Varianza explicada - Improved color and visibility
    fig.add_trace(
        go.Bar(x=list(range(1, len(var_explicada)+1)), 
               y=var_explicada, 
               name='Varianza explicada',
               marker_color='rgba(30, 144, 255, 0.8)',  # Strong blue color with transparency
               marker_line_color='rgba(30, 144, 255, 1.0)',  # Solid border
               marker_line_width=1),
        row=1, col=1
    )
    
    # 2. Varianza acumulada
    fig.add_trace(
        go.Bar(x=list(range(1, len(var_acumulada)+1)), 
               y=var_acumulada, 
               name='Varianza acumulada',
               marker_color='rgba(64, 224, 208, 0.8)',  # Turquoise/aqua green color
               marker_line_color='rgba(64, 224, 208, 1.0)',  # Solid border
               marker_line_width=1),
        row=1, col=2
    )
    
    # Add 95% line
    fig.add_hline(y=0.95, line_dash="dash", line_color="red", row=1, col=2)
    
    # 3. 2D and 3D visualizations (if data provided)
    if X_train_pca is not None and y_train is not None:
        colors = ['red', 'blue']
        labels = ['Background', 'Face']
        
        # 2D scatter
        for class_value in [0, 1]:
            mask = y_train == class_value
            fig.add_trace(
                go.Scatter(x=X_train_pca[mask, 0], y=X_train_pca[mask, 1],
                          mode='markers', name=labels[class_value],
                          marker=dict(color=colors[class_value], opacity=0.6)),
                row=2, col=1
            )
        
        # 3D scatter
        for class_value in [0, 1]:
            mask = y_train == class_value
            fig.add_trace(
                go.Scatter3d(x=X_train_pca[mask, 0], 
                           y=X_train_pca[mask, 1], 
                           z=X_train_pca[mask, 2],
                           mode='markers', name=f'{labels[class_value]} 3D',
                           marker=dict(color=colors[class_value], opacity=0.7, size=3)),
                row=2, col=2
            )
    
    # Update layout - Adjusted for 2x2 grid
    fig.update_layout(
        height=700,
        width=1200,
        title_text=f"Interactive PCA Analysis - {results['model_class'].__name__}",
        title_font_size=16,
        showlegend=True
    )
    
    fig.show()


def print_confusion_matrix(results):
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    y_pred_binary = results['y_pred_final']
    y_test = results['y_test']

    # --- Cálculo usando la Matriz de Confusión ---
    # confusion_matrix(y_true, y_pred)
    # Retorna: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred_binary)
    print("Matriz de Confusión:\n", cm)

    TN, FP, FN, TP = cm.ravel() # Desempaquetar los valores de la matriz

    print(f"\nVerdaderos Positivos (TP): {TP}")
    print(f"Falsos Positivos (FP): {FP}")
    print(f"Falsos Negativos (FN): {FN}")
    print(f"Verdaderos Negativos (TN): {TN}")

    # Calcular TPR y FPR manualmente
    tpr_manual = TP / (TP + FN)
    fpr_manual = FP / (FP + TN)

    print(f"\nTPR (Sensibilidad/Recall) calculado manualmente: {tpr_manual:.4f}")
    print(f"FPR calculado manualmente: {fpr_manual:.4f}")
    
    return cm, tpr_manual, fpr_manual


def plot_roc_curves_comparison(results_list, model_names=None, save_path=None, figsize=(12, 10)):
    """
    Plot ROC curves for multiple optimized models for comparison
    
    Parameters:
    -----------
    results_list : list
        List of results dictionaries from optimize_model_and_faces_with_optuna function
    model_names : list, optional
        List of custom names for the models. If None, uses model class names
    save_path : str, optional
        Path to save the plot. If None, only displays the plot
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns:
    --------
    dict: Dictionary containing results for each model:
        - model_name: {'fpr', 'tpr', 'auc_score', 'thresholds'}
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    
    if not results_list:
        print("Error: Empty results list provided")
        return None
    
    # Prepare model names - Fix: Ensure all names are strings
    if model_names is None:
        model_names = []
        for results in results_list:
            if hasattr(results['model_class'], '__name__'):
                model_names.append(results['model_class'].__name__)
            else:
                model_names.append(str(results['model_class']))
    elif len(model_names) != len(results_list):
        print("Error: Number of model names doesn't match number of results")
        return None
    
    # Ensure all model names are strings
    model_names = [str(name) for name in model_names]

    
    # Colors for different models
    colors = ['darkorange', 'darkblue', 'darkgreen', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Store results for each model
    all_results = {}
    
    # Process each model
    for i, (results, model_name) in enumerate(zip(results_list, model_names)):
        try:
            # Extract necessary data
            model = results['trained_model']
            X_test_pca = results['X_test_pca']
            y_test = results['y_test']
            
            # Get prediction probabilities
            try:
                y_proba = model.predict_proba(X_test_pca)[:, 1]
            except AttributeError:
                try:
                    y_proba = model.decision_function(X_test_pca)
                except AttributeError:
                    print(f"Warning: Skipping {model_name} - no probability prediction available")
                    continue
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            auc_score = auc(fpr, tpr)
            
            # Plot ROC curve
            color = colors[i % len(colors)]
            plt.plot(fpr, tpr, color=color, lw=2, 
                     label=f'{model_name} (AUC = {auc_score:.4f})')
            
            # Store results
            all_results[model_name] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc_score': auc_score,
                'thresholds': thresholds,
                'test_f1': results['test_score'],
                'cv_f1': results['cv_score']
            }
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5000)')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves Comparison - Optimized Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC comparison plot saved to: {save_path}")
    
    plt.show()
    
    # Print comparison table - Fix: Ensure proper string formatting
    print(f"\n=== ROC COMPARISON - {len(all_results)} MODELS ===")
    print(f"{'Model':<25} {'AUC':<8} {'Test F1':<10} {'CV F1':<8}")
    print("-" * 55)
    
    # Sort by AUC score (descending)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['auc_score'], reverse=True)
    
    for model_name, metrics in sorted_results:
        # Fix: Ensure model_name is properly formatted as string
        model_name_str = str(model_name)[:24]  # Truncate if too long
        print(f"{model_name_str:<25} {metrics['auc_score']:<8.4f} {metrics['test_f1']:<10.4f} {metrics['cv_f1']:<8.4f}")
    
    # Find best model
    if sorted_results:
        best_model, best_metrics = sorted_results[0]
        best_model_str = str(best_model)
        print(f"\nBest model by AUC: {best_model_str} (AUC = {best_metrics['auc_score']:.4f})")
    
    return all_results
