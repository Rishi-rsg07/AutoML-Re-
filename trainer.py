import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
import tensorflow as tf

def train_best_model(preprocessor, X_train, X_test, y_train, y_test, task_type: str, output_dir="/tmp"):
    
    # Transforming raw data through our dynamic preprocessor
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    input_dim = X_train_trans.shape[1]
    best_score = -float('inf')
    best_model_artifact = None
    model_tech = "sklearn"
    metrics = {}

    if task_type == "classification":
        # Candidate 1: Random Forest Grid Search
        rf = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train_trans, y_train)
        
        rf_preds = grid.best_estimator_.predict(X_test_trans)
        rf_acc = accuracy_score(y_test, rf_preds)
        
        metrics["RandomForest_Accuracy"] = float(rf_acc)
        best_score = rf_acc
        best_model_artifact = grid.best_estimator__
        
    else:
        # Candidate 2: Regression Suite
        reg = RandomForestRegressor(random_state=42)
        param_grid = {'n_estimators': [50, 100]}
        grid = GridSearchCV(reg, param_grid, cv=3, scoring='r2')
        grid.fit(X_train_trans, y_train)
        
        reg_preds = grid.best_estimator_.predict(X_test_trans)
        reg_r2 = r2_score(y_test, reg_preds)
        
        metrics["RandomForest_R2"] = float(reg_r2)
        best_score = reg_r2
        best_model_artifact = grid.best_estimator_

    # Fallback Option: Neural Network via TensorFlow Keras
    # (Exemplifying a highly adaptable MLP implementation)
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1 if task_type == "regression" else len(np.unique(y_train)), 
                              activation='linear' if task_type == "regression" else 'softmax')
    ])
    
    loss_fn = 'mse' if task_type == "regression" else 'sparse_categorical_crossentropy'
    optimizer_algo = tf.keras.optimizers.Adam(learning_rate=0.01)
    nn_model.compile(optimizer=optimizer_algo, loss=loss_fn, metrics=['accuracy' if task_type == "classification" else 'mae'])
    
    # Train the neural network
    nn_model.fit(X_train_trans, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Evaluate TensorFlow performance
    nn_eval = nn_model.evaluate(X_test_trans, y_test, verbose=0)
    
    if task_type == "classification":
        metrics["TensorFlow_Accuracy"] = float(nn_eval[1])
        if nn_eval[1] > best_score:
            best_score = nn_eval[1]
            best_model_artifact = nn_model
            model_tech = "tensorflow"
    else:
        # For simplicity of metric ranking, inverse of MAE can be mapped or tracked directly
        metrics["TensorFlow_MAE"] = float(nn_eval[1])

    # Package whole pipeline (Preprocessing + Model Estimator) for true portability
    model_path = f"{output_dir}/model"
    if model_tech == "sklearn":
        export_pipeline = Pipeline([('preprocessor', preprocessor), ('estimator', best_model_artifact)])
        model_path += ".pkl"
        joblib.dump(export_pipeline, model_path)
    else:
        # For mixed deployment environments, save structural weights separate from preprocessor object
        model_path += ".h5"
        best_model_artifact.save(model_path)
        joblib.dump(preprocessor, f"{output_dir}/preprocessor.pkl")
        
    return metrics, model_path, model_tech