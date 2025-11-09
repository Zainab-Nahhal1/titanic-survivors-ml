import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

def train_final_and_save_submission(X, y, X_test, test_df, grid_search=True):
    # Optionally tune hyperparameters
    if grid_search:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5]
        }
        grid = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1)
        grid.fit(X, y)
        model = grid.best_estimator_
        print('✅ Grid search done. Best params:', grid.best_params_)
        print('Best CV score:', grid.best_score_)
    else:
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X, y)

    # Fit on full training set
    model.fit(X, y)
    y_train_pred = model.predict(X)
    print('\nTraining evaluation:')
    print('Accuracy:', accuracy_score(y, y_train_pred))
    print('Precision:', precision_score(y, y_train_pred))
    print('Recall:', recall_score(y, y_train_pred))
    print('F1-score:', f1_score(y, y_train_pred))
    print('\nConfusion Matrix:\n', confusion_matrix(y, y_train_pred))
    print('\nClassification Report:\n', classification_report(y, y_train_pred))

    # Predict on test and save
    y_test_pred = model.predict(X_test)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': y_test_pred
    })
    submission.to_csv('results/TitanicFinalPrediction.csv', index=False)
    print("\n✅ Predictions saved to results/TitanicFinalPrediction.csv")
    return model
