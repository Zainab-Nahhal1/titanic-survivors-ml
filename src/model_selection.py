from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def compare_models(X, y):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    print("\nModel comparison (5-fold CV accuracy):")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results[name] = scores.mean()
        print(f" - {name}: {scores.mean():.4f} (± {scores.std():.4f})")
    best = max(results, key=results.get)
    print(f"\n✅ Best model by CV accuracy: {best} ({results[best]:.4f})\n")
    return best, results
