import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score

def nested_cross_validation(x_train, y_train, model, param_grid, outer_cv, inner_cv=None, scoring='accuracy'):
    # Outer loop: model evaluation
    outer_scores = []
    
    # the same number of folds, shuffling, and random state as the outer loop 
    # so it's like we are not using an inner loop
    if inner_cv is None:
        inner_cv = outer_cv
    
    for train_index, test_index in outer_cv.split(x_train):
        X_train_fold, X_val_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Inner loop: hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
        grid_search.fit(X_train_fold, y_train_fold)

        # Evaluate the model with the best hyperparameters on the validation fold
        best_model = grid_search.best_estimator_
        score = best_model.score(X_val_fold, y_val_fold)
        outer_scores.append(score)

    cv_scores = cross_val_score(model, x_train, y_train, cv=outer_cv, scoring=scoring)
    avg_score = np.mean(outer_scores)

    return cv_scores, avg_score