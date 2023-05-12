import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

def nested_cross_validation(data, target, model, param_grid, outer_cv, inner_cv=None, scoring='accuracy'):
    # Outer loop: model evaluation
    outer_scores = []
    best_score = 0
    best_train_data = None
    best_train_target = None
    best_test_data = None
    best_test_target = None
    
    # the same number of folds, shuffling, and random state as the outer loop 
    # so it's like we are not using an inner loop
    if inner_cv is None:
        inner_cv = outer_cv
    
    for _ in outer_cv.split(data):
        x_train_fold, x_val_fold, y_train_fold, y_val_fold = train_test_split(data, target, test_size=0.2, random_state=42)

        # Inner loop: hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
        grid_search.fit(x_train_fold, y_train_fold)

        # Evaluate the model with the best hyperparameters on the validation fold
    
        score = grid_search.best_estimator_.score(x_val_fold, y_val_fold)
        outer_scores.append(score)
        if score > best_score:
            best_score = score
            best_train_data = x_train_fold
            best_train_target = y_train_fold
            best_test_data = x_val_fold
            best_test_target = y_val_fold
            best_model = grid_search.best_estimator_

    cv_scores = cross_val_score(model, data, target, cv=outer_cv, scoring=scoring)
    avg_score = np.mean(outer_scores)
    best_model.fit(best_train_data, best_train_target)

    return cv_scores, avg_score, best_train_data, best_train_target, best_test_data, best_test_target