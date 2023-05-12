from sklearn.model_selection import train_test_split

def nested_cross_validation(data, target, model, folds):
    outer_scores = []
    best_score = 0
    best_train_data = None
    best_train_target = None
    best_test_data = None
    best_test_target = None
    
    for _ in folds.split(data):

        x_train_fold, x_val_fold, y_train_fold, y_val_fold = train_test_split(data, target, test_size=0.2, random_state=42)

        # Inner loop: model training
        model.fit(x_train_fold, y_train_fold)

        # Evaluate the model on the validation fold
        score = model.score(x_val_fold, y_val_fold)
        outer_scores.append(score)
        if score > best_score:
            best_score = score
            best_train_data = x_train_fold
            best_train_target = y_train_fold
            best_test_data = x_val_fold
            best_test_target = y_val_fold
            best_model = model
    
    best_model.fit(best_train_data, best_train_target)

    return best_train_data, best_train_target, best_test_data, best_test_target