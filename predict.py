import pandas as pd

def predict_cuisine(model, feature_df, user_ingredients):
    # Create an empty DataFrame with the same columns as the feature_df
    user_input_vector = pd.DataFrame(columns=feature_df.columns, index=[0]).fillna(0)

    # Update the input vector with the selected ingredients
    for ingredient in user_ingredients:
        if ingredient in user_input_vector.columns:
            user_input_vector[ingredient] = 1

    # Predict probabilities for the input vector
    proba = model.predict_proba(user_input_vector)
    classes = model.classes_
    result_df = pd.DataFrame(data=proba, columns=classes)

    return result_df
