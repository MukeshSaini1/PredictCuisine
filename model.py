import pickle

def load_model_and_features():
    # Load your trained model and feature DataFrame
    with open('model1.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('feature_df.pkl', 'rb') as feature_file:
        feature_df = pickle.load(feature_file)

    return model, feature_df
