import streamlit as st
from predict import predict_cuisine
from display import show_results, show_documentation
from model import load_model_and_features


def main():
    st.sidebar.title("Cuisine Predictor")

    # Load model and features
    model, feature_df = load_model_and_features()
    ingredients = list(feature_df.columns)

    option = st.sidebar.radio("Select an option:", ["Home", "Documentation", "About Us"])

    if option == "Home":
        st.title("Cuisine Predictor")
        st.write("Select 5 unique ingredients to predict the cuisine.")

        with st.form(key='cuisine_form'):
            selected_ingredients = []
            for i in range(1, 6):
                selected_ingredient = st.selectbox(f"Select Ingredient {i}:", options=ingredients, key=i)
                if selected_ingredient:
                    selected_ingredients.append(selected_ingredient)

            submit_button = st.form_submit_button(label='Predict Cuisine')

        if submit_button:
            if len(selected_ingredients) != len(set(selected_ingredients)):
                st.error("Please select 5 unique ingredients.")
            elif len(selected_ingredients) < 5:
                st.error("Please select all 5 ingredients.")
            else:
                result_df = predict_cuisine(model, feature_df, selected_ingredients)
                show_results(result_df, selected_ingredients)

    elif option == "Documentation":
        st.title("Documentation")
        st.write("Explore the details of the cuisine prediction model.")
        show_documentation()

    elif option == "About Us":
        st.title("About Us")
        st.write(
            "This app is built to predict cuisines based on selected ingredients using a Logistic Regression model.")


if __name__ == '__main__':
    main()
