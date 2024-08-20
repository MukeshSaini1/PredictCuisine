import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load your trained model and feature DataFrame
with open('model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('feature_df.pkl', 'rb') as feature_file:
    feature_df = pickle.load(feature_file)

# Extract the ingredients (feature names) from the DataFrame
ingredients = list(feature_df.columns)


def predict_cuisine(user_ingredients):
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


def show_results(result_df, selected_ingredients):
    st.title("Cuisine Prediction Result")

    # Display selected ingredients
    st.write("**Selected Ingredients:**")
    st.write(", ".join(selected_ingredients))

    # Convert result_df to a dictionary for displaying in a table
    result_dict = result_df.T.to_dict()[0]

    # Display the results in a larger table
    st.write("**Prediction Results:**")
    result_table = pd.DataFrame(list(result_dict.items()), columns=['Cuisine', 'Probability'])
    result_table['Probability'] = result_table['Probability'] * 100
    st.dataframe(result_table.style.set_properties(**{'font-size': '20px'}))

    # Plot pie chart
    fig = px.pie(
        result_table,
        names='Cuisine',
        values='Probability',
        title='Cuisine Prediction Probability Distribution',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    st.plotly_chart(fig, use_container_width=True)

    # Add "Predict Again" button
    if st.button('Predict Again'):
        st.experimental_rerun()


def show_documentation():
    st.title("Logistic Regression Model with Cuisines Data")

    st.header("Cuisine Predictor Model with Cuisines Data")

    st.subheader("Step 1: Importing Libraries and Reading the Data")
    st.write("""
    We start by importing the necessary libraries and reading the data:
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df = pd.read_csv("cuisines.csv")
    df
    ```
    **Explanation:** This code imports necessary libraries: pandas for data manipulation, numpy for numerical operations, and matplotlib for plotting graphs. Then, it reads the CSV file `cuisines.csv` into a DataFrame `df`.
    """)

    st.subheader("Step 2: Visualizing Cuisine Distribution")
    st.write("""
    We can visualize the distribution of different cuisines using a bar chart:
    ```python
    df.cuisine.value_counts().plot.barh()
    ```
    **Explanation:** The `value_counts()` method counts the occurrences of each unique value in the cuisine column. The `.plot.barh()` method then plots this distribution as a horizontal bar chart, showing how many dishes belong to each cuisine.
    """)

    st.subheader("Step 3: Filtering Data by Specific Cuisines")
    st.write("""
    We create separate DataFrames for specific cuisines:
    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    ```
    **Explanation:** This code creates separate DataFrames for each specified cuisine (Thai, Japanese, Chinese, Indian, Korean) by filtering the original DataFrame.
    """)

    st.subheader("Step 4: Creating Ingredient DataFrame for Each Cuisine")
    st.write("""
    Next, we create a DataFrame showing the most common ingredients for each cuisine:
    ```python
    def create_ingredient_df(df):
        ing_df = df.T.drop(['cuisine', 'Unnamed: 0']).sum(axis=1).to_frame('value')
        ing_df = ing_df[(ing_df.T != 0).any()]
        ing_df = ing_df.sort_values(by='value', ascending=False, inplace=False)
        return ing_df

    thai_ing_df = create_ingredient_df(thai_df)
    thai_ing_df.head(10).plot.barh()
    ```
    **Explanation:** The `create_ingredient_df` function:
    - Transposes the DataFrame to switch rows and columns.
    - Drops the `cuisine` and `Unnamed: 0` columns, which are not needed for ingredient analysis.
    - Sums the occurrences of each ingredient across all rows (i.e., dishes) and creates a new DataFrame with a single column named `value`.
    - Filters out ingredients that do not appear in any dish.
    - Sorts the ingredients by their frequency in descending order.
    - Returns the resulting DataFrame.

    Visualization: The `thai_ing_df.head(10).plot.barh()` line visualizes the top 10 ingredients in Thai cuisine as a horizontal bar chart.
    """)

    st.subheader("Step 5: Preparing Features and Labels for Model Training")
    st.write("""
    We then prepare the features and labels for model training:
    ```python
    feature_df = df.drop(["cuisine", "Unnamed: 0", "rice", "garlic", "ginger"], axis=1)
    labels_df = df.cuisine
    feature_df.head()
    ```
    **Explanation:**
    - `feature_df`: This is the DataFrame with features used to predict the cuisine. It drops the `cuisine` (the target label), `Unnamed: 0`, and common ingredients like `rice`, `garlic`, and `ginger` from the original DataFrame, as they might not help in differentiating between cuisines.
    - `labels_df`: This contains the `cuisine` column, which is the target variable for prediction.
    """)

    st.subheader("Step 6: Handling Imbalanced Data with SMOTE")
    st.write("""
    To handle imbalanced data, we apply SMOTE:
    ```python
    from imblearn.over_sampling import SMOTE
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    print(transformed_label_df.value_counts())
    print(labels_df.value_counts())
    ```
    **Explanation:** The dataset might have an imbalanced number of samples for each cuisine. SMOTE (Synthetic Minority Over-sampling Technique) is used here to balance the classes by oversampling the minority classes, generating synthetic examples. The new balanced feature and label DataFrames are `transformed_feature_df` and `transformed_label_df`.
    """)

    st.subheader("Step 7: Training a Logistic Regression Model")
    st.write("""
    Now, we train a logistic regression model:
    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    x_train, x_test, y_train, y_test = train_test_split(transformed_feature_df, transformed_label_df, test_size=0.3)
    lr = LogisticRegression(multi_class="ovr", solver="liblinear")
    model = lr.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    accuracy
    ```
    **Explanation:**
    - **Splitting Data:** The dataset is split into training (70%) and testing (30%) sets.
    - **Logistic Regression Model:** A logistic regression model is created with ovr (one-vs-rest) strategy for multiclass classification.
    - **Model Training:** The model is trained using the training data.
    - **Model Accuracy:** The accuracy of the model on the test data is calculated and printed.
    """)

    st.subheader("Step 8: Making Predictions and Evaluating the Model")
    st.write("""
    We can make predictions and evaluate the model's performance:
    ```python
    print(x_test.iloc[10][x_test.iloc[10] != 0].keys())
    print(y_test.iloc[10])

    test = x_test.iloc[10].values.reshape(-1,1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    resultdf
    ```
    **Explanation:**
    - The code snippet prints out the ingredients present in the 11th test sample and the actual cuisine.
    - The model predicts the probability of each cuisine for the 11th test sample, and the results are stored in `resultdf`.
    """)

    st.subheader("Step 9: Generating a Classification Report")
    st.write("""
    Finally, we generate a classification report to evaluate model performance:
    ```python
    from sklearn.metrics import classification_report

    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    ```
    **Explanation:** The `classification_report` provides a detailed performance evaluation of the model, including precision, recall, and F1-score for each cuisine.
    """)


def main():
    st.sidebar.title("Cuisine Predictor")



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
                result_df = predict_cuisine(selected_ingredients)
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
