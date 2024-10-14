import streamlit as st
import pandas as pd
import numpy as np
from utils import process_raw_input
from utils import load_model

# # Load the trained model
# def load_model():
#     model = joblib.load('lasso_model.pkl')
#     return model

# Define the Streamlit app
def main():
    st.title('House Price Prediction App')

    # Introduction
    st.write("This app predicts house prices based on input features. Adjust the sliders and fields below to see the predicted house price.")

    # Sidebar for user input
    st.sidebar.header('User Input Features')

    # Function to take inputs from user
    def user_input_features():
        # Load the CSV file
        df = pd.read_csv('train_all.csv')

        # Remove the 'Id' column if it exists
        if 'Id' in df.columns:
            df = df.drop(columns=['Id'])

        # Create an empty dictionary to store the user input values
        user_inputs = {}

        # Loop through each column to create appropriate input widgets
        for column in df.columns:
            col_type = df[column].dtype

            if pd.api.types.is_numeric_dtype(col_type):  # For numeric columns
                min_value = float(df[column].min())
                max_value = float(df[column].max())
                mean_value = float(df[column].mean())
                if min_value >= max_value:
                    user_inputs["column"] = min_value
                    continue
                print(min_value, max_value, column)
                user_inputs[column] = st.sidebar.slider(
                    f'Select {column}', min_value, max_value, value=min_value)

            elif pd.api.types.is_categorical_dtype(col_type) or df[column].nunique() < 10:  # For categorical columns
                options = df[column].unique().tolist()
                user_inputs[column] = st.sidebar.selectbox(f'Select {column}', options)

            else:  # For other column types (fallback to text input)
                user_inputs[column] = df[column].mode()


        # Create a dataframe from the input features
        data = pd.DataFrame(user_inputs, index=[0])
        return data

    # Get user input features
    input_df = user_input_features()

    # Display the user inputs
    st.subheader('User Input Features')
    st.write(input_df)
    print(input_df.T)

    processed_df = process_raw_input(input_df)

    # Load the trained model
    model = load_model()

    # Make predictions
    prediction = model.predict(processed_df)

    # Display prediction
    st.subheader('Predicted House Price:')
    st.write(f"${prediction[0]:,.2f}")


if __name__ == '__main__':
    main()
