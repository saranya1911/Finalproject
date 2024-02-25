import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Set page title and favicon
st.set_page_config(page_title="Department-Wide Sales Prediction", page_icon="📊")
# Load the pickled model
with open('C:\\Users\\psara\\Desktop\\SARANYA_VS\\guvifinal\\reg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the home page
def home():
    st.title('Department-Wide Sales Prediction App')
    st.write("""
    ## Project Overview
    This web application predicts department-wide sales for retail stores based on historical data. It utilizes a trained machine learning model to make predictions based on user input.

    ### Project Skills Takeaway:
    - Data preprocessing and cleaning
    - Exploratory Data Analysis (EDA)       
    - Feature engineering
    - Model training and evaluation
    - Deployment of machine learning models using Streamlit

    ## Problem Statement
    Retail businesses often face challenges in accurately forecasting sales, especially across different departments and stores. This project aims to address this challenge by developing a predictive model that can forecast department-wide sales based on various factors such as store size, type, location, economic indicators, and historical sales data.

    ## Features Used in the Model
    - Store Number
    - Department Number
    - Store Size
    - Store Type
    - Is Holiday
    - Fuel Price
    - CPI (Consumer Price Index)
    - Day
    - Month
    - Year
    - Temperature
    - Total Markdown
    - Unemployment Rate
    - Expected Sales

    """)



# Load the pickled model
with open('C:\\Users\\psara\\Desktop\\SARANYA_VS\\guvifinal\\reg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the basic insights page
def basic_insights():
    st.title('Basic Insights')
    st.write("Here are some basic insights about the data:")
    # Load sample data
    sample_data = pd.read_csv('sales_prediction.csv')

    # Display first few rows of the data
    st.subheader("Sample Data")
    st.write(sample_data.head())

   

    # Calculate total weekly sales for each type and size# Map numerical store type values to corresponding letters
    type_mapping = {1: 'A', 2: 'B', 3: 'C'}
    sample_data['Type'] = sample_data['Type'].map(type_mapping)
    
    weekly_sales_by_type_size = sample_data.groupby(['Type', 'Size'])['Weekly_Sales_log'].sum().reset_index()

    # Create a sunburst chart
    fig = px.sunburst(weekly_sales_by_type_size, path=['Type', 'Size'], values='Weekly_Sales_log')

    # Update layout and display the chart
    fig.update_layout(title="Weekly Sales Distribution by Store Type and Size")
    st.plotly_chart(fig)



# Line Plot of Weekly Sales
    st.subheader("Line Plot of Weekly Sales")
    sample_data['Date'] = pd.to_datetime(sample_data[['Year', 'Month', 'Day']])
    sample_data.set_index('Date', inplace=True)
    weekly_sales = sample_data.resample('W').sum()  # Aggregate weekly sales
    fig, ax = plt.subplots(figsize=(10, 6))
    weekly_sales['Weekly_Sales_log'].plot(ax=ax)
    plt.title("Weekly Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    st.pyplot(fig)

# Line Plot of Markdown over Year
    st.subheader("Markdown Over Year")
    markdown_over_year = sample_data.groupby('Year')['MarkDown_Total_log'].sum()
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(markdown_over_year.index, markdown_over_year.values, marker='o')
    ax3.set_title("Markdown Over Year")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Total Markdown")
    ax3.grid(True)
    st.pyplot(fig3)


# Load the reference data
reference_data = pd.read_csv('C:\\Users\\psara\Desktop\\SARANYA_VS\\guvifinal\\sales_prediction.csv')

# Define the predict function
def predict():
    st.title('Predict Department-Wide Sales')
    st.write("Enter the required features to predict department-wide sales:")
    
    # Display minimum and maximum values for each feature based on the reference data
    st.subheader("Reference Data for Input Values:")
    min_values = reference_data.min()
    max_values = reference_data.max()
    reference_table = pd.DataFrame({'Min Value': min_values, 'Max Value': max_values})
    st.write(reference_table)

    # User inputs with min and max values
    store = st.number_input('Store Number', min_value=int(min_values['Store']), max_value=int(max_values['Store']), value=int(min_values['Store']))
    store_type = st.selectbox('Store Type', [1, 2, 3], format_func=lambda x: 'A' if x == 1 else 'B' if x == 2 else 'C')
    size = st.number_input('Store Size', min_value=int(min_values['Size']), max_value=int(max_values['Size']), value=int(min_values['Size']))
    dept = st.number_input('Department Number', min_value=int(min_values['Dept']), max_value=int(max_values['Dept']), value=int(min_values['Dept']))
    is_holiday = st.checkbox('Is Holiday')  # Include IsHoliday in user input
    fuel_price = st.number_input('Fuel Price', min_value=min_values['Fuel_Price'], max_value=max_values['Fuel_Price'], value=min_values['Fuel_Price'])
    cpi = st.number_input('Consumer Price Index (CPI)', min_value=min_values['CPI'], max_value=max_values['CPI'], value=min_values['CPI'])
    day = st.number_input('Day', min_value=int(min_values['Day']), max_value=int(max_values['Day']), value=int(min_values['Day']))
    month = st.number_input('Month', min_value=int(min_values['Month']), max_value=int(max_values['Month']), value=int(min_values['Month']))
    year = st.number_input('Year', min_value=int(min_values['Year']), max_value=int(max_values['Year']), value=int(min_values['Year']))
    temperature = st.number_input('Temperature', min_value=min_values['Temperature_log'], max_value=max_values['Temperature_log'], value=min_values['Temperature_log'])
    markdown_total = st.number_input('Total Markdown', min_value=min_values['MarkDown_Total_log'], max_value=max_values['MarkDown_Total_log'], value=min_values['MarkDown_Total_log'])
    unemployment = st.number_input('Unemployment Rate', min_value=min_values['Unemployment_log'], max_value=max_values['Unemployment_log'], value=min_values['Unemployment_log'])
    expected_sales = st.number_input('Expected Sales', min_value=min_values['Expected_Sales'], max_value=max_values['Expected_Sales'], value=min_values['Expected_Sales'])

    # Button to make predictions
    if st.button('Predict'):
        # Create a DataFrame with the user input
        user_data = pd.DataFrame([[store, dept, size, store_type, is_holiday, fuel_price, cpi, day, month, year, temperature, markdown_total, unemployment, expected_sales]],
                                 columns=['Store','Type','Size','Dept','IsHoliday','Fuel_Price', 'CPI', 'Day', 'Month', 'Year', 'Temperature_log', 'MarkDown_Total_log', 'Unemployment_log', 'Expected_Sales'])
        # Make predictions
        y_pred = model.predict(user_data)
        predicted_sales = np.exp(y_pred[0])  # Convert from log scale if necessary
        # Display the predicted sales
        st.success(f'Predicted Department-Wide Sales: {predicted_sales}')
        st.balloons()

def main():
    menu = st.sidebar.selectbox('Menu', ['Home', 'Basic Insights', 'Predict'])
    
    if menu == 'Home':
        home()
    elif menu == 'Basic Insights':
        basic_insights()
    elif menu == 'Predict':
        predict()

if __name__ == "__main__":
    main()


