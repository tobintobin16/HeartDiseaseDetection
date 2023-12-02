import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model



# Add title to the page
st.title("Heart Disease Prediction")
st.markdown("<hr style='margin-top: -1px; margin-bottom: 1px;'>", unsafe_allow_html=True)

# Add a brief description
st.markdown(
    """
        <p style="font-size:25px">
            Input your's dataset to predict Heart disease.
        </p>
    """, unsafe_allow_html=True)

# Create dictionaries to store input field values and their respective empty statuses
input_values = {}
input_empty = {}

def display_text_input(label, key):
    input_values[key] = st.text_input(label, key=key)
    input_empty[key] = not input_values[key]

# Display text input fields in the columns
display_text_input("Age", "Age")

input_values['Sex'] = st.radio(
    "Sex",
    ["Male","Female"], horizontal=True)

input_values['Chest Pain Type'] = st.selectbox('Chest Pain Type',('TA','ATA','NAP','ASY'), key="Chest Pain Type")

display_text_input("Resting BP", "Resting BP")
display_text_input("Cholesterol", "Cholesterol")

input_values['Fasting BS'] = st.radio(
    "Fasting BS",
    ["0 Blood sugar < 120 mg","1 Blood sugar > 120 mg"], horizontal=True, key="Fasting BS")

input_values['Resting ECG'] = st.selectbox('Resting ECG',('Normal','ST','LVH'), key="Resting ECG")

display_text_input("Max HR", "Max HR")

input_values['Exercise Angina'] = st.radio(
    "Exercise Angina",
    ["Yes","No"], horizontal=True, key="Exercise Angina")

display_text_input("Old peak", "Old peak")

input_values['St Slope'] = st.selectbox('St Slope',('Up','Flat','Down'), key="St Slope")

if st.button("Predict"):
    empty_fields = [label for label, key in input_empty.items() if key]
    if empty_fields:
        st.warning(f"Please fill in the following fields: {',  '.join(empty_fields)}")
    else:
        def is_float(value):
            try:
                float(value)
                return True
            except ValueError:
                return False
        
        for i in input_values.keys():
            if input_values[i].isnumeric():
                input_values[i] = int(input_values[i])
                print(input_values[i])
                continue
            if is_float(input_values[i]):
                input_values[i] = float(input_values[i])
                print(input_values[i])
                continue

        # Sample DataFrame
        get_values = input_values
        df = pd.DataFrame([get_values])

        # Replace specific values in the DataFrame
        df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
        df['Chest Pain Type'] = df['Chest Pain Type'].map({'ASY': 0, 'NAP': 1, 'ATA': 2, 'TA': 3})
        df['Fasting BS'] = df['Fasting BS'].map({'0 Blood sugar < 120 mg': 0, '1 Blood sugar > 120 mg': 1})
        df['Resting ECG'] = df['Resting ECG'].map({'Normal': 0, 'LVH': 1, 'ST': 2})
        df['Exercise Angina'] = df['Exercise Angina'].map({'No': 0, 'Yes': 1})
        df['St Slope'] = df['St Slope'].map({'Flat': 0, 'Up': 1, 'Down': 2})
        
        #mixmaxscale
        model = load_model('Model.hdf5')
        prediction = model.predict(df)
        print(prediction)
        if (prediction[0][0] > 0.5):
            st.error("The person has Heart's disease!!")
        else:
            st.info("The person is safe from Heart's disease")



