import requests
import streamlit as st
import pandas as pd

st.title("Receipt Inference Web Application")

st.write("The model predicts the approximate number of the scanned receipts for each month of the next year based on \
    the provided the number of the observed scanned receipts each day from the previous year. ")

# input 
display = st.select_slider(
    "Which model would you like to use for prediction?",
    options=["LSTM", "GRU"]
)


if st.button("Submit"):
    inputs = {
        "inputs" : [
            {"return": display}
        ]
    }

    m = inputs["inputs"][0]['return']


    # Posting inputs to ML API
    response = requests.post("http://host.docker.internal:8001/api/v1/predict?input_data={}".format(m), verify=False)
    json_response = response.json()

    predictions = json_response.get("predictions")[0]
    
    
    df = pd.DataFrame.from_records([predictions])


    st.table(df)


