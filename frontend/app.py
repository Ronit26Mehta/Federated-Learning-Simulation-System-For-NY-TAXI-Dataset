import streamlit as st
import os
import subprocess
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("Dataset & Visualization", "Federated Server", "Client 1", "Client 2")
)

# Dataset and Visualization Page
if page == "Dataset & Visualization":
    st.title("Federated Learning with Flower - Dataset & Visualization")
    
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    
    if uploaded_file:
        # Save and read the dataset
        dataset_path = "uploaded_dataset.csv"
        with open(dataset_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Dataset uploaded successfully!")
        
        df = pd.read_csv(dataset_path)
        st.write("Preview of the dataset:")
        st.dataframe(df.head())
        
        st.write("Columns in the dataset:")
        st.write(df.columns.tolist())

        # Preprocessing: Label Encoding for categorical columns
        label_encoder = LabelEncoder()
        categorical_columns = ["rate_code", "store_and_fwd_flag", "payment_type", 
                               "pickup_location_id", "dropoff_location_id", "day_of_week"]
        for col in categorical_columns:
            if col in df.columns:
                df[col] = label_encoder.fit_transform(df[col].astype(str))

        st.subheader("Data Visualization")
        features = [
            "trip_distance", "rate_code", "store_and_fwd_flag", "payment_type", "fare_amount", 
            "extra", "mta_tax", "tip_amount", "tolls_amount", "imp_surcharge", "pickup_location_id", 
            "dropoff_location_id", "year", "month", "day", "day_of_week", "hour_of_day", "trip_duration"
        ]
        selected_feature = st.selectbox("Select a feature to visualize", features)

        # Histogram
        st.write(f"### Histogram for {selected_feature}")
        fig = px.histogram(df, x=selected_feature, nbins=20, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Box Plot
        st.write(f"### Box Plot for {selected_feature}")
        fig = px.box(df, y=selected_feature, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# Federated Server Page
elif page == "Federated Server":
    st.title("Federated Learning with Flower - Server Operations")
    
    if st.button("Start Federated Server"):
        st.write("Starting Federated Server...")
        with open("server.log", "w") as log_file:
            subprocess.Popen(
                ["python", "fed-learn.py", "server"],
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
        st.success("Server started successfully! Check logs for progress.")
    
    st.subheader("Server Logs")
    if st.button("Refresh Logs"):
        if os.path.exists("server.log"):
            with open("server.log", "r") as log_file:
                logs = log_file.read()
                st.text_area("Server Logs", logs, height=300)
        else:
            st.write("No logs available yet.")

# Client 1 Page
elif page == "Client 1":
    st.title("Federated Learning with Flower - Client 1")
    
    if st.button("Start Client 1"):
        st.write("Starting Client 1...")
        with open("client1.log", "w") as log_file:
            subprocess.Popen(
                ["python", "fed-learn.py", "client1"],
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
        st.success("Client 1 started successfully! Check logs for progress.")
    
    st.subheader("Client 1 Logs")
    if st.button("Refresh Logs"):
        if os.path.exists("client1.log"):
            with open("client1.log", "r") as log_file:
                logs = log_file.read()
                st.text_area("Client 1 Logs", logs, height=300)
        else:
            st.write("No logs available yet.")

# Client 2 Page
elif page == "Client 2":
    st.title("Federated Learning with Flower - Client 2")
    
    if st.button("Start Client 2"):
        st.write("Starting Client 2...")
        with open("client2.log", "w") as log_file:
            subprocess.Popen(
                ["python", "fed-learn.py", "client2"],
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
        st.success("Client 2 started successfully! Check logs for progress.")
    
    st.subheader("Client 2 Logs")
    if st.button("Refresh Logs"):
        if os.path.exists("client2.log"):
            with open("client2.log", "r") as log_file:
                logs = log_file.read()
                st.text_area("Client 2 Logs", logs, height=300)
        else:
            st.write("No logs available yet.")
