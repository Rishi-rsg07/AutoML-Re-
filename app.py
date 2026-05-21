import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="Auto-Streamline Engine", layout="centered")
st.title("⚙️ Auto-Streamline")
st.subheader("Minimalist No-Code Tabular & Text Training")

uploaded_file = st.file_uploader("Drop your dataset down below (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview (First 5 Rows)")
    st.dataframe(df.head())
    
    # Expose interactive parameter binding to users dynamically
    columns = df.columns.tolist()
    target_col = st.selectbox("Select Target Variable (Y labels)", options=columns)
    
    remaining_cols = [c for c in columns if c != target_col]
    text_cols = st.multiselect("Identify explicit text content blocks (optional/TF-IDF)", options=remaining_cols)
    
    if st.button("Trigger Training Run"):
        with st.spinner("Executing structural pipeline, tuning architecture parameters..."):
            # Reset file pointer read stream before posting data away
            uploaded_file.seek(0)
            
            payload = {
                "target": target_col,
                "text_cols": json.dumps(text_cols)
            }
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            
            try:
                response = requests.post("http://backend:8000/api/v1/train", data=payload, files=files)
                result = response.json()
                
                if result.get("status") == "Success":
                    st.success(" AutoML Execution Finished Successfully!")
                    
                    st.json(result["evaluation_metrics"])
                    
                    # Direct binary model artifact recovery link setup
                    download_url = f"http://localhost:8000{result['download_route']}"
                    st.markdown(f"[📥 Download Trained Model Wrapper]({download_url})")
                else:
                    st.error(f"Execution Error: {result.get('detail')}")
            except Exception as ex:
                st.error(f"Could not reach server pipeline nodes: {ex}")