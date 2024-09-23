import streamlit as st
import pandas as pd
import numpy as np
from asapdiscovery.ml.inference import GATInference
from asapdiscovery.ml.models import ASAPMLModelRegistry
from rdkit import Chem
from streamlit_ketcher import st_ketcher


def _is_valid_smiles(smi):
    if smi is None or smi == "":
        return False
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return False
        else:
            return True
    except:
        return False


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


# Set the title of the Streamlit app
st.title('ASAPDiscovery Machine Learning')

input = st.selectbox("How would you like to enter your input?", ["Upload a CSV file", "Draw a molecule", "Enter SMILES"])

if input == "Draw a molecule":
    st.write("Draw a molecule")
    smiles = st_ketcher(None)
    if _is_valid_smiles(smiles):
        st.success("Valid SMILES string", icon="✅")
    else:
        st.error("Invalid SMILES string", icon="🚨")
        st.stop()
    smiles = [smiles]
    df = pd.DataFrame(smiles, columns=["SMILES"])
    smiles_column = df["SMILES"]
elif input == "Enter SMILES":
    st.write("Enter SMILES")
    smiles = st.text_input("Enter a SMILES string")
    if _is_valid_smiles(smiles):
        st.success("Valid SMILES string", icon="✅")
    else:
        st.error("Invalid SMILES string", icon="🚨")
        st.stop()
    smiles = [smiles]
    df = pd.DataFrame(smiles, columns=["SMILES"])
    smiles_column = df["SMILES"]
elif input == "Upload a CSV file":
    st.write("Upload a CSV file")

    # Create a file uploader for CSV files
    uploaded_file = st.file_uploader("Choose a CSV file to upload your predictions to", type="csv")

    # If a file is uploaded, parse it into a DataFrame
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()
    # Select a column from the DataFrame
    column = st.selectbox("Select a column of SMILES analyze", df.columns)

    smiles_column = df[column]

    # check if the smiles are valid
    valid_smiles = [_is_valid_smiles(smi) for smi in smiles_column]
    if not all(valid_smiles):
        st.error("Some of the SMILES strings are invalid, please check the input", icon="🚨")
        st.stop()
    st.success("All SMILES strings are valid, proceeding with prediction", icon="✅")

    

targets = ASAPMLModelRegistry.get_targets_with_models()
# filter out None values
targets = [t for t in targets if t is not None]
# Select a target value from the preset list
target_value = st.selectbox("Select a biological target ", targets)
# endpoints
endpoints = ASAPMLModelRegistry.get_endpoints()
# Select a target value from the preset list
endpoint_value = st.selectbox("Select a property ", endpoints)

if not ASAPMLModelRegistry.endpoint_has_target(endpoint_value):
    _target = None
    _global_model = True
    _target_str = "global"
else:
    _target = target_value
    _target_str = target_value
# Get the latest model for the target and endpoint
model = ASAPMLModelRegistry.get_latest_model_for_target_type_and_endpoint(_target, "GAT", endpoint_value)
if model is None:
    st.write(f"No model found for {target_value} {endpoint_value}")
    st.stop()
    # retry with a different target or endpoint


st.write(f"Predicting {_target_str} {endpoint_value} using model {model.name}")
# Create a GATInference object from the model
infr = GATInference.from_ml_model_spec(model)
if infr.is_ensemble:
    st.write(f"Ensemble model with {len(model.models)} models, will estimate uncertainty using ensemble variance")
# Predict the property value for each SMILES string
predictions = [infr.predict_from_smiles(smiles, return_err=True) for smiles in smiles_column]
predictions = np.asarray(predictions)
# check if second column is all np.nan
if np.all(np.isnan(predictions[:, 1])):
    preds = predictions[:, 0]
    err = None
else:
    preds = predictions[:, 0]
    err = predictions[:, 1] # rejoin with the original dataframe


df["predictions"] = preds
df["prediction_error"] = err


# allow the user to download the predictions
csv = convert_df(df)
st.download_button(
     label="Download data as CSV",
     data=csv,
     file_name=f"predictions_{model.name}.csv",
     mime="text/csv",
 )



    
    




# else:
#     st.write("Please upload a CSV file to view its contents.")
