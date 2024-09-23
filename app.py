import streamlit as st
import os

asap_prod_streamlit = int(os.getenv("ASAP_OE_PROD_STREAMLIT", None))

if asap_prod_streamlit == 1:
    def sort_out_openeye_license():
        import os
        # need to write the license file to disk
        txt = st.secrets.openeye_credentials.license_file_txt
        if not txt:
            raise ValueError("No OpenEye license file found")
        with open("oe_license.txt", "w") as f:
            f.write(txt)
        # set the license file environment variable
        os.environ["OE_LICENSE"] = "oe_license.txt"

    sort_out_openeye_license()



import pandas as pd
import numpy as np
from asapdiscovery.ml.inference import GATInference
from asapdiscovery.ml.models import ASAPMLModelRegistry
from rdkit import Chem
from streamlit_ketcher import st_ketcher
from io import StringIO
import schedule

# need to update the registry periodically
schedule.every(4).hours.do(ASAPMLModelRegistry.update_registry)

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
    
def sdf_str_to_rdkit_mol(sdf):
    from io import BytesIO
    bio = BytesIO(sdf.encode())
    suppl = Chem.ForwardSDMolSupplier(bio, removeHs=False)
    mols = [mol for mol in suppl if mol is not None]
    return mols


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")



    

# Set the title of the Streamlit app
st.title('ASAPDiscovery Machine Learning')

st.markdown("## Intro")

st.markdown("The [ASAPDiscovery antiviral drug discovery consortium](https://asapdiscovery.org) has developed a series of machine learning models (primarily Graph Attention Networks (GATs)) to predict molecular properties based on our experimental data, much of which is [available](https://asapdiscovery.org/outputs/) as part of our [open science](https://asapdiscovery.org/open-science/) and public disclosure policy.")
st.markdown("These models are trained on a variety of endpoints, including in-vitro activity, assayed LogD, and more \n Some models are specific to a target, while others are global models that predict properties across all targets.")
st.markdown("This web app gives you easy API-less access to the models, I hope you find it useful!\n As scientists we should always be looking to get our models into people's hands as easily as possible.")
st.markdown("These models are trained bi-weekly. The latest models are used for prediction where possible. Note that predictions are pre-alpha and are provided as is, we are still working very actively on improving and validating models.")

st.markdown("## Select input")


input = st.selectbox("How would you like to enter your input?", ["Upload a CSV file", "Draw a molecule", "Enter SMILES", "Upload an SDF file"])

multismiles = False
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
    column = "SMILES"
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
    column = "SMILES"
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
    multismiles = True
    smiles_column = df[column]

    # check if the smiles are valid
    valid_smiles = [_is_valid_smiles(smi) for smi in smiles_column]
    if not all(valid_smiles):
        st.error("Some of the SMILES strings are invalid, please check the input", icon="🚨")
        st.stop()
    st.success("All SMILES strings are valid, proceeding with prediction", icon="✅")

elif input == "Upload an SDF file":
    st.write("Upload an SDF file")
    # Create a file uploader for SDF files
    uploaded_file = st.file_uploader("Choose a SDF file to upload your predictions to", type="sdf")
    # read with rdkit
    if uploaded_file is not None:
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        mols = sdf_str_to_rdkit_mol(string_data)
        smiles = [Chem.MolToSmiles(m) for m in mols]
        df = pd.DataFrame(smiles, columns=["SMILES"])
            # st.error("Error reading the SDF file, please check the input", icon="🚨")
            # st.stop()
    else:
        st.stop()
    
    st.success("All SMILES strings are valid, proceeding with prediction", icon="✅")
    column = "SMILES"
    smiles_column = df["SMILES"]
    multismiles = True

st.markdown("## Select your model parameters")


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

st.markdown("## Prediction time 🚀")


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

# sort the dataframe by predictions
df = df.sort_values(by="predictions", ascending=False)

if multismiles:
    # plot the predictions and errors
    st.scatter_chart(df, x=column, y="predictions", color="prediction_error", use_container_width=True, x_label="SMILES", y_label=f"Predicted {_target_str} {endpoint_value} ")

else:
    # just print the prediction
    preds = df["predictions"].values[0]
    smiles = df["SMILES"].values[0]
    if err:
        err = df["prediction_error"].values[0]
        errstr = f"± {err:.2f}"
    else:
        errstr = ""
    
    st.markdown("### 🕵️")
    st.markdown(f"Predicted {_target_str} {endpoint_value} for {smiles} is {preds:.2f} {errstr} using model {infr.model_name}")

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
