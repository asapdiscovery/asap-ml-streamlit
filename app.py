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
import matplotlib.pyplot as plt
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
st.title("ASAP Discovery Local Models (ML)")

st.markdown("## Background")

st.markdown(
    "**The [ASAP Discovery antiviral drug discovery consortium](https://asapdiscovery.org) has developed a series of local machine learning models (GAT architecture) to predict properties based on our local data, much of which is [available](https://asapdiscovery.org/outputs/) as part of our [open science policy](https://asapdiscovery.org/open-science/).**"
)
st.markdown(
    "**These models are trained on a variety of experimental endpoints that are found in ASAP's CDD vault, including biochemical and antiviral potency, assayed LogD, and more. Some models are specific to a target, while others are global models that predict properties across all targets.**"
)
st.markdown(
    "This web app gives you easy access to the trained models without having to write or execute any code. The intention is to empower anyone across ASAP to make these predictions."
)
st.markdown("---")
st.markdown(
    "These models are trained bi-weekly. The latest models are used for prediction where possible. Note that predictions are pre-alpha and are provided as is, work is on-going on improving and validating these models. As a general rule of thumb, predictions on your data will be better when your query compound(s) is/are closer chemically to the compounds in the CDD. Are you having problems using this UI or do you have a feature request? Please open an issue on [our issue tracker](https://github.com/asapdiscovery/asap-ml-streamlit/issues/new)."
)

st.markdown("## Input :clipboard:")


input = st.selectbox(
    "How would you like to enter your input?",
    ["Upload a CSV file", "Draw a molecule", "Enter SMILES", "Upload an SDF file"],
)

multismiles = False
if input == "Draw a molecule":
    smiles = st_ketcher(None)
    if _is_valid_smiles(smiles):
        st.success("Valid molecule", icon="✅")
    else:
        st.error("Invalid molecule", icon="🚨")
        st.stop()
    smiles = [smiles]
    queried_df = pd.DataFrame(smiles, columns=["SMILES"])
    smiles_column_name = "SMILES"
    smiles_column = queried_df[smiles_column_name]
elif input == "Enter SMILES":
    smiles = st.text_input("Enter a SMILES string")
    if _is_valid_smiles(smiles):
        st.success("Valid SMILES string", icon="✅")
    else:
        st.error("Invalid SMILES string", icon="🚨")
        st.stop()
    smiles = [smiles]
    queried_df = pd.DataFrame(smiles, columns=["SMILES"])
    smiles_column_name = "SMILES"
    smiles_column = queried_df[smiles_column_name]
elif input == "Upload a CSV file":
    # Create a file uploader for CSV files
    uploaded_file = st.file_uploader(
        "Choose a CSV file to upload your predictions to", type="csv"
    )

    # If a file is uploaded, parse it into a DataFrame
    if uploaded_file is not None:
        queried_df = pd.read_csv(uploaded_file)
    else:
        st.stop()
    # Select a column from the DataFrame
    smiles_column_name = st.selectbox("Select a SMILES column", queried_df.columns)
    multismiles = True
    smiles_column = queried_df[smiles_column_name]

    # check if the smiles are valid
    valid_smiles = [_is_valid_smiles(smi) for smi in smiles_column]
    if not all(valid_smiles):
        st.error(
            "Some of the SMILES strings are invalid, please check the input", icon="🚨"
        )
        st.stop()
    st.success(
        f"All SMILES strings are valid (n={len(valid_smiles)}), proceeding with prediction",
        icon="✅",
    )

elif input == "Upload an SDF file":
    # Create a file uploader for SDF files
    uploaded_file = st.file_uploader(
        "Choose a SDF file to upload your predictions to", type="sdf"
    )
    # read with rdkit
    if uploaded_file is not None:
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        mols = sdf_str_to_rdkit_mol(string_data)
        smiles = [Chem.MolToSmiles(m) for m in mols]
        queried_df = pd.DataFrame(smiles, columns=["SMILES"])
        # st.error("Error reading the SDF file, please check the input", icon="🚨")
        # st.stop()
    else:
        st.stop()

    st.success(
        f"All molecule entries are valid (n={len(queried_df)}), proceeding with prediction",
        icon="✅",
    )
    smiles_column_name = "SMILES"
    smiles_column = queried_df[smiles_column_name]
    multismiles = True

st.markdown("## Model parameters :nut_and_bolt:")


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
model = ASAPMLModelRegistry.get_latest_model_for_target_type_and_endpoint(
    _target, "GAT", endpoint_value
)
if model is None:
    st.write(f"No model found for {target_value} {endpoint_value}")
    st.stop()
    # retry with a different target or endpoint

st.markdown("## Prediction 🚀")


st.write(
    f"Predicting **{_target_str} {endpoint_value}** using model:\n\n `{model.name}`"
)
# Create a GATInference object from the model
infr = GATInference.from_ml_model_spec(model)
if infr.is_ensemble:
    st.write(
        f"_Using ensemble model (n={len(model.models)}); estimating uncertainty as variance of predictions._"
    )
# Predict the property value for each SMILES string
predictions = [
    infr.predict_from_smiles(smiles, return_err=True) for smiles in smiles_column
]
predictions = np.asarray(predictions)
# check if second column is all np.nan
if np.all(np.isnan(predictions[:, 1])):
    preds = predictions[:, 0]
    err = None
else:
    preds = predictions[:, 0]
    err = predictions[:, 1]  # rejoin with the original dataframe


pred_column_name = f"{_target_str}_computed-{endpoint_value}"
unc_column_name = f"{_target_str}_computed-{endpoint_value}_uncertainty"
queried_df[pred_column_name] = preds
queried_df[unc_column_name] = err

st.markdown("---")
if multismiles:
    # plot the predictions and errors
    # Histogram first
    fig, ax = plt.subplots()

    sorted_df = queried_df.sort_values(by=pred_column_name)
    n_bins = int(len(sorted_df[pred_column_name]) / 10)
    if n_bins < 5:  # makes the histogram slightly more interpretable with low data
        n_bins = 5

    ax.hist(sorted_df[pred_column_name], bins=n_bins)

    ax.set_ylabel("Count")
    ax.set_xlabel(f"Computed {endpoint_value}")
    ax.set_title(f"Histogram of computed {endpoint_value} for target: {_target_str}")

    st.pyplot(fig)

    # then a barplot
    fig, ax = plt.subplots()

    ax.bar(range(len(sorted_df)), sorted_df[pred_column_name])

    ax.set_xticks([])
    ax.set_xlabel(f"Query compounds")
    ax.set_ylabel(f"Computed {endpoint_value}")

    ax.set_title(f"Barplot of computed {endpoint_value} for target: {_target_str}")

    st.pyplot(fig)

    if endpoint_value == "pIC50":
        from rdkit.Chem.Descriptors import MolWt
        import seaborn as sns

        # then a scatterplot of uncertainty vs MW
        queried_df["MW"] = [
            MolWt(Chem.MolFromSmiles(smi)) for smi in sorted_df[smiles_column_name]
        ]
        fig, ax = plt.subplots()

        ax = sns.scatterplot(
            x="MW",
            y=pred_column_name,
            hue=unc_column_name,
            palette="coolwarm",
            data=queried_df,
        )

        norm = plt.Normalize(
            queried_df[unc_column_name].min(), queried_df[unc_column_name].max()
        )
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        cbar = ax.figure.colorbar(sm, ax=ax)
        ax.annotate(
            f"Computed {endpoint_value} uncertainty",
            xy=(1.2, 0.3),
            xycoords="axes fraction",
            rotation=270,
        )

        ax.set_title(
            f"Scatterplot of predicted {endpoint_value} versus MW\ntarget: {_target_str}"
        )
        ax.set_xlabel(f"Molecular weight (Da)")
        ax.set_ylabel(f"Computed {endpoint_value}")
        st.pyplot(fig)

else:
    # just print the prediction
    preds = queried_df[pred_column_name].values[0]
    smiles = queried_df["SMILES"].values[0]
    if err:
        err = queried_df[unc_column_name].values[0]
        errstr = f"± {err:.2f}"
    else:
        errstr = ""

    st.markdown(
        f"Predicted {_target_str} {endpoint_value} for {smiles} is {preds:.2f} {errstr}."
    )

# allow the user to download the predictions
csv = convert_df(queried_df)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name=f"predictions_{model.name}.csv",
    mime="text/csv",
)
