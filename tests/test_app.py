import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

class STTester:
    timeout = 500


@pytest.fixture()
def app_path():
    return "../app.py"

@pytest.fixture()
def smiles_dataframe_data():
    # synthetic data with multiple columns and SMILES
    data = {
        "mySmiles": ["CC", "CCC", "CCCC"],
        "blah": ["1", "2", "3"],
        "bleh": ["a", "b", "c"],
    }
    return pd.DataFrame(data)

@pytest.fixture()
def smiles_dataframe_data_csv(tmp_path, smiles_dataframe_data):
    # synthetic data with multiple columns and SMILES
    csv_path = tmp_path / "smiles_data.csv"
    smiles_dataframe_data.to_csv(csv_path, index=False)
    return csv_path



class TestSMILES(STTester):

    @pytest.mark.parametrize("target", ["SARS-CoV-2-Mpro", "MERS-CoV-Mpro"])
    @pytest.mark.parametrize("endpoint", ["pIC50", "LogD"])
    def test_smiles(self, app_path, endpoint, target):
        at = AppTest.from_file(app_path)
        at.run(timeout=self.timeout)
        at.selectbox(key="input").select("Enter SMILES").run(timeout=self.timeout)
        at.text_input(key="smiles_user_input").input("CC").run(timeout=self.timeout)
        at.selectbox(key="target").select(target).run(timeout=self.timeout)
        at.selectbox(key="endpoint").select(endpoint).run(timeout=self.timeout)
        val = at.markdown[-1].value # last markdown
        assert "CC" in val
        if not endpoint == "LogD":
            assert target in val
        else:
            assert "global" in val
        assert endpoint in val
        assert at.success




class TestDataframe(STTester):

    @pytest.mark.xfail(reason="No ability to mock file upload, see https://github.com/streamlit/streamlit/issues/8438")
    @pytest.mark.parametrize("target", ["SARS-CoV-2-Mpro", "MERS-CoV-Mpro"])
    @pytest.mark.parametrize("endpoint", ["pIC50", "LogD"])
    def test_dataframe(self, app_path, smiles_dataframe_data_csv, target, endpoint):
        at = AppTest.from_file(app_path)
        at.run(timeout=self.timeout)
        at.selectbox(key="input").select("Upload a CSV file").run(timeout=self.timeout)
        # cant be bother to mock testing internals to get this to work
        # you can also possibly use selenium to do this but seems like a lot of work
        at.file_uploader(key="csv_file").upload(smiles_dataframe_data).run(timeout=self.timeout)
        at.selectbox(key="df_smiles_column").select("mySmiles").run(timeout=self.timeout)
        at.selectbox(key="target").select("SARS-CoV-2-Mpro").run(timeout=self.timeout)
        at.selectbox(key="endpoint").select("pIC50").run(timeout=self.timeout)
        assert at.success
