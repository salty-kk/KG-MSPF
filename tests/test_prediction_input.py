import pandas as pd
import pytest
from hetero_data import CohortTable


def test_prediction_input_does_not_require_future_outcome(tmp_path):
    p=tmp_path/'new_patient.csv'
    pd.DataFrame({'patient_id':['new_patient'],'age_at_diagnosis':[60],'expr:GATA3':[.5]}).to_csv(p,index=False)
    pred=CohortTable.from_harmonized_csv(p,require_outcomes=False)
    assert len(pred)==1
    with pytest.raises(ValueError,match='lacks columns'):
        CohortTable.from_harmonized_csv(p)
