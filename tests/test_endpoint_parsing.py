import numpy as np
import pandas as pd
from hetero_data import parse_mutation_series, canonical_subtype_label

def test_mutation_calls_preserve_missingness():
    result=parse_mutation_series(pd.Series(['0','H1047R',None,'E545K G1049R',0,1.0,'nan']))
    np.testing.assert_array_equal(np.nan_to_num(result,nan=-1),[0,1,-1,1,0,1,-1])

def test_subtype_aliases():
    assert canonical_subtype_label('BRCA_LumB')=='Luminal_B'
    assert canonical_subtype_label('claudin-low')=='Other'
