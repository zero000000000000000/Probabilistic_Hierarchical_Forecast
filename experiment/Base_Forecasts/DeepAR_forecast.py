import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput

data = pd.read_csv('d:\HierarchicalCode\experiment\Data\Tourism\Tourism_process.csv')
prediction_length = 12
print(data.head())