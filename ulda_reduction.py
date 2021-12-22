import pandas as pd
import numpy as np
from ulda import ulda, ulda_feature_reduction

df = pd.read_csv('features.csv', index_col=0)
df_clean = df.dropna(how='any')

data_classes = pd.factorize(df_clean['class'])[0]
data_feat = np.array(df_clean[df_clean.columns[3:]])

ulda_feature_reduction(data_feat, data_classes, 2)