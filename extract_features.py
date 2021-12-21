import numpy as np
import os
import pandas as pd
import warnings

from multiprocessing import Pool
from tqdm import tqdm
from features import get_features

if __name__ == "__main__":
  names = []
  classes = []
  s_paths = []

  for group in os.listdir(f'../N2001/'):
    for muscle in os.listdir(f'../N2001/{group:s}'):
      for subject in os.listdir(f'../N2001/{group:s}/{muscle:s}'):
        # print(f'../N2001/{group:s}/{muscle:s}/{subject:s}/{subject:s}.bin')

        names.append(subject)
        classes.append(group)
        s_paths.append(f'../N2001/{group:s}/{muscle:s}/{subject:s}/{subject:s}.bin')

  num_threads = int(os.cpu_count() / 2)

  with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    with Pool(num_threads) as pool:
      pop_feat = list(tqdm(pool.imap(get_features, s_paths), total=len(s_paths)))  
      df = pd.DataFrame(pop_feat)

  df['class'] = classes
  df['subject'] = names

  df.to_csv('features.csv')