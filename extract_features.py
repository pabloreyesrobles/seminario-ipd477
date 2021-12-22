import numpy as np
import os
import pandas as pd
import warnings

from multiprocessing import Pool
from tqdm import tqdm
from features import *

if __name__ == "__main__":
  names = []
  classes = []
  s_paths = []
  muscles = []

  for group in os.listdir(f'../N2001/'):
    for muscle in os.listdir(f'../N2001/{group:s}'):
      for subject in os.listdir(f'../N2001/{group:s}/{muscle:s}'):
        # print(f'../N2001/{group:s}/{muscle:s}/{subject:s}/{subject:s}.bin')

        names.append(subject)
        classes.append(group)
        muscles.append(muscle)

        s_paths.append(f'../N2001/{group:s}/{muscle:s}/{subject:s}')

  num_threads = int(os.cpu_count() / 2)
  preload = False

  with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    with Pool(num_threads) as pool:
      args = [(name, path, preload) for name, path in zip(names, s_paths)]
      pop_feat = np.array(list(tqdm(pool.imap(get_features, args[:3]), total=len(args[:3]))))
      df = pd.DataFrame(pop_feat)
  
  columns = [f'WL_imf{i:d}' for i in range(1, 14)] + \
            [f'ZC_imf{i+1:d}' for i in zc_map] + \
            [f'SSC_imf{i+1:d}' for i in ssc_map] + \
            [f'RMS_imf{i:d}' for i in range(1, 14)] + \
            [f'AR{j+1}_imf{i+1:d}' for i in ar_map.keys() for j in ar_map[i]] + \
            [f'MAV_imf{i:d}' for i in range(1, 14)] + \
            [f'IAV_imf{i:d}' for i in range(1, 14)] + \
            [f'AMB_imf{i+1:d}' for i in amb_map] + \
            [f'FMB_imf{i+1:d}' for i in fmb_map] + \
            [f'SMPDS_imf{i+1:d}' for i in smpds_map] + \
            [f'MFD_imf{i+1:d}' for i in mfd_map] + \
            [f'MIF_imf{i+1:d}' for i in mif_map] + \
            [f'PPSD_imf{i+1:d}' for i in ppsd_map]
  df.columns = columns

  df['class'] = classes[:3]
  df['subject'] = names[:3]
  df['muscles'] = muscles[:3]

  cols = list(df.columns)
  df = df[cols[-3:] + cols[:-3]]

  df.to_csv('_features.csv')