import numpy as np
import struct
import pywt
import time
import os
import pandas as pd
import warnings

from scipy import signal
from PyEMD import EEMD, EMD
from statsmodels.tsa.arima.model import ARIMA
from multiprocessing import Pool
from tqdm import tqdm

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

# Features extractor. 103 per signal
def get_features(path):
  with open(path, 'rb') as f:
    data = f.read()

  arr = np.array(list(data), dtype=np.uint8)[20:]
  data = np.array(struct.unpack('h' * int(arr.shape[0] / 2), arr.tobytes()))

  sr = 23437.5
  t = np.linspace(0, data.shape[0] / sr, data.shape[0], endpoint=False)

  b_notch, a_notch = signal.iirnotch(60.0, 30.0, sr)
  data_notched = signal.filtfilt(b_notch, a_notch, data)

  b, a = signal.butter(3, 30, 'hp', fs=sr)
  data_blo = signal.filtfilt(b, a, data_notched)

  wavelet_type='db4'
  level = 4
  DWTcoeffs = pywt.wavedec(data_blo, wavelet_type, mode='per', level=level, axis=-1)
  thresh = 0.63 * np.nanmax(data)
  DWTcoeffs[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in DWTcoeffs[1:])

  data_dwt = pywt.waverec(DWTcoeffs, wavelet_type, mode='per', axis=-1) 
  
  for i in range(10, 25):
    emd = EMD(FIXE=i)
    IMFs = emd.emd(data_dwt, t, max_imf=13)

    if IMFs.shape[0] == 14:
      break

  nIMFs = IMFs.shape[0] - 1

  # 10 epochs
  q = t.shape[0] % 10

  # mean filtered signal
  mdata_dwt = np.mean(np.split(data_dwt[:-q], 10), axis=0)

  # mean IMFs
  mIMFs = np.mean(np.split(IMFs[:-1, :-q], 10, axis=1), axis=0)
  mt = np.split(t[:-q], 10)[0]

  # Time features
  zc_map = np.array([0, 1, 2, 5, 6, 7, 8, 9, 10, 11])
  ssc_map = np.array([0, 1, 4, 5])

  ar_map = {0: np.array([0, 1, 3]),
            1: np.array([1, 2]),
            2: np.array([2]),
            3: np.array([3]),
            4: np.array([4]),
            6: np.array([0]),
            7: np.array([1]),
            8: np.array([2]),
            12: np.array([0])}

  wl = np.sum(np.abs(np.diff(mIMFs)), axis=1)
  zc = np.sum(np.abs(np.diff(np.sign(mIMFs), axis=1)) == 2, axis=1)[zc_map]
  ssc = np.sum(np.abs(np.diff(np.sign(np.diff(mIMFs, axis=1)), axis=1)) == 2, axis=1)[ssc_map]
  rms = np.sqrt(np.mean(mIMFs ** 2, axis=1))

  ar = []
  for i, imf in enumerate(mIMFs):
    # print(f'Running imf_{i+1:d}:')
    if i not in ar_map:
      # print('\tskipping')
      continue
    else:
      try:
        _ar = ARIMA(imf, order=[6, 0, 0]).fit().polynomial_ar[1:][ar_map[i]]
      except:
        _ar = np.zeros(6)[ar_map[i]]
      ar.append(_ar)
  ar = np.concatenate(ar)
      
  mav = np.mean(np.abs(mIMFs), axis=1)
  iav = np.sum(np.abs(mIMFs), axis=1)

  f_feat = np.concatenate([wl, zc, ssc, rms, ar, mav, iav])

  # Freq - hilbert features
  analytic_sig = np.zeros_like(mIMFs, dtype=np.complex64)

  mif = np.zeros(nIMFs)
  mfd = np.zeros(nIMFs)
  smpds = np.zeros(nIMFs)
  amb = np.zeros(nIMFs)
  fmb = np.zeros(nIMFs)
  ppsd = np.zeros(nIMFs)

  amb_map = np.arange(5)
  fmb_map = np.arange(4)
  smpds_map = np.array([0, 1, 2, 3, 4, 7])
  mfd_map = np.array([3, 4])
  mif_map = np.array([0, 1, 2, 3, 5])
  ppsd_map = np.array([0, 3, 7])

  for i, imf in enumerate(mIMFs):
      # no queda claro en el paper. Revisar cálculo de amplitud, fase y frecuencia
      # ver https://ccrma.stanford.edu/~jos/st/Analytic_Signals_Hilbert_Transform.html
      analytic_sig = signal.hilbert(imf)
      amps = np.abs(analytic_sig)
      phases = np.angle(analytic_sig)
      
      freqs = np.zeros_like(phases)
      freqs[1:] = np.diff(phases)
      
      psd = freqs ** 2 / (2 * np.pi)
      
      mif[i] = np.sum(freqs * amps ** 2) / np.sum(amps ** 2)
      mfd[i] = np.sum(np.diff(freqs)) / (freqs.shape[0])
      smpds[i] = np.sum(np.arange(1, psd.shape[0] + 1) * psd)
      amb[i] = np.sum(np.diff(amps ** 2)) / np.sum(psd[:-1]) # así sale en el paper (?)
      fmb[i] = np.sum((freqs - mif[i]) * amps ** 2) / np.sum(psd)
      ppsd[i] = np.max(psd)
      
  mif = mif[mif_map]
  mfd = mfd[mfd_map]
  smpds = smpds[smpds_map]
  amb = amb[amb_map]
  fmb = fmb[fmb_map]
  ppsd = ppsd[ppsd_map]

  t_feat = np.concatenate([mif, mfd, smpds, amb, fmb, ppsd])

  features = np.concatenate([f_feat, t_feat])

  return features

num_threads = int(os.cpu_count() / 2)
pop_feat = []

with warnings.catch_warnings():
  warnings.filterwarnings('ignore')
  for s in tqdm(s_paths):
    pop_feat.append(get_features(s))

  # with Pool(num_threads) as pool:
  #   pop_feat = list(tqdm(pool.imap(get_features, s_paths), total=len(s_paths)))

df = pd.DataFrame(pop_feat)
df['class'] = classes
df['subject'] = names

df.to_csv('features.csv')