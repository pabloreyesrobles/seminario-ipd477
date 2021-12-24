import numpy as np
import struct
import pywt
import os

from scipy import signal
from PyEMD import EMD
from statsmodels.tsa.arima.model import ARIMA

# Time features mapping
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

# Freq features mapping
amb_map = np.arange(5)
fmb_map = np.arange(4)
smpds_map = np.array([0, 1, 2, 3, 4, 7])
mfd_map = np.array([3, 4])
mif_map = np.array([0, 1, 2, 3, 5])
ppsd_map = np.array([0, 3, 7])

def split_features(split_IMFs, nIMFs):

  try:
    wl = np.sum(np.abs(np.diff(split_IMFs)), axis=2)
    zc = np.sum(np.abs(np.diff(np.sign(split_IMFs))) == 2, axis=2)[:, zc_map]
    ssc = np.sum(np.abs(np.diff(np.sign(np.diff(split_IMFs)))) == 2, axis=2)[:, ssc_map]
    rms = np.sqrt(np.mean(split_IMFs ** 2, axis=2))

    mav = np.mean(np.abs(split_IMFs), axis=2)
    iav = np.sum(np.abs(split_IMFs), axis=2)

    ar = []
    for i in range(nIMFs):
        if i not in ar_map:
          continue
        
        i_ar = []
        for imf in split_IMFs[:, i]:
          try:
            _ar = ARIMA(imf, order=[6, 0, 0]).fit(method='yule_walker').polynomial_ar[1:][ar_map[i]]
          except:
            _ar = np.zeros(6)[ar_map[i]]
          i_ar.append(_ar)        
        ar.append(np.array(i_ar)) 
    ar = np.concatenate(ar, axis=1)
        
    t_feat = np.concatenate([wl, zc, ssc, rms, ar, mav, iav], axis=1)

    # Freq - hilbert features
    mif = np.zeros([10, nIMFs])
    mfd = np.zeros([10, nIMFs])
    smpds = np.zeros([10, nIMFs])
    amb = np.zeros([10, nIMFs])
    fmb = np.zeros([10, nIMFs])
    ppsd = np.zeros([10, nIMFs])

    amb_map = np.arange(5)
    fmb_map = np.arange(4)
    smpds_map = np.array([0, 1, 2, 3, 4, 7])
    mfd_map = np.array([3, 4])
    mif_map = np.array([0, 1, 2, 3, 5])
    ppsd_map = np.array([0, 3, 7])

    for i in range(nIMFs):
      # no queda claro en el paper. Revisar cálculo de amplitud, fase y frecuencia
      # ver https://ccrma.stanford.edu/~jos/st/Analytic_Signals_Hilbert_Transform.html

      # IMF split params
      t_params = np.zeros([10, 6])
      for j, imf in enumerate(split_IMFs[:, i]):
        analytic_sig = signal.hilbert(imf)
        amps = np.abs(analytic_sig)
        phases = np.angle(analytic_sig)

        freqs = np.zeros_like(phases)
        freqs[1:] = np.diff(phases)

        psd = freqs ** 2 / (2 * np.pi)

        t_params[j, 0] = np.sum(freqs * amps ** 2) / np.sum(amps ** 2) # mif 
        t_params[j, 1] = np.sum(np.diff(freqs)) / (freqs.shape[0]) # mfd 
        t_params[j, 2] = np.sum(np.arange(1, psd.shape[0] + 1) * psd) # smpds 
        t_params[j, 3] = np.sum(np.diff(amps ** 2)) / np.sum(psd[:-1]) # amb, así sale en el paper
        t_params[j, 4] = np.sum((freqs - t_params[j, 0]) * amps ** 2) / np.sum(psd) # fmb 
        t_params[j, 5] = np.max(psd) # ppsd

      mif[:, i] = t_params[:, 0]
      mfd[:, i] = t_params[:, 1]
      smpds[:, i] = t_params[:, 2]
      amb[:, i] = t_params[:, 3]
      fmb[:, i] = t_params[:, 4]
      ppsd[:, i] = t_params[:, 5]

    mif = mif[:, mif_map]
    mfd = mfd[:, mfd_map]
    smpds = smpds[:, smpds_map]
    amb = amb[:, amb_map]
    fmb = fmb[:, fmb_map]
    ppsd = ppsd[:, ppsd_map]

    f_feat = np.concatenate([mif, mfd, smpds, amb, fmb, ppsd], axis=1)
    features = np.concatenate([t_feat, f_feat], axis=1)
  
  except:
    features = np.empty([10, 103])
    features[:] = np.nan

  return features

def mean_features(mIMFs, nIMFs):

  try:
    wl = np.sum(np.abs(np.diff(mIMFs)), axis=1)
    zc = np.sum(np.abs(np.diff(np.sign(mIMFs), axis=1)) == 2, axis=1)[zc_map]
    ssc = np.sum(np.abs(np.diff(np.sign(np.diff(mIMFs, axis=1)), axis=1)) == 2, axis=1)[ssc_map]
    rms = np.sqrt(np.mean(mIMFs ** 2, axis=1))

    ar = []
    for i, imf in enumerate(mIMFs):
      if i not in ar_map:
        continue
      else:
        try:
          _ar = ARIMA(imf, order=[6, 0, 0]).fit(method='yule_walker').polynomial_ar[1:][ar_map[i]]
        except:
          _ar = np.zeros(6)[ar_map[i]]
        ar.append(_ar)
    ar = np.concatenate(ar)
        
    mav = np.mean(np.abs(mIMFs), axis=1)
    iav = np.sum(np.abs(mIMFs), axis=1)

    t_feat = np.concatenate([wl, zc, ssc, rms, ar, mav, iav])

    # Freq - hilbert features
    analytic_sig = np.zeros_like(mIMFs, dtype=np.complex64)

    mif = np.zeros(nIMFs)
    mfd = np.zeros(nIMFs)
    smpds = np.zeros(nIMFs)
    amb = np.zeros(nIMFs)
    fmb = np.zeros(nIMFs)
    ppsd = np.zeros(nIMFs)

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

    f_feat = np.concatenate([mif, mfd, smpds, amb, fmb, ppsd])
    features = np.concatenate([t_feat, f_feat])
  
  except:
    features = np.empty(103)
    features[:] = np.nan

  return features
  
# Features extractor. 103 per signal
# args[0]: signal name
# args[1]: signal path
# args[2]: preload
# args[3]: mean_imfs
def get_features(args):

  path = f'{args[1]:s}/{args[0]:s}.npy'
  if os.path.isfile(path) and args[2] == True:
    features = np.load(path)
    if features.shape == (103,) or features.shape == (10, 103):
      return features

  path = f'{args[1]:s}/{args[0]:s}.bin'
  with open(path, 'rb') as f:
    data = f.read()

  arr = np.array(list(data), dtype=np.uint8)[20:]
  data = np.array(struct.unpack('h' * int(arr.shape[0] / 2), arr.tobytes()))
  data = data / 10 # according to papers values. It is not specified

  if data.shape[0] != 262134:
    if args[3]: features = np.empty(103)
    else: features = np.empty([10, 103])

    features[:] = np.nan
    np.save(f'{args[1]:s}/{args[0]:s}.npy', features)

    return features

  sr = 23437.5
  t = np.linspace(0, data.shape[0] / sr, data.shape[0], endpoint=False)

  b_notch, a_notch = signal.iirnotch(50.0, 30.0, sr)
  data_notched = signal.filtfilt(b_notch, a_notch, data)

  b, a = signal.butter(3, 30, 'hp', fs=sr)
  data_blo = signal.filtfilt(b, a, data_notched)

  wavelet_type='db4'
  level = 4
  DWTcoeffs = pywt.wavedec(data_blo, wavelet_type, mode='per', level=level, axis=-1)
  thresh = 0.63 * np.nanmax(data)
  DWTcoeffs[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in DWTcoeffs[1:])

  data_dwt = pywt.waverec(DWTcoeffs, wavelet_type, mode='per', axis=-1) 
  
  for i in range(5, 25):
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
  split_IMFs = np.array(np.split(IMFs[:-1, :-q], 10, axis=1))

  # check if splitted or mean imfs features
  if args[3]: features = mean_features(mIMFs, nIMFs)
  else: features = split_features(split_IMFs, nIMFs)
  
  np.save(f'{args[1]:s}/{args[0]:s}.npy', features)

  return features