from __future__ import division # for non integer division
import numpy as np
from numpy.fft import fft, ifft
from math import *

def convfft(x,y):
  # convfft(x,y)
  # convolution of numpy vectors x and y with fft.
  
  x_xt = np.zeros(x.shape[0]+y.shape[0]-1)
  x_xt[0:x.shape[0]] = x
  y_xt = np.zeros(x.shape[0]+y.shape[0]-1)
  y_xt[0:y.shape[0]] = y

  return (ifft(fft(x_xt)*fft(y_xt))).real
  
def convPart(h,x):
  # convPart(h,x)
  # partitionned convolution of short vector h with long vector x.
  M = h.shape[0];
  N = ceil(x.shape[0]/M);
  
  x_buff = np.zeros((M,N));
  
  if x.shape[0]%M == 0:
    x_buff[:,:] = x.reshape((M,N),order='F');
  else:
    x[0:floor(x.shape[0]/M)*M].reshape((M,floor(x.shape[0]/M)));
    x_buff[:,0:-1] = x[0:floor(x.shape[0]/M)*M].reshape((M,floor(x.shape[0]/M)),order='F');
    x_buff[0:x.shape[0]-floor(x.shape[0]/M)*M,-1] = x[floor(x.shape[0]/M)*M:];
  
  
  Nfft = 2*M ;
  x_buff_ft = fft(x_buff, n=Nfft, axis=0)
  h_ft = fft(h,n=Nfft)
  
  cv_buff = np.real(ifft(h_ft.reshape((Nfft,1))*x_buff_ft, axis=0))
  
  cv = np.zeros(max(Nfft*ceil(N/2),M+Nfft*floor(N/2)));
 
  cv[0:Nfft*ceil(N/2)] = cv_buff[:,0::2].reshape((Nfft*ceil(N/2)), order='F')
  cv[M:M+Nfft*floor(N/2)] += cv_buff[:,1::2].reshape((Nfft*floor(N/2)), order='F')
  
  return cv[:x.shape[0]+M-1]
