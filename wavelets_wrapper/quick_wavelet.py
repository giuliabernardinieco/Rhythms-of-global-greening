
import numpy as np
import sys
# import matplotlib.pyplot as plt
import pycwt
import pandas as pd

# functions from the processing_wav.py file
from wavelets_wrapper.processing_wav import pad, autoscales, recon, fourier_from_scales, icwt_fixed



# This function performs a full wavelet analysis on the input data file.
def run_full_wavelet_analysis(signal, dt=10000., mirror=True, cut1=None, cut2=None, wf='morlet', dj=0.1, om0=6, normmean=True, mirrormethod=1):

	p = int(om0) #om0 (read as the greek letter and then 0) is the parameter that define the central frequency of the wavelet, 
 					#for morlet this is often set to 6 as this provides a good compromise between time and frequency resolution.
	if wf == 'morlet':
		fullwavelet = pycwt.wavelet.Morlet(p)
	elif wf == 'dog':
		fullwavelet = pycwt.wavelet.DOG(p)
	elif wf == 'paul':
		fullwavelet = pycwt.wavelet.Paul(p)
	else:
		sys.exit('Could not recognise desired mother wavelet, exiting...')

	## read in signal ##
	x =signal #<====== GB addition
	x_orig = x #<====== GB addition
	
 	# initiate some dfs to avoid UnboundLocalError
	xmirror_df = None #<====== GB addition
	signal_df = None #<====== GB addition
	fft_df = None #<====== GB addition
	fft_inv_df = None #<====== GB addition
	scales_df = None #<====== GB addition
	scales_orig_df = None #<====== GB addition
	sumpower_df = None #<====== GB addition
	fft_pycwt_df = None #<====== GB addition
	icwt_df = None #<====== GB addition
	icwt_part1_df = None #<====== GB addition
	icwt_part2_df = None #<====== GB addition
	wavelet_power_df = None #<====== GB addition
 
	# xtemp = np.loadtxt(infile) # read the data, which usually has lat long and then the signal
	# if len(np.shape(xtemp)) > 1: # if more than one column, take the last column as the signal
	# 	x = [ i[-1] for i in xtemp ]
	# else:
	# 	x = xtemp
	# x_orig = x

	## mirror/flip and save mean of result ##
	#  In your notes you can find more text on why mirroring and normalisation are important  
	if mirror == True : # if mirror is true, then the signal is mirrored
		xmax = np.amax(x)

		if mirrormethod == 1:
			xstart = x[0]
			xend = x[-1]
			x_rev = x[::-1]
			x_rev_flip = np.multiply(x_rev,-1.) + xend*2.
			x_flip = np.multiply(x,-1.) + xstart*2
			x = np.concatenate((x_orig,x_rev_flip,x_flip,x_rev,x_orig,x_rev_flip,x_flip,x_rev),axis=0)
		if mirrormethod == 3:
			x_rev = x[::-1]
			x = np.concatenate((x,x_rev,x,x_rev,x,x_rev,x,x_rev),axis=0)
		if mirrormethod == 2:
			x_rev = x[::-1]
			x_flip = (x * -1.)
			x_rev_flip = (x_rev * -1.) 
			x = np.concatenate((x,x_rev,x_flip,x_rev_flip,x,x_rev,x_flip,x_rev_flip),axis=0)

		xmirrormean = np.mean(x)
		xmirror_df = pd.DataFrame({'mirror_mean': [xmirrormean]}) #<====== GB addition

		if normmean == True: # if normmean is true, then the mean is subtracted from the signal.
			# xmname = outdir + '/xmirror.mean'
			# np.savetxt(xmname, np.array([xmirrormean]))
			x = x - xmirrormean
	else:
		if normmean == True:
			x = x - np.mean(x)

	## padding, save signal to be analysed ##
	# actually in this code padding is not applied. padding is usually applied for managing the edge effects, but since we are mirroring the signal that is not necessary
	# the signal is saved in a file called signal.h
	[x_pad, x_pad_orig] = pad(x,method='zeros')
	x_pad = x # no padding!
	xlen = len(x_pad)

	signal_df = pd.DataFrame({'signal': x_pad}) #<====== GB addition

	## calculate fft of signal ##
	fft_signal = np.fft.rfft(x_pad) # calculate the real-valued Fast Fourier Transform (FFT) of the signal
	fkinv = np.fft.irfft(fft_signal) # calculate the inverse FFT to go back to the time domain
	fft_freq = np.fft.rfftfreq(xlen, d=dt) # returns the Discrete Fourier Transform sample frequencies corresponding to the FFT result
							# xlen: Length of the input signal
							# d: Sample spacing (inverse of the sampling rate)
 

	fft_freq = fft_freq[1:int((xlen/2)+1)] # we are trimming the to exclude the zero frequency (the mean) ? - to be checked with G
	fft_signal = (2. / xlen) * abs(fft_signal[1:int(xlen/2)+1]) # not sure what this does ? Are these the amplitudes - to be checked with G
	fft_power  = abs(fft_signal[0:int(xlen/2)+1]) ** 2. # calculate the power of the FFT - to be checked with G

	fftave = np.convolve(fft_power, np.ones(5)/5, mode='same') # This line smooths the power spectrum using a moving average filter. 

	### saving np fft results separate to those done internally by pycwt (by pyfftw.interfaces.scipy_fftpack - see helpers.py from pycwt)
	fft_freq_np = fft_freq
	fft_power_np = fft_power
	fft_ave_np = fftave
	fkinv_np = fkinv
  
	fft_df = pd.DataFrame({'frequency': fft_freq_np, 'power': fft_power_np, 'smoothed_power': fft_ave_np})  #<====== GB addition
	fft_inv_df = pd.DataFrame({'inverse_fft': fkinv_np.real}) #<====== GB addition
		
  	
	## calculate wavelet scales ##
	N=int(x_pad.shape[0]) # N is the length of the signal (padded, but in this case it is not padded)
	N_orig = int(len(x_orig)) # N_orig is the length of the original signal

	s0 = 2.*dt # The smallest scale to be used in the wavelet transform, set to twice the sampling interval (dt)

	j_full = (1/dj) * np.log(N*dt /s0) / np.log(2.) # Number of scales for the padded signal
	j_orig = (1/dj) * np.log(N_orig*dt /s0) / np.log(2.) # Number of scales for the original signal.
	# dj Scale resolution (spacing between scales), set at the beginning of the function

	## perform continuous wavelet transform ##
	cwtX = pycwt.cwt(x_pad, dt, dj=dj, s0=s0, J=j_full, wavelet=fullwavelet) # perform the continuous wavelet transform on padded signal
	cwtX_orig = pycwt.cwt(x_orig, dt, dj=dj, s0=s0, J=j_orig, wavelet=fullwavelet)# perform the continuous wavelet transform on the original signal
	X = cwtX[0] # Wavelet coefficients (complex values representing the signal's features at different scales and positions)
	sj = cwtX[1] # Scales used in the CWT
	sj_orig = cwtX_orig[1] # Scales used in the CWT for the original signal
	coi = cwtX[3] # cone of influence
	fft_power_pycwt = cwtX[4] ** 2. # Power spectrum from the CWT
	fft_freq_pycwt = cwtX[5] # Frequency from the CWT
	fftave_pycwt = np.convolve(fft_power_pycwt, np.ones(5)/5, mode='same') # Smoothed power spectrum from the CWT
	power = (np.abs(X))**2. # Wavelet power spectrum (magnitude squared of the wavelet coefficients)
	sumpow = np.sum(power, axis=1) # Sum of the wavelet power spectrum over all scales, like Gareth said this correspond to averaging on one direction (horizontal)
	rect_power = sumpow/sj
	# rect_power = np.mean(power, axis=1)/xlen # GB addition, corrected from previos calculation that was: rect_power = sumpow/scales*xlen


 
	scales = sj 
	scales_orig = sj_orig

	## convert scales to fourier periods ## - Not sure about this, to be checked with Gareth
	period = fourier_from_scales(scales,wf=wf,p=p)
	scale_len = len(scales)
	scales_df = pd.DataFrame({'index': range(scale_len), 'scale': scales, 'period': period, 'frequency': 1./period}) #<====== GB addition

	period_orig = fourier_from_scales(scales_orig,wf=wf,p=p)
	scale_len_orig = len(scales_orig)
	# fig = plt.figure()
	# ax = fig.add_subplot(2, 1, 1)
	# ax.plot(period, rect_power)
	# ax.set_xscale('log')
	# ax.set_yscale('log')
	# fig.show()

  	# this file contains the scales and periods used in the wavelet transform, it contains 4 columns: n, scale, period, 1/period
	scales_orig_df = pd.DataFrame({'index': range(scale_len_orig), 'scale': scales_orig, 'period': period_orig, 'frequency': 1./period_orig}) #<====== GB addition

  	# RECTIFIED POWER this file contains the sum of the wavelet power spectrum over all scales, it contains 5 columns: n, t, power, period, 1/period
	sumpower_df = pd.DataFrame({'scaled_power': sumpow / xlen, 'period': period, 'frequency': 1./period, 'scale': scales, 'rectified_power': sumpow / (scales * xlen)}) #<====== GB addition
	# sumpower_df = pd.DataFrame({'scaled_power': sumpow / xlen, 'period': period, 'frequency': 1./period, 'scale': scales, 'rectified_power': rect_power}) #<====== GB addition

  	# this file contains the FOURIER power spectrum of the signal, it contains 3 columns: frequency, power, smoothed power
	fft_pycwt_df = pd.DataFrame({'frequency': fft_freq_pycwt, 'power': fft_power_pycwt, 'smoothed_power': fftave_pycwt}) #<====== GB addition


	## Gabor limit check ##
		# The Gabor limit (or Gabor-Heisenberg limit) is a principle in signal processing that states there is a trade-off between
  		# time and frequency resolution. 
		# It essentially means that you cannot simultaneously achieve high resolution in both time and frequency domains.
		# The inverse wavelet transform is used to reconstruct the original signal from its wavelet coefficients. 
		# This step verifies that the correct scales have been used and checks the accuracy of the wavelet transform.
	x_array = dt*np.arange(1, len(x_pad)+1)
	freqs = 1/period
	s1 = np.std(x_array)
	s2 = np.std(freqs)
	gtest = s1*s2
	if gtest < 1/(4*np.pi):
		print('\nWarning, signal spacing/frequency choice may not conform to Gabor limit...\n')

	# perform inverse transform (to check correct scales have been used)...
	## NOTE there was an error in previous versions of pycwt - check wavelet.py line 170 in icwt - sj should be square rooted on bottom of iW = ...
	## AND brackets should be added... should read:
	# iW = (dj * np.sqrt(dt) / (wavelet.cdelta * wavelet.psi(0)) *
	#          (np.real(W) / np.sqrt(sj)).sum(axis=0))
	# and then will work fine
	# INCLUDES calculation of recon factor (from empirical cdelta) unlike mlpy

#	x_icwt = pycwt.icwt(X, sj, dt, dj=dj, wavelet=wf).real
	x_icwt = icwt_fixed(X, sj, dt, dj=dj, wavelet=fullwavelet).real
	icwt_df = pd.DataFrame({'inverse_cwt': x_icwt}) #<====== GB addition
	

	# calculate mean squared error of icwt... not used currently
	diff = np.sqrt((x_pad - x_icwt)**2.)
	diffmean = np.mean(diff)
 

	if cut1 is not None and cut2 is None:
		print('Please specify two cut-off wavelengths to calculate ICWTs...') 
    #Bandpass filter the signal using the inverse CWT
	elif cut1 is not None and cut2 is not None:
		# Ensure cut1 < cut2
		cut_low = min(cut1, cut2)
		cut_high = max(cut1, cut2)

		X_numcols = len(X[0])

		# Get indices and scales within the band window
		scales_window = [s for s in scales if cut_low < s <= cut_high]
		scale_indices_window = [i for i, s in enumerate(scales) if cut_low < s <= cut_high]

		# Slice the wavelet coefficient matrix using selected indices
		X_cut_window = X[scale_indices_window, 0:X_numcols]

		# Inverse CWT using the windowed scale range
		x_icwt_window = icwt_fixed(X_cut_window, np.array(scales_window), dt, dj=dj, wavelet=fullwavelet).real

		# Save as DataFrame
		icwt_part1_df = pd.DataFrame({'inverse_cwt_part1': x_icwt_window.real})

	# elif cut1 is not None and cut2 is not None:
	# 	## use only scales above cut1 or cut2 for reconstruction of signal...
	# 	X_numcols = len(X[0])
	# 	scales_cut1 = []
	# 	scales_cut2 = []
	# 	for i in scales:
	# 		if i > cut1:
	# 			scales_cut1.append(i)
	# 		if i > cut2:
	# 			scales_cut2.append(i)

	# 	ncut1 = len(scales_cut1)
	# 	ncut2 = len(scales_cut2)
	# 	X_cut1 = X[len(scales)-ncut1:len(scales),0:X_numcols]
	# 	X_cut2 = X[len(scales)-ncut2:len(scales),0:X_numcols]

	# 	x_icwt_part1 = icwt_fixed(X_cut1, np.array(scales_cut1), dt, dj=dj, wavelet=fullwavelet).real
	# 	x_icwt_part2 = icwt_fixed(X_cut2, np.array(scales_cut2), dt, dj=dj, wavelet=fullwavelet).real

	# 	# saving as dataframes
	# 	icwt_part1_df = pd.DataFrame({'inverse_cwt_part1': x_icwt_part1.real}) #<====== GB addition
	# 	icwt_part2_df = pd.DataFrame({'inverse_cwt_part2': x_icwt_part2.real}) #<====== GB addition
		

	# 	# calculate mean squared error of both filtered icwts... not used currently
	# 	diff_cut1 = np.sqrt((x_pad - x_icwt_part1.real)**2.)
	# 	diffmean_cut1 = np.mean(diff_cut1)
	# 	diff_cut2 = np.sqrt((x_pad - x_icwt_part2.real)**2.)
	# 	diffmean_cut2 = np.mean(diff_cut2)
	else:
		print('Not calculating filtered ICWTs...')
  

	## output power normalised by scale [e.g. Liu et al., 2007] as list of matrix elements and their values
	numrows = len(power)
	numcols = len(power[0])
	
	
	
	wavelet_power_list = []
	time = np.arange(X.shape[1]) * dt
	# print('X.shape[1]:', X.shape[1])
	for i, scale in enumerate(scales):
		for j, t in enumerate(time):
			wavelet_power_list.append([i, t, power[i, j], period[i], 1./period[i], power[i, j] / scale, X[i, j].real, X[i, j].imag])
	
	wavelet_power_df = pd.DataFrame(wavelet_power_list, columns=['row', 'time', 'power', 'period', 'frequency', 'rectified_power', 'WxR', 'WxI']) #<====== GB addition
	
	# return (cwtX, scales, x_pad, xlen, period, xmirror_df, signal_df, fft_df, fft_inv_df, scales_df, scales_orig_df, sumpower_df, fft_pycwt_df, icwt_df, icwt_part1_df, icwt_part2_df, wavelet_power_df)
	# NOTE: cwtX, scales, x_pad, xlen, period need to be returned to be used in the run_double_wavelet function
	return (cwtX, scales, x_pad, xlen, period, xmirror_df, signal_df, fft_df, fft_inv_df, scales_df, scales_orig_df, sumpower_df, fft_pycwt_df, icwt_df, icwt_part1_df, icwt_part2_df, wavelet_power_df)

def run_double_wavelet_analysis(signal1, signal2, dt=10000., mirror=True, cut1=None, cut2=None, wf='morlet', dj=0.1, om0=6, normmean=True, mirrormethod=1):

	print('\nWarning, currently run_double_wavelet_analysis will overwrite any single wavelet analysis results in the same directory!\n')
	print('Also, not saving cut-off inverse transforms...')
 
	len1 = len(signal1)
	len2 = len(signal2)
	if len1 != len2:
		raise ValueError('The two input signals need to be the same length!')
	if len1 == 0. or len2 == 0.:
		raise ValueError('The signals cannot have zero length.')

	# run wavelet analysis on both signals 
	result1 = run_full_wavelet_analysis(signal1, dt=dt, mirror=mirror, cut1=cut1, cut2=cut2, wf=wf, dj=dj, om0=om0, normmean=normmean, mirrormethod=mirrormethod)
	result2 = run_full_wavelet_analysis(signal2, dt=dt, mirror=mirror, cut1=cut1, cut2=cut2, wf=wf, dj=dj, om0=om0, normmean=normmean, mirrormethod=mirrormethod)
	
	cwtX, scales, x_pad, xlen, period, xmirror_df1, signal_df1, fft_df1, fft_inv_df1, scales_df1, scales_orig_df1, sumpower_df1, fft_pycwt_df1, icwt_df1, icwt_part1_df1, icwt_part2_df1, wavelet_power_df1 = result1 
	cwtY, scales, y_pad, ylen, period, xmirror_df2, signal_df2, fft_df2, fft_inv_df2, scales_df2, scales_orig_df2, sumpower_df2, fft_pycwt_df2, icwt_df2, icwt_part1_df2, icwt_part2_df2, wavelet_power_df2 = result2
	
	X = cwtX[0] #wavelet transform of signal 1
	Y = cwtY[0] #wavelet transform of signal 2
	sjx = cwtX[1] #scales used in the wavelet transform of signal 1
	sjy = cwtY[1] #scales used in the wavelet transform of signal 2
	coix = cwtX[3] # cone of influence for signal 1
	coiy = cwtY[3] # cone of influence for signal 2
	powerx = (np.abs(X))**2.
	powery = (np.abs(Y))**2.
	sumpowx = np.sum(powerx, axis=1)
	sumpowy = np.sum(powery, axis=1)
	rectpowx = []
	distavgrectpowx = []
	rectpowy = []
	distavgrectpowy = []
	for i in range(len(scales)):
		rectpowx.append(powerx[i]/scales[i])
		distavgrectpowx.append(sumpowx[i] / (scales[i] * xlen ))
		rectpowy.append(powery[i]/scales[i])
		distavgrectpowy.append(sumpowy[i] / (scales[i] * ylen ))

	## cross wavelets and coherence
	Wxy = X * np.conjugate(Y) # cross wavelet transform
	xypower = np.abs(Wxy) # cross wavelet power
	Wyy = Y * np.conjugate(Y)
	yypower = np.abs(Wyy)
	phasexy = np.angle(Wxy,deg=True) # cross wavelet phase

	xypower_rectified = xypower / scales[:, np.newaxis] # rectified cross wavelet power
	# print("crosspower min:", xypower.min())
	# print("crosspower max:", xypower.max())
	# print("X:", X.min())
	# print("X:", X.max())
	# print("Y:", Y.min())
	# print("Y:", Y.max())
	# print("|X| min:", np.abs(X).min())
	# print("|X| max:", np.abs(X).max())
	# print("|Y| min:", np.abs(Y).min())
	# print("|Y| max:", np.abs(Y).max())

	
	
	if wf == 'morlet':
		fullwavelet = pycwt.wavelet.Morlet(om0)
	elif wf == 'dog':
		fullwavelet = pycwt.wavelet.DOG(om0)
	elif wf == 'paul':
		fullwavelet = pycwt.wavelet.Paul(om0)
	else:
		sys.exit('Could not recognise desired mother wavelet, exiting...')

	print("\nCalculating coherence with pycwt...\n")
	# j_full = (1/dj) * np.log(int(x_pad.shape[0])*dt /scales[0]) / np.log(2.)
	# R2ns, R_phase, coi, freq, sig = pycwt.wct(x_pad,y_pad,dt,dj=dj,s0=scales[0],J=j_full-1,sig=False, wavelet=fullwavelet) # R2ns is the coherence, R_phase is the phase of the coherence, coi is the cone of influence, freq is the frequency, sig is the significance
	R2ns, R_phase, coi, freq, sig = pycwt.wct(x_pad,y_pad,dt,dj=dj,s0=scales[0],J=len(scales)-1,sig=False, wavelet=fullwavelet) # R2ns is the coherence, R_phase is the phase of the coherence, coi is the cone of influence, freq is the frequency, sig is the significance
	# Save results as dataframes
	xwsumpow = np.sum(xypower, axis=1) # sum of the cross wavelet power over all scales
	sumcoh = np.sum(R2ns, axis=1) # sum of the coherence over all scales

	# used for plot distance-averaged crosspower
	xwsumpow_df = pd.DataFrame({'scaled_power': xwsumpow / xlen, 
                             	'period': period, 
                             	'frequency': 1. / period, 
                             	'scale': scales, 
                             	'rectified_power': xwsumpow / (scales * xlen)})
 
	sumcoh_df = pd.DataFrame({'scaled_coherence': sumcoh / xlen, 
                           		'period': period, 
                           		'frequency': 1. / period, 
                           		'scale': scales, 
                           		'rectified_coherence': sumcoh / (scales * xlen)})
 
	# used to plot the crosspower (xypower), time, vs period
	xypower_df = pd.DataFrame({'row': np.repeat(np.arange(len(powery)), len(powery[0])), 
                            	'time': np.tile(np.arange(len(powery[0])) * dt, len(powery)), 
                            	'xypower': xypower.flatten(), 
								'xypower_rectified': xypower_rectified.flatten(),
                             	'period': np.repeat(period, len(powery[0])), 
                            	'frequency': np.repeat(1. / period, len(powery[0]))}) # File that you need for crosspower
 
	phasexy_df = pd.DataFrame({'row': np.repeat(np.arange(len(powery)), len(powery[0])), 
                            	'time': np.tile(np.arange(len(powery[0])) * dt, len(powery)), 
                            	'phasexy': phasexy.flatten(),
                            	'period': np.repeat(period, len(powery[0])), 
                            	'frequency': np.repeat(1. / period, len(powery[0]))}) # File that you need for phase
 
	R2ns_df = pd.DataFrame({'row': np.repeat(np.arange(len(powery)), len(powery[0])), 
                         		'time': np.tile(np.arange(len(powery[0])) * dt, len(powery)), 
                         		'R2ns': R2ns.flatten(), 
                           		'period': np.repeat(period, len(powery[0])), 
                         		'frequency': np.repeat(1. / period, len(powery[0]))}) # File that you need for coherence

	return (xmirror_df1, signal_df1, fft_df1, fft_inv_df1, scales_df1, scales_orig_df1, sumpower_df1, fft_pycwt_df1, icwt_df1, icwt_part1_df1, icwt_part2_df1, wavelet_power_df1,
            xmirror_df2, signal_df2, fft_df2, fft_inv_df2, scales_df2, scales_orig_df2, sumpower_df2, fft_pycwt_df2, icwt_df2, icwt_part1_df2, icwt_part2_df2, wavelet_power_df2,
            xwsumpow_df, sumcoh_df, xypower_df, phasexy_df, R2ns_df, coix, coiy, coi, cwtX, cwtY, scales)
