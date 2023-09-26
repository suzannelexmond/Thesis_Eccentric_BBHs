# from generating_waveforms import SimInspiral
import lalsimulation as lalsim
import lal
import pycbc
from pycbc import waveform
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, cuda
from timeit import default_timer as timer
from sklearn import preprocessing
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning
import warnings
from scipy.signal import find_peaks

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

plt.switch_backend('WebAgg')

def SimInspiral(total_mass, mass_ratio, eccmin, freqmin, DeltaT = 1./2048., lalDict = lal.CreateDict()):
    mass1 = total_mass/((1/mass_ratio) + 1)
    mass2 = total_mass - mass1
    
    hp, hc = lalsim.SimInspiralTD(m1 = lal.MSUN_SI*mass1, m2 = lal.MSUN_SI*mass2, 
                              S1x = 0., S1y = 0., S1z = 0., 
                              S2x = 0., S2y = 0., S2z = 0.,
                              distance = 400.*1e6*lal.PC_SI, inclination = 0.,
                              phiRef = 0., longAscNodes = 0, eccentricity = eccmin, meanPerAno = 0.,
                              deltaT = DeltaT, f_min = freqmin, f_ref = freqmin,
                              LALparams = lalDict, approximant = lalsim.EccentricTD)

    hp_TS = pycbc.types.timeseries.TimeSeries(hp.data.data, delta_t=hp.deltaT)
    hc_TS = pycbc.types.timeseries.TimeSeries(hc.data.data, delta_t=hc.deltaT)

    return hp_TS, hc_TS



def plot_frequency(total_mass, mass_ratio, eccmin, freqmin, DeltaT = 1./2048., lalDict = lal.CreateDict()):
    # AMPLITUDE AND FREQUENCY METHOD

    hp_TS, hc_TS = SimInspiral(total_mass, mass_ratio, eccmin, freqmin)


    amp = waveform.utils.amplitude_from_polarizations(hp_TS, hc_TS)
    freq = waveform.utils.frequency_from_polarizations(hp_TS, hc_TS)

    fig1, axs = plt.subplots(3, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    axs[0].plot(-hp_TS.sample_times[::-1], hp_TS)
    axs[0].set_ylabel('h$_{+}$')
    # plt.xlim(-7, 0)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_title('Waveform')




    amp_times = -amp.sample_times[::-1]

    ampmin = -amp
    pericenters_index, _per = find_peaks(amp, height=0)
    apocenters_index, _apo = find_peaks(ampmin, height=-100)

    pericenters_amp = _per['peak_heights']
    apocenters_amp = -_apo['peak_heights']

    axs[1].plot(amp_times, amp, color='orange')
    axs[1].scatter(amp_times[pericenters_index], pericenters_amp, color='blue', s=5 , label='pericenters')
    axs[1].scatter(amp_times[apocenters_index], apocenters_amp, color='magenta', s=5 , label='apocenters')
    axs[1].set_ylabel('A$_+$')
    axs[1].set_xlim(-7, 0)
    # axs[1].set_ylim(0, 50)
    axs[1].set_xlabel('t [s]')
    axs[1].legend(loc='upper left')
    axs[1].set_title('Amplitude Method')
    # axs[0].invert_xaxis()


    freq_times = -freq.sample_times[::-1]

    freqmin = -freq
    pericenters_index, _per = find_peaks(freq, height=0)
    apocenters_index, _apo = find_peaks(freqmin, height=-100)

    pericenters_freq = _per['peak_heights']
    apocenters_freq = -_apo['peak_heights']


    axs[2].plot(freq_times, freq, color='orange')
    axs[2].scatter(freq_times[pericenters_index], pericenters_freq, color='blue', s=5 , label='pericenters')
    axs[2].scatter(freq_times[apocenters_index], apocenters_freq, color='magenta', s=5 , label='apocenters')
    axs[2].set_ylabel('f$_+$ = $\omega_+$/2$\pi$ [Hz]')
    axs[2].set_xlim(-7, 0)
    axs[2].set_ylim(0, 50)
    axs[2].set_xlabel('t [s]')
    axs[2].legend(loc='upper left')
    axs[2].set_title('Frequency Method')

    # plt.show()
    # figname = 'Apocenters_Epicenters_FrequencyMethod_zoom'
    # fig1.savefig('Images/' + figname)

    ########################################################################################################
    #RESIDUAL AMPLITUDE AND FREQUENCY METHOD

    hp_TS_circ, hc_TS_circ = SimInspiral(total_mass, mass_ratio, 1e-10, 5)

    amp_circ = waveform.utils.amplitude_from_polarizations(hp_TS_circ, hc_TS_circ)
    freq_circ = waveform.utils.frequency_from_polarizations(hp_TS_circ, hc_TS_circ)

    amp_circ = amp_circ[len(amp_circ) - len(amp):]
    freq_circ = freq_circ[len(freq_circ) - len(freq):]

    freq_times = -freq.sample_times[::-1]
    amp_times = -amp.sample_times[::-1]
    
    # Residual frequency 
    Res_f_t = np.zeros(len(freq_times))
    Res_f = np.zeros(len(freq))

    for i in range(len(freq_times)):
        Res_f_t[i] = freq_times[i]
        Res_f[i] = freq[i] - freq_circ[i]

    Res_f_min = -Res_f
    pericenters_index_f, _per = find_peaks(Res_f, height=0)
    apocenters_index_f, _apo = find_peaks(Res_f_min, height=-100)

    pericenters_res_f = _per['peak_heights']
    apocenters_res_f = -_apo['peak_heights']

    # Residual Amplitude
    Res_A_t = np.zeros(len(amp_times))
    Res_A = np.zeros(len(amp))

    for i in range(len(amp_times)):
        Res_A_t[i] = amp_times[i]
        Res_A[i] = amp[i] - amp_circ[i]

    Res_A_min = -Res_A
    pericenters_index_amp, _per = find_peaks(Res_A, height=0)
    apocenters_index_amp, _apo = find_peaks(Res_A_min, height=-100)

    pericenters_res_amp = _per['peak_heights']
    apocenters_res_amp = -_apo['peak_heights']



    fig2, axs = plt.subplots(3, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    axs[0].plot(-hp_TS.sample_times[::-1], hp_TS)
    axs[0].set_ylabel('h$_{+}$')
    # axs.xlim(-7, 0)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_title('Waveform')

    # axs[1].plot(amp_times, amp, label='A$_+$', color='orange' )
    # axs[1].plot(amp_times, amp_circ, label='A$_+^{circ}$', color='red')
    axs[1].plot(Res_A_t, Res_A, label='Residual Amplitude')
    axs[1].scatter(amp_times[pericenters_index_amp], pericenters_res_amp, color='blue', s=5 , label='pericenters')
    axs[1].scatter(amp_times[apocenters_index_amp], apocenters_res_amp, color='magenta', s=5 , label='apocenters')
    axs[1].set_ylabel('A$_+$')
    # axs[1].set_xlim(-7, 0)
    # axs[1].set_ylim(0, 50)
    axs[1].set_xlabel('t [s]')
    axs[1].legend(loc='upper left')
    axs[1].set_title('Residual amplitude Method')


    # axs[2].plot(freq_times, freq, label='$\omega_+$', color='orange')
    # axs[2].plot(freq_times, freq_circ, label='$\omega_+^{circ}$', color='red')
    axs[2].plot(Res_f_t, Res_f, label = 'Residual frequency')
    axs[2].scatter(freq_times[pericenters_index_f], pericenters_res_f, color='blue', s=5 , label='pericenters')
    axs[2].scatter(freq_times[apocenters_index_f], apocenters_res_f, color='magenta', s=5 , label='apocenters')
    axs[2].set_ylabel('f$_+$ = $\omega_+$/2$\pi$ [Hz]')
    # axs[2].set_xlim(-7, 0)
    axs[2].set_ylim(-50, 50)
    axs[2].set_xlabel('t [s]')
    axs[2].legend(loc='upper left')
    axs[2].set_title('Residual frequency Method')

    plt.show()
    figname = 'Residual_'
    fig2.savefig('Images/' + figname)

plot_frequency(50, 4, 0.3, 5)
# print('before')
# SimInspiral(50, 4, 0.3, 5)
# SimInspiral(50, 4, 1e-10, 5)
# print('after')