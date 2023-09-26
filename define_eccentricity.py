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

@jit(target_backend='cuda')  # Explicitly set nopython=True
def SimInspiral_t_over_M(total_mass, mass_ratio, eccmin, freqmin, DeltaT = 1./2048., lalDict = lal.CreateDict()):
    """ Input: total_mass in M_sun, mass_ratio >= 1 (mass1 > mass2)
        Output: 
    
    """

    mass1 = total_mass/((1/mass_ratio) + 1)
    mass2 = total_mass - mass1

    timesTD, hp_TS, hc_TS = SimInspiral(mass1, mass2, eccmin, freqmin)
    # time_before_merger = abs(timesTD + timesTD[-1])

    t_over_M = timesTD / (lal.MTSUN_SI * total_mass )

    hp_TS_over_M = hp_TS/total_mass
    hc_TS_over_M = hc_TS/total_mass

    norm_hp_TS_over_M = (hp_TS_over_M - hp_TS_over_M.min())/ (hp_TS_over_M.max() - hp_TS_over_M.min())
    norm_hc_TS_over_M = (hc_TS_over_M - hc_TS_over_M.min())/ (hc_TS_over_M.max() - hc_TS_over_M.min())

    return t_over_M, hp_TS, hc_TS

def get_peaks_t_over_M(values, total_mass):

    valuesmin = -values
    pericenters_index, _per = find_peaks(values, height=0)
    apocenters_index, _apo = find_peaks(valuesmin, height=-100)

    pericenters = _per['peak_heights']
    apocenters = -_apo['peak_heights']

    return pericenters_index, pericenters, apocenters_index, apocenters


def estimate_eccentricity(freq_per, freq_apo):
    w_per, w_apo = freq_per*(2*np.pi), freq_apo*(2*np.pi)

    w_ecc = np.zeros(len(freq_per))
    for i in range(len(freq_per)):
        # Freq epocenters has one more value than the freq pericenters. We start counting from apocenters[0] + i
        w_ecc[i] = (np.sqrt(w_per[i]**2) - np.sqrt(w_apo[i + 1]**2)) / ((np.sqrt(w_per[i]**2) + np.sqrt(w_apo[i + 1]**2)))
    
    return w_ecc


def plot_frequency(total_mass, mass_ratio, eccmin, freqmin, DeltaT = 1./2048., lalDict = lal.CreateDict()):
    # AMPLITUDE AND FREQUENCY METHOD

    hp_TS, hc_TS = SimInspiral(total_mass, mass_ratio, eccmin, freqmin)


    amp = waveform.utils.amplitude_from_polarizations(hp_TS, hc_TS)
    freq = waveform.utils.frequency_from_polarizations(hp_TS, hc_TS)

    fig1, axs = plt.subplots(3, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    hp_times = -hp_TS.sample_times[::-1] / (lal.MTSUN_SI * total_mass )
    amp_times = -amp.sample_times[::-1] / (lal.MTSUN_SI * total_mass )
    freq_times = -freq.sample_times[::-1] / (lal.MTSUN_SI * total_mass )

    axs[0].plot(hp_times, hp_TS)
    axs[0].set_ylabel('h$_{+}$')
    # plt.xlim(-7, 0)
    axs[0].set_xlabel('t [M]')
    axs[0].set_title('Waveform')

    
    amp_pericenters_index, amp_pericenters, amp_apocenters_index, amp_apocenters = get_peaks_t_over_M(amp, total_mass)

    axs[1].plot(amp_times, amp, color='orange')
    axs[1].scatter(amp_times[amp_pericenters_index], amp_pericenters, color='blue', s=5 , label='pericenters')
    axs[1].scatter(amp_times[amp_apocenters_index], amp_apocenters, color='magenta', s=5 , label='apocenters')
    axs[1].set_ylabel('A$_+$')
    axs[1].set_xlim(-20000, 0)
    # axs[1].set_ylim(0, 50)
    axs[1].set_xlabel('t [M]')
    axs[1].legend(loc='upper left')
    axs[1].set_title('Amplitude Method')


    freq_pericenters_index, freq_pericenters, freq_apocenters_index, freq_apocenters = get_peaks_t_over_M(freq, total_mass)

    axs[2].plot(freq_times, freq, color='orange')
    axs[2].scatter(freq_times[freq_pericenters_index], freq_pericenters, color='blue', s=5 , label='pericenters')
    axs[2].scatter(freq_times[freq_apocenters_index], freq_apocenters, color='magenta', s=5 , label='apocenters')
    axs[2].set_ylabel('f$_+$ = $\omega_+$/2$\pi$ [Hz]')
    axs[2].set_xlim(-20000, 0)
    axs[2].set_ylim(0, 50)
    axs[2].set_xlabel('t [M]')
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

    
    # Residual frequency 
    res_freq_times = np.zeros(len(freq_times))
    res_freq = np.zeros(len(freq))

    for i in range(len(freq_times)):
        res_freq_times[i] = freq_times[i]
        res_freq[i] = freq[i] - freq_circ[i]

    res_freq_pericenters_index, res_freq_pericenters, res_freq_apocenters_index, res_freq_apocenters = get_peaks_t_over_M(res_freq, total_mass)


    # Residual Amplitude
    res_amp_times = np.zeros(len(amp_times))
    res_amp = np.zeros(len(amp))

    for i in range(len(amp_times)):
        res_amp_times[i] = amp_times[i]
        res_amp[i] = amp[i] - amp_circ[i]

    res_amp_pericenters_index, res_amp_pericenters, res_amp_apocenters_index, res_amp_apocenters = get_peaks_t_over_M(res_amp, total_mass)

   

    fig2, axs = plt.subplots(3, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    axs[0].plot(hp_times, hp_TS)
    axs[0].set_ylabel('h$_{+}$')
    # axs.xlim(-7, 0)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_title('Waveform; total mass={}, mass ratio={}, eccmin={}, freqmin={}'.format(total_mass, mass_ratio, eccmin, freqmin))

    # axs[1].plot(amp_times, amp, label='A$_+$', color='orange' )
    # axs[1].plot(amp_times, amp_circ, label='A$_+^{circ}$', color='red')
    axs[1].plot(res_amp_times, res_amp, label='Residual Amplitude', color='orange')
    axs[1].scatter(res_amp_times[res_amp_pericenters_index], res_amp_pericenters, color='blue', s=5 , label='pericenters')
    axs[1].scatter(res_amp_times[res_amp_apocenters_index], res_amp_apocenters, color='magenta', s=5 , label='apocenters')
    axs[1].set_ylabel('$\Delta$A$_+$')
    # axs[1].set_xlim(-20000, 0)
    # axs[1].set_ylim(0, 50)
    axs[1].set_xlabel('t [M]')
    axs[1].legend(loc='upper left')
    axs[1].set_title('Residual amplitude Method')


    # axs[2].plot(freq_times, freq, label='$\omega_+$', color='orange')
    # axs[2].plot(freq_times, freq_circ, label='$\omega_+^{circ}$', color='red')
    axs[2].plot(res_freq_times, res_freq, label = 'Residual frequency', color='orange')
    axs[2].scatter(res_freq_times[res_freq_pericenters_index], res_freq_pericenters, color='blue', s=5 , label='pericenters')
    axs[2].scatter(res_freq_times[res_freq_apocenters_index], res_freq_apocenters, color='magenta', s=5 , label='apocenters')
    axs[2].set_ylabel('$\Delta$f$_+$ = $\Delta\omega_+$/2$\pi$ [Hz]')
    # axs[2].set_xlim(-20000, 0)
    axs[2].set_ylim(-50, 50)
    axs[2].set_xlabel('t [M]')
    axs[2].legend(loc='upper left')
    axs[2].set_title('Residual frequency Method')

    # plt.show()
    # figname = 'Residual_Amp_Freq_over_M_zoom'
    # fig2.savefig('Images/' + figname)

    ####################################################
    # Eccentricity estimate
    # print(len(res_amp_apocenters), len(res_amp_pericenters))
    # print(res_amp_apocenters, res_amp_pericenters)
    # print(len(res_freq_pericenters), len(res_freq_apocenters))
    
    res_ecc_w = estimate_eccentricity(res_freq_pericenters, res_freq_apocenters)

    fig3, axs = plt.subplots(1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    axs.plot(res_freq_times[res_freq_pericenters_index], res_ecc_w)
    axs.set_ylabel('ecc$_{w}$')
    axs.set_xlabel('t [M]')

    plt.show()



plot_frequency(50, 4, 0.4, 5)
# print('before')
# SimInspiral(50, 4, 0.3, 5)
# SimInspiral(50, 4, 1e-10, 5)
# print('after')



