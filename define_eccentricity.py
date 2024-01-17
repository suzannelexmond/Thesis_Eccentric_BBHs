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
from scipy.optimize import curve_fit

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

def get_peaks_t_over_M(values):

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


def Freq_Amp_Method(total_mass, mass_ratio, eccmin, freqmin, DeltaT = 1./2048., lalDict = lal.CreateDict()):
    # AMPLITUDE AND FREQUENCY METHOD

    hp_TS, hc_TS = SimInspiral(total_mass, mass_ratio, eccmin, freqmin)

    amp = waveform.utils.amplitude_from_polarizations(hp_TS, hc_TS)
    freq = waveform.utils.frequency_from_polarizations(hp_TS, hc_TS)


    hp_times = -hp_TS.sample_times[::-1] / (lal.MTSUN_SI * total_mass )
    amp_times = -amp.sample_times[::-1] / (lal.MTSUN_SI * total_mass )
    freq_times = -freq.sample_times[::-1] / (lal.MTSUN_SI * total_mass )

    amp_pericenters_index, amp_pericenters, amp_apocenters_index, amp_apocenters = get_peaks_t_over_M(amp)
    freq_pericenters_index, freq_pericenters, freq_apocenters_index, freq_apocenters = get_peaks_t_over_M(freq)

    fig1, axs = plt.subplots(3, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    axs[0].plot(hp_times, hp_TS)
    axs[0].set_ylabel('h$_{+}$')
    # plt.xlim(-7, 0)
    axs[0].set_xlabel('t [M]')
    axs[0].set_title('Waveform; total mass={}, mass ratio={}, eccmin={}, freqmin={}'.format(total_mass, mass_ratio, eccmin, freqmin))


    axs[1].plot(amp_times, amp, color='orange')
    axs[1].scatter(amp_times[amp_pericenters_index], amp_pericenters, color='blue', s=5 , label='pericenters')
    axs[1].scatter(amp_times[amp_apocenters_index], amp_apocenters, color='magenta', s=5 , label='apocenters')
    axs[1].set_ylabel('A$_+$')
    axs[1].set_xlim(-20000, 0)
    # axs[1].set_ylim(0, 50)
    axs[1].set_xlabel('t [M]')
    axs[1].legend(loc='upper left')
    axs[1].set_title('Amplitude Method')


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
    # figname = 'Apocenters_Epicenters_FrequencyMethod_zoom.png'
    # fig1.savefig('Images/' + figname)

    return freq_times[freq_apocenters_index], freq_apocenters, freq_times[freq_pericenters_index], freq_pericenters, amp_times[amp_apocenters_index], amp_apocenters, amp_times[amp_pericenters_index], amp_pericenters, freq, amp, freq_times, amp_times



def Residual_Freq_Amp_Method(total_mass, mass_ratio, eccmin, freqmin):
    #RESIDUAL AMPLITUDE AND FREQUENCY METHOD

    hp_TS, hc_TS = SimInspiral(total_mass, mass_ratio, eccmin, freqmin)

    amp = waveform.utils.amplitude_from_polarizations(hp_TS, hc_TS)
    freq = waveform.utils.frequency_from_polarizations(hp_TS, hc_TS)

    hp_times = -hp_TS.sample_times[::-1] / (lal.MTSUN_SI * total_mass )
    amp_times = -amp.sample_times[::-1] / (lal.MTSUN_SI * total_mass )
    freq_times = -freq.sample_times[::-1] / (lal.MTSUN_SI * total_mass )

    hp_TS_circ, hc_TS_circ = SimInspiral(total_mass, mass_ratio, 1e-10, 5)


    amp_circ = waveform.utils.amplitude_from_polarizations(hp_TS_circ, hc_TS_circ)
    freq_circ = waveform.utils.frequency_from_polarizations(hp_TS_circ, hc_TS_circ)

    amp_circ = amp_circ[len(amp_circ) - len(amp):]
    freq_circ = freq_circ[len(freq_circ) - len(freq):]

    
    # Residual frequency 
    res_freq = np.zeros(len(freq))

    for i in range(len(freq_times)):
        res_freq[i] = freq[i] - freq_circ[i]

    res_freq_pericenters_index, res_freq_pericenters, res_freq_apocenters_index, res_freq_apocenters = get_peaks_t_over_M(res_freq)


    # Residual Amplitude
    res_amp = np.zeros(len(amp))

    for i in range(len(amp_times)):
        res_amp[i] = amp[i] - amp_circ[i]

    res_amp_pericenters_index, res_amp_pericenters, res_amp_apocenters_index, res_amp_apocenters = get_peaks_t_over_M(res_amp)

   

    fig2, axs = plt.subplots(3, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    axs[0].plot(hp_times, hp_TS)
    axs[0].set_ylabel('h$_{+}$')
    # axs.xlim(-7, 0)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_title('Waveform; total mass={}, mass ratio={}, eccmin={}, freqmin={}'.format(total_mass, mass_ratio, eccmin, freqmin))

    # axs[1].plot(amp_times, amp, label='A$_+$', color='orange' )
    # axs[1].plot(amp_times, amp_circ, label='A$_+^{circ}$', color='red')
    axs[1].plot(amp_times, res_amp, label='Residual Amplitude', color='orange')
    axs[1].scatter(amp_times[res_amp_pericenters_index], res_amp_pericenters, color='blue', s=5 , label='pericenters')
    axs[1].scatter(amp_times[res_amp_apocenters_index], res_amp_apocenters, color='magenta', s=5 , label='apocenters')
    axs[1].set_ylabel('$\Delta$A$_+$')
    axs[1].set_xlim(-20000, 0)
    # axs[1].set_ylim(0, 50)
    axs[1].set_xlabel('t [M]')
    axs[1].legend(loc='upper left')
    axs[1].set_title('Residual amplitude Method')


    # axs[2].plot(freq_times, freq, label='$\omega_+$', color='orange')
    # axs[2].plot(freq_times, freq_circ, label='$\omega_+^{circ}$', color='red')
    axs[2].plot(freq_times, res_freq, label = 'Residual frequency', color='orange')
    axs[2].scatter(freq_times[res_freq_pericenters_index], res_freq_pericenters, color='blue', s=5 , label='pericenters')
    axs[2].scatter(freq_times[res_freq_apocenters_index], res_freq_apocenters, color='magenta', s=5 , label='apocenters')
    axs[2].set_ylabel('$\Delta$f$_+$ = $\Delta\omega_+$/2$\pi$ [Hz]')
    axs[2].set_xlim(-20000, 0)
    axs[2].set_ylim(-50, 50)
    axs[2].set_xlabel('t [M]')
    axs[2].legend(loc='upper left')
    axs[2].set_title('Residual frequency Method')

    plt.show()
    # figname = 'Residual_Amp_Freq_over_M_zoom.png'
    # fig2.savefig('Images/' + figname)

    return freq_times[res_freq_apocenters_index], res_freq_apocenters, freq_times[res_freq_pericenters_index], res_freq_pericenters, amp_times[res_amp_apocenters_index], res_amp_apocenters, amp_times[res_amp_pericenters_index], res_amp_pericenters, res_freq, res_amp



def FrequencyFits_Method(total_mass, mass_ratio, eccmin, freqmin):
    # FREQUENCY FITS AND AMPLITUDE FITS METHOD

    # Pericenters and Apocenters of frequency and amplitude
    apos_time_freq, apos_freq, peris_time_freq, peris_freq, apos_time_amp, apos_amp, peris_time_amp, peris_amp, freq, amp, freq_times, amp_times = Freq_Amp_Method(total_mass, mass_ratio, eccmin, freqmin)

   
    def fit_func(t, f0, f1, t_merg):
        # Fit guess for frequency (reduced correlation)
        t_mid=t[(int(len(t)/2))]
        n = -f1*(t_merg - t_mid)/f0
        A = f0*(t_merg - t_mid)**(-n)
        return A*(t_merg - t)**n

    def fit_function(t, A, n, t_merg):
        # Fit guess for frequency
        return A*(t_merg - t)**n 

    # Fit the function for frequency apocenters and epicenters
    params_w_a, covariance_w_a = curve_fit(fit_function, apos_time_freq[3:250], apos_freq[3:250], [350, -0.5, 0], bounds = ([300, -0.5, -0.9], [500, 0, 1]))
    params_w_p, covariance_w_p = curve_fit(fit_function, peris_time_freq[3:250], peris_freq[3:250], [200, -0.5, 0], bounds = ([100, -0.5, -0.9], [400, 0, 1]))
    print("Fitted parameters w_apos; A:{}, t_merg:{}, n:{}".format(*params_w_a),
          "\nFitted parameters w_peris; A:{}, t_merg:{}, n:{}".format(*params_w_p))
    
    
    # Fit the function for amplitude apocenters and epicenters
    params_A_a, covariance_A_a = curve_fit(fit_function, apos_time_amp[0:250], apos_amp[0:250], [1e-21, -0.3, 0], bounds = ([8.00e-22, -0.5, 0], [5.00e-21, 0, 1]))
    params_A_p, covariance_A_p = curve_fit(fit_function, peris_time_amp[0:250], peris_amp[0:250], [5e-22, -0.1, 0], bounds = ([3.00e-22, -0.5, 0], [1.00e-21, 0, 1]))
    print("Fitted parameters A_apos; A:{}, t_merg:{}, n:{}".format(*params_A_a),
          "\nFitted parameters A_peris; A:{}, t_merg:{}, n:{}".format(*params_A_p))    



    def U_fit(t, y, params_fit_a, params_fit_p):
        # Determine U_(t) for substraction of fit function
        
        fit_p = fit_function(t, *params_fit_p)
        U_p = np.zeros(len(y))

        fit_a = fit_function(t, *params_fit_a)
        U_a = np.zeros(len(y))

        for i in range(len(y)):
            U_p[i] = y[i] - fit_p[i]
            U_a[i] = -(y[i] - fit_a[i])

        return U_a, U_p

    # U_p(t) = w_22 - w_22^fit_p 
    U_w_a, U_w_p = U_fit(freq_times, freq, params_w_a, params_w_p)
    # U_a(t) = w_22 - w_22^fit_a
    U_A_a, U_A_p = U_fit(amp_times, amp, params_A_a, params_A_p)



    fig3, axs = plt.subplots(4, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    # Fit for frequency
    axs[0].plot(amp_times, fit_function(amp_times, *params_A_a), 'r--', label='fit peri: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(params_A_a))
    axs[0].plot(amp_times, fit_function(amp_times, *params_A_p), 'b--', label='fit apo: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(params_A_p))
    axs[0].scatter(peris_time_amp[0:250], peris_amp[0:250], color='blue', s=5 , label='pericenters')
    axs[0].scatter(apos_time_amp[0:250], apos_amp[0:250], color='magenta', s=5 , label='apocenters')
    axs[0].set_ylabel('A$_+$')
    axs[0].set_xlim(-200000, 0)
    axs[0].set_ylim([0, 6e-22])
    axs[0].set_xlabel('t [M]')
    axs[0].legend(loc='upper left')
    axs[0].set_ylabel('A$_+$')
    axs[0].set_title('AmplitudeFits Method')

    # Fit for amplitude
    axs[1].plot(freq_times, freq, color='orange')
    axs[1].plot(freq_times, fit_function(freq_times, *params_w_p), 'b--', label='fit peri: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(params_w_p))
    axs[1].plot(freq_times, fit_function(freq_times, *params_w_a), 'r--', label='fit apo: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(params_w_a))
    axs[1].scatter(peris_time_freq[3:250], peris_freq[3:250], color='blue', s=5 , label='pericenters')
    axs[1].scatter(apos_time_freq[3:250], apos_freq[3:250], color='magenta', s=5 , label='apocenters')
    axs[1].set_xlim(-100000, 0)
    axs[1].set_ylim([0, 40])
    axs[1].set_xlabel('t [M]')
    axs[1].set_ylabel('f$_+$ = $\omega_+$/2$\pi$ [Hz]')
    axs[1].set_title('FrequencyFits Method')
    axs[1].legend(loc='upper left')

    # AmplitudeFits Method U(t)
    axs[2].plot(amp_times, U_A_p, label='U(t) = w$_{22}$ - w$_{22}^{fit_p}$', color='blue')
    axs[2].plot(amp_times, U_A_a, label='U(t) = -(w$_{22}$ - w$_{22}^{fit_a}$)', color='magenta')
    # axs[2].scatter(fit_amp_times_peris[pericenters_index_amp], peris_amp_diff, color='blue', s=5 , label='pericenters')
    # axs[2].scatter(fit_amp_times_apos[apocenters_index_amp], apos_amp_diff, color='magenta', s=5 , label='apocenters')
    axs[2].set_ylabel('$\Delta$A$_+$')
    axs[2].set_xlim(-200000, -120000)
    axs[2].set_ylim(-2e-22, 2e-22)
    axs[2].set_xlabel('t [M]')
    axs[2].legend(loc='upper left')
    axs[2].set_title('AmplitudeFits Method')

    # FrequencyFits Method U(t)
    axs[3].plot(freq_times, U_w_p, label = 'U(t) = A$_{22}$ - A$_{22}^{fit_p}$', color='blue')
    axs[3].plot(freq_times, U_w_a, label = 'U(t) = -(A$_{22}$ - A$_{22}^{fit_a}$)', color='magenta')
    # axs[3].scatter(fit_freq_times[fit_freq_pericenters_index], fit_freq_pericenters, color='blue', s=5 , label='pericenters')
    # axs[3].scatter(fit_freq_times[fit_freq_apocenters_index], fit_freq_apocenters, color='magenta', s=5 , label='apocenters')
    axs[3].set_ylabel('$\Delta$f$_+$ = $\Delta\omega_+$/2$\pi$ [Hz]')
    axs[3].set_xlim(-200000, -120000)
    axs[3].set_ylim(-7, 4)
    axs[3].set_xlabel('t [M]')
    axs[3].legend(loc='upper left')
    axs[3].set_title('FrequencyFits Method')

    
    figname = 'Freq_Amp_Fits_Method_e={}.png'.format(eccmin)
    fig3.savefig('Images/' + figname)
    plt.show()





# Freq_Amp_Method(50, 4, 0.4, 5)
# Residual_Freq_Amp_Method(50, 4, 0.4, 5)
FrequencyFits_Method(50, 4, 0.3, 5)



















# ####################################################
# # Eccentricity estimate
# # print(len(res_amp_apocenters), len(res_amp_pericenters))
# # print(res_amp_apocenters, res_amp_pericenters)
# # print(len(res_freq_pericenters), len(res_freq_apocenters))

# res_ecc_w = estimate_eccentricity(res_freq_pericenters, res_freq_apocenters)

# fig3, axs = plt.subplots(1, figsize=(10, 10))
# plt.subplots_adjust(hspace=0.5)

# axs.plot(res_freq_times[res_freq_pericenters_index], res_ecc_w)
# axs.set_ylabel('ecc$_{w}$')
# axs.set_xlabel('t [M]')

# plt.show()








