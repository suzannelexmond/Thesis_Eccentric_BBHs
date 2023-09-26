import lalsimulation as lalsim
import lal
import astropy.constants as c
import pycbc
import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform, get_template_amplitude_norm
from pycbc.waveform import td_approximants, fd_approximants
from numba import jit, cuda
from timeit import default_timer as timer
from sklearn import preprocessing
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)



plt.switch_backend('WebAgg')



def SimInspiral(mass1, mass2, eccmin, freqmin, DeltaT = 1./2048., lalDict = lal.CreateDict()):
    hp, hc = lalsim.SimInspiralTD(m1 = lal.MSUN_SI*mass1, m2 = lal.MSUN_SI*mass2, 
                              S1x = 0., S1y = 0., S1z = 0., 
                              S2x = 0., S2y = 0., S2z = 0.,
                              distance = 400.*1e6*lal.PC_SI, inclination = 0.,
                              phiRef = 0., longAscNodes = 0, eccentricity = eccmin, meanPerAno = 0.,
                              deltaT = DeltaT, f_min = freqmin, f_ref = freqmin,
                              LALparams = lalDict, approximant = lalsim.EccentricTD)
    
    hp_TS = pycbc.types.timeseries.TimeSeries(hp.data.data, delta_t=hp.deltaT)
    hc_TS = pycbc.types.timeseries.TimeSeries(hc.data.data, delta_t=hc.deltaT)
    epochTD = hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds/1e9
    timesTD = np.arange(hp.data.length)*hp.deltaT + epochTD
    # np.savetxt('Straindata/SimInspiral_{}M_{}ecc.txt'.format((mass1 + mass2), eccmin), np.column_stack([timesTD,hp_TS,hc_TS]))

    # print("Absolute value of the maximum amplitude for M_tot = {}, eccmin = {}:".format((mass1 + mass2), eccmin),
    #       "\nh_x = ", abs(max(hc_TS)),
    #       "\nh_+ = ", abs(max(hp_TS)))

    return timesTD, hp_TS, hc_TS



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

    return t_over_M, norm_hp_TS_over_M, norm_hc_TS_over_M

# SimInspiral_t_over_M(20, 1, 0.3, 10.)




def plot_Siminspiral_t_over_M(M_total, mass_ratio, eccmin):
    """ Input: M_total: A list of total masses in solar mass, 
        mass_ratio: A list of mass ratio's for 0 <= q <= 1, 
        eccmin: A list of eccentricities for 0 <= e <= 1
    """
    fig, axs = plt.subplots(2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle('Waveform in units of mass')

    # # for index, total_mass in enumerate(M_total):
    # for index, ratio in enumerate(mass_ratio):
    #     start = timer()
    #     t_over_M, hp_TS, hc_TS = SimInspiral_t_over_M(total_mass, ratio, eccmin, freqmin=50/total_mass)
        
    #     axs[index].plot(t_over_M, hp_TS, label='hp')
    #     # axs[index].plot(t_over_M, hc_TS, label = 'hc')
    #     axs[index].set_xlim([-7e3, 1e3])
    #     # axs[0].set_ylim([-1e7, 1e2])
    #     axs[index].set_title("Mass ratio = {}, Total mass = {}, Ecc = {}".format(ratio, total_mass, eccmin))
    #     axs[index].legend(loc = 'upper left')
    #     axs[index].set_xlabel('t [M_total]')
    #     axs[index].set_ylabel('h$_{+}$/M')
    #     # axs[index].invert_xaxis()
    #     print('Strain is calculated')
    #     print("with GPU:", (timer()-start)/60, ' minutes') 
        

    for total_mass in M_total:
        for ratio in mass_ratio:
            for eccentricity in eccmin:

                start = timer()

                t_over_M, hp_TS_over_M, hc_TS_over_M = SimInspiral_t_over_M(total_mass, ratio, eccentricity, freqmin=50/total_mass)

                    

                axs[0].plot(t_over_M, hp_TS_over_M, label = 'M = {} $(M_\odot)$, q = {}, e = {}'.format(total_mass, ratio, eccentricity))
                axs[0].set_xlim([-7e3, 5e2])
                # axs[0].set_xlim(-0.25, 0.2)
                # axs[0].set_ylim([-1e7, 1e2])
                # axs[0].set_title("total mass = {}, mass ratio = {}, ecc = {}".format(total_mass, ratio, eccentricity))
                axs[0].legend(loc = 'upper left')
                axs[0].set_xlabel('t/M')
                axs[0].set_ylabel('Normalized h$_{+}$/M')

                axs[1].plot(t_over_M, hc_TS_over_M, label = 'M = {} $(M_\odot)$, q = {}, e = {}'.format(total_mass, ratio, eccentricity))
                axs[1].set_xlim([-7e3, 1e3])
                # axs[1].set_xlim(-0.25, 0.2)
                # axs[1].set_ylim([-5e-22, 5e-22])
                # axs[1].set_title("total mass = {}, mass ratio = {}, ecc = {}".format(total_mass, ratio, eccentricity))
                axs[1].legend(loc = 'upper left')
                axs[1].set_xlabel('t/M')
                axs[1].set_ylabel('Normalized h$_{x}$/M')
                print("time GPU:", (timer()-start)/60, ' minutes')
                print('Strain is calculated')
            
    figname = 'total mass = {}, mass ratio = {}, ecc = {}.png'.format(M_total, mass_ratio, eccmin)
    # fig.savefig('Images/' + figname)
    print('fig is saved')
    plt.show()

plot_Siminspiral_t_over_M([20], [1], [0.3, 0.6])



def time_difference_peak_t0(total_mass, mass_ratio, eccmin):
    """ Input: total mass, mass ratio and eccentricity of a BBH merger for which mass1 >= mass2.
        Output: two floats (time_difference_hp, time_difference_hc) which describe the time difference between the peak amplitude of the strain and t = 0 for the plus and cross polarizations.
    """
    mass1 = total_mass/((1/mass_ratio) + 1)
    mass2 = total_mass - mass1

    timesTD, hp_TS, hc_TS = SimInspiral(mass1, mass2, eccmin, freqmin=10., DeltaT = 1./2048., lalDict = lal.CreateDict())

    # Time difference between peak and t = 0
    peak_index_hp, peak_index_hc = abs(hp_TS).numpy().argmax(), abs(hc_TS).numpy().argmax()
    peak_time_hp, peak_time_hc = timesTD[peak_index_hp], timesTD[peak_index_hc]

    # Time-axis doesn't range till t = 0. The closest point is -4.88e-4. Difference is added
    return abs(peak_time_hp + timesTD[-1]), abs(peak_time_hc + timesTD[-1]) 