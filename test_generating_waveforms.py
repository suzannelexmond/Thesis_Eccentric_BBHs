import lalsimulation as lalsim
import lal
import astropy.constants as c
import pycbc
import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform
from pycbc.waveform import td_approximants, fd_approximants
from numba import jit, cuda


plt.switch_backend('WebAgg')

DeltaT = 1./2048.

mass1 = 50.
mass2 = 50.
eccmin = 0.4
freqmin = 10.
lalDict = lal.CreateDict()

####################################################
# Generate eccentric waveform with SimInspiralTD

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
np.savetxt('siminspiralTD.txt', np.column_stack([timesTD,hp_TS,hc_TS]))

fig1, (ax1, ax2) = plt.subplots(2)
ax1.plot(timesTD, hp_TS, label='hp')
ax1.plot(timesTD, hc_TS, label = 'hc')
ax1.set_title("SimInspiralTD()")
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Strain')
ax1.legend()

#######################################################3
# Generate eccentric waveform with ChooseTDWaveform

hp, hc = lalsim.SimInspiralChooseTDWaveform(m1 = lal.MSUN_SI*mass1, m2 = lal.MSUN_SI*mass2,
                              s1x = 0., s1y = 0., s1z = 0., 
                              s2x = 0., s2y = 0., s2z = 0.,
                              distance = 400.*1e6*lal.PC_SI, inclination = 0.,
                              phiRef = 0., longAscNodes = 0, eccentricity = eccmin, meanPerAno = 0.,
                              deltaT = DeltaT, f_min = freqmin, f_ref = freqmin,
                              params = lalDict, approximant = lalsim.EccentricTD)


hp_TS = pycbc.types.timeseries.TimeSeries(hp.data.data, delta_t=hp.deltaT)
hc_TS = pycbc.types.timeseries.TimeSeries(hc.data.data, delta_t=hc.deltaT)
epochTD = hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds/1e9
timesTD = np.arange(hp.data.length)*hp.deltaT + epochTD
np.savetxt('SimInspiralCTD.txt', np.column_stack([timesTD,hp_TS,hc_TS]))

ax2.plot(timesTD, hp_TS, label='hp')
ax2.plot(timesTD, hc_TS, label = 'hc')
ax2.set_title("SimInspiralChooseTDWaveform()")
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Strain')
ax2.legend()
plt.subplots_adjust(hspace = 0.5)

############################################################################################
# Test different parameters for SimInspiralTD generation method

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
    np.savetxt('SimInspiral_{}M_{}ecc.txt'.format((mass1 + mass2), eccmin), np.column_stack([timesTD,hp_TS,hc_TS]))

    # print("Absolute value of the maximum amplitude for M_tot = {}, eccmin = {}:".format((mass1 + mass2), eccmin),
    #       "\nh_x = ", abs(max(hc_TS)),
    #       "\nh_+ = ", abs(max(hp_TS)))

    return timesTD, hp_TS, hc_TS

# Low total mass and eccentricity
mass1_low = 30.
mass2_low = 30.
eccentricities_low = [0.1, 0.2, 0.3]

fig2, axs = plt.subplots(3, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5)
fig2.suptitle('Low mass and eccentricity')

for index, eccmin in enumerate(eccentricities_low):
    timesTD, hp_TS, hc_TS = SimInspiral(mass1_low, mass2_low, eccmin, freqmin=10.)
    axs[index].plot(timesTD, hp_TS, label='hp')
    axs[index].plot(timesTD, hc_TS, label = 'hc')
    axs[index].set_title("M_tot = {}, ecc = {}".format((mass1_low + mass2_low), eccmin))
    axs[index].legend(loc = 'upper left')
    axs[index].set_xlabel('Time [s]')
    axs[index].set_ylabel('Strain')


# # High total mass and eccentricity
mass1_high = 55.
mass2_high = 55.
eccentricities_high = [0.4, 0.45, 0.5]

fig3, axs = plt.subplots(3, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5)
fig3.suptitle('High mass and eccentricity')

for index, eccmin in enumerate(eccentricities_high):
    timesTD, hp_TS, hc_TS = SimInspiral(mass1_high, mass2_high, eccmin, freqmin=10.)
    axs[index].plot(timesTD, hp_TS, label='hp')
    axs[index].plot(timesTD, hc_TS, label = 'hc')
    axs[index].set_title("M_tot = {}, ecc = {}".format((mass1_high + mass2_high), eccmin))
    axs[index].legend(loc = 'upper left')
    axs[index].set_xlabel('Time [s]')
    axs[index].set_ylabel('Strain')

# plt.show()

#################################################################3

def time_difference_peak_t0(total_mass, mass_ratio, ecc):
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

#print(time_difference_peak_t0(50, 1, 0.2))





def SimInspiral_t_over_M(total_mass, mass_ratio, eccmin, freqmin, DeltaT = 1./2048., lalDict = lal.CreateDict()):
    """ Input: total_mass in M_sun, mass_ratio >= 1 (mass1 > mass2)
        Output: 
    
    """

    mass1 = total_mass/((1/mass_ratio) + 1)
    mass2 = total_mass - mass1

    timesTD, hp_TS, hc_TS = SimInspiral(mass1, mass2, eccmin, freqmin)
    time_before_merger = abs(timesTD + timesTD[-1])

    SolarTime = c.G * c.M_sun / c.c**3
    t_over_M_sol = time_before_merger / (SolarTime * total_mass )

    return t_over_M_sol, hp_TS, hc_TS

# time_before_merger = 100e-3
# total_mass = 20
# SolarTime = c.G * c.M_sun / c.c**3
# t_over_M_sol = time_before_merger / (SolarTime * total_mass )
# print(t_over_M_sol)

# plt.figure(5, figsize=(8,4))
# plt.plot(t_over_M, hp_TS, label='hp')
# plt.plot(t_over_M, hc_TS, label = 'hc')
# plt.gca().invert_xaxis()
# plt.title("SimInspiral_M()")
# plt.xlabel('t/M [s/kg]')
# plt.ylabel('Strain')
# plt.legend()
# plt.subplots_adjust(hspace = 0.5)
# plt.show()

M_total = [20, 30, 40]

fig4, axs = plt.subplots(4, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5)
fig4.suptitle('Time scaled in units of mass')

for index, total_mass in enumerate(M_total):
    
    t_over_M, hp_TS, hc_TS = SimInspiral_t_over_M(total_mass, mass_ratio=1, eccmin=0.3, freqmin=10.)
    axs[index].plot(t_over_M, hp_TS, label='hp')
    axs[index].plot(t_over_M, hc_TS, label = 'hc')
    # axs[index].set_xlim([-1000, 30000])
    # axs[index].set_ylim([-5e-22, 5e-22])
    axs[index].set_title("M_tot = {}".format(total_mass))
    axs[index].legend(loc = 'upper left')
    axs[index].set_xlabel('t [M_total]')
    axs[index].set_ylabel('Strain')
    axs[index].invert_xaxis()
    print('Strain is calculated')
    
fig4.savefig('freq_over_M.png')
plt.show()

# fig5, axs = plt.subplots(2, figsize=(10, 10))
# plt.subplots_adjust(hspace=0.5)
# fig4.suptitle('t/M 20')

# t_over_M, hp_TS_M, hc_TS_M = SimInspiral_t_over_M(total_mass=20., mass_ratio=1, eccmin=0.3, freqmin=10./total_mass)
# timesTD, hp_TS, hc_TS = SimInspiral(total_mass=20., mass_ratio=1, eccmin=0.3, freqmin=10./total_mass)

# axs[0].plot(t_over_M, hp_TS, label='hp')
# axs[0].plot(t_over_M, hc_TS, label = 'hc')
# axs[0].set_title("M_tot = {}".format(20))
# axs[0].legend(loc = 'upper left')
# axs[0].set_xlabel('t/M [s/kg]')
# axs[0].set_ylabel('Strain')
# axs[0].invert_xaxis()

# axs[1].plot(timesTD, hp_TS, label='hp')
# axs[1].plot(timesTD, hc_TS, label = 'hc')
# axs[1].set_title("M_tot = {}".format(20))
# axs[1].legend(loc = 'upper left')
# axs[1].set_xlabel('time [s]')
# axs[1].set_ylabel('Strain')
# # axs[1].invert_xaxis()




