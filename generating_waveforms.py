import lalsimulation as lalsim
import lal
import pycbc
import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform
from pycbc.waveform import td_approximants, fd_approximants

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

def SimInspiral(mass1, mass2, eccmin, DeltaT = 1./2048., freqmin = 10., lalDict = lal.CreateDict()):
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

    print("Absolute value of the maximum amplitude for M_tot = {}, eccmin = {}:".format((mass1 + mass2), eccmin),
          "\nh_x = ", abs(max(hc_TS)),
          "\nh_+ = ", abs(max(hp_TS)))

    return timesTD, hp_TS, hc_TS

# Low total mass and eccentricity
mass1_low = 30.
mass2_low = 30.
eccentricities_low = [0.1, 0.2, 0.3]

fig2, axs = plt.subplots(3, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5)
fig2.suptitle('Low mass and eccentricity')

for index, eccmin in enumerate(eccentricities_low):
    timesTD, hp_TS, hc_TS = SimInspiral(mass1_low, mass2_low, eccmin)
    axs[index].plot(timesTD, hp_TS, label='hp')
    axs[index].plot(timesTD, hc_TS, label = 'hc')
    axs[index].set_title("M_tot = {}, ecc = {}".format((mass1_low + mass2_low), eccmin))
    axs[index].legend(loc = 'upper left')
    axs[index].set_xlabel('Time [s]')
    axs[index].set_ylabel('Strain')


# High total mass and eccentricity
mass1_high = 55.
mass2_high = 55.
eccentricities_high = [0.4, 0.45, 0.5]

fig3, axs = plt.subplots(3, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5)
fig3.suptitle('High mass and eccentricity')

for index, eccmin in enumerate(eccentricities_high):
    timesTD, hp_TS, hc_TS = SimInspiral(mass1_high, mass2_high, eccmin)
    axs[index].plot(timesTD, hp_TS, label='hp')
    axs[index].plot(timesTD, hc_TS, label = 'hc')
    axs[index].set_title("M_tot = {}, ecc = {}".format((mass1_high + mass2_high), eccmin))
    axs[index].legend(loc = 'upper left')
    axs[index].set_xlabel('Time [s]')
    axs[index].set_ylabel('Strain')


plt.show()
