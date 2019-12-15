import numpy as np
from scipy import stats
from scipy import integrate
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

plt.rcParams['xtick.top'] = plt.rcParams['ytick.right'] = True

import pandas as pd

import fair
from fair.forward import fair_scm
from fair.RCPs import rcp85

import sys
sys.path.append(r'/Users/andrewwilliams/Desktop/Uni_4th_year/MPhys_project')
from functions import running_mean, discharge, glaciers, fettweis, GrIS_smb, AIS_smb, thermosteric2

## 05/03/2019 --- Script to do projections of SLR to 2100 under for multi-model ensemble RCP2.6

## Read in RCP2.6 temperature pathways from Excel file

df = pd.read_excel('tas.models.rcp85.xlsx')

m = np.shape(df)[1]

model_mean = np.zeros(100)

fig, ax = plt.subplots(1,1)

# set edges
plt.ylim(-0.1, 2.5)
plt.xlim(2000, 2110)

# Generating a major/minor grid so easier to see
minor_xticks = np.arange(2000, 2110, 5)
major_xticks = np.arange(2000, 2110, 10)

minor_yticks = np.arange(0, 2.5, 0.125)
major_yticks = np.arange(0, 2.5, 0.25)

ax.set_xticks(minor_xticks, minor = True)
ax.set_xticks(major_xticks)

ax.set_yticks(minor_yticks, minor = True)
ax.set_yticks(major_yticks)

ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

for i in range(m):
    print(i)
    if i ==0:
        continue

    # Load CMIP5 temperature data and ignore the NaNs
    T_cmip5 = df.iloc[:, i].values # temperature time-series zeroed around 1986-2005 from 1850-2100

    T_cmip5[np.where(np.isnan(T_cmip5))[0]] = 0

    # Get total radiative forcing from FAIR model -- bit rough and ready, but it's okay
    C, F, T   = fair_scm(emissions = rcp85.Emissions.emissions)

    F_tot = np.zeros(len(F[:,0]))

    for i in range(len(F[0,:])): # sum over all radiative forcings
        F_tot += F[:, i]
    F_tot = F_tot[1850-1765: 2100-1765]

    # Set yearly time-series and various Temperature time-series'
    time_full = np.linspace(1850, 2099, 250).astype(int)

    time_2000 = time_full[np.where(time_full == 2000)[0][0]:]
    time_2006 = time_full[np.where(time_full == 2006)[0][0]:]
    T_pre_ind = T_cmip5 - np.mean(T_cmip5[0:50+1]) # Temperature timeseries relative to pre-industrial (1850-1900)

    T_1980_1999 = T_cmip5 - np.mean(T_cmip5[1980-1850:(1999-1850)+1]) # Temperature timeseries relative to pre-industrial (1850-1900)
    T_1980_1999 = T_1980_1999[2000-1850:]

    #T_ref     = T[np.where(time_full == 2000)[0][0]:] - np.mean(T[np.where(time_full == 1985)[0][0]:np.where(time_full == 2005)[0][0]+1]) # REPLACE WITH T_cmip5 in rest of script!!!
    T_ref     = T_cmip5[2000-1850:]
    T_g       = T_pre_ind[np.where(time_full == 2006)[0][0]:] # Temperature timeseries for glaciers


    ## Process-based SLR contributions

    # Steric

    eps       = 0.11 # +- 0.01 m YJ^-1
    alpha     = 1.13 # +- 0.325

    steric    = thermosteric2(F_tot, T_pre_ind, np.divide(eps, 10**24), alpha)
    steric_u  = thermosteric2(F_tot, T_pre_ind, np.divide(eps+0.01, 10**24), alpha-0.325)
    steric_d  = thermosteric2(F_tot, T_pre_ind, np.divide(eps-0.01, 10**24), alpha+0.325)

    # shift mean
    steric    = steric - np.mean(steric[np.where(time_full == 1986)[0][0]: np.where(time_full == 2006)[0][0]])
    steric_u  = steric_u - np.mean(steric_u[np.where(time_full == 1986)[0][0]: np.where(time_full == 2006)[0][0]])
    steric_d  = steric_d - np.mean(steric_d[np.where(time_full == 1986)[0][0]: np.where(time_full == 2006)[0][0]])

    steric_rms_u = np.sqrt(np.square(steric-steric_u))
    steric_rms_d = np.sqrt(np.square(steric-steric_d))

    # Glaciers
    f         = 4.2175 # [mm K^-1 yr^-1]
    p         = 0.709  # dimensionless

    glac      = glaciers(T_pre_ind[np.where(time_full == 2006)[0][0]:], f, p)
    glac_u    = glaciers(T_pre_ind[np.where(time_full == 2006)[0][0]:], f+1, p+0.02885)
    glac_d    = glaciers(T_pre_ind[np.where(time_full == 2006)[0][0]:], f-1, p-0.02885)

    glacier_rms_u = np.sqrt(np.square(glac-glac_u))
    glacier_rms_d = np.sqrt(np.square(glac-glac_d))

    # GrIS_SMB

    ## --- TO-DO: Find upper/lower GrIS_smb bounds!
        # --> Implemented by taking 1-sigma from lognormal (mu=0, sigma = 0.4) and the 1-1.15 uniform prob function, see de vries et al

    GrIS_SMB  = -1.075*GrIS_smb(T_1980_1999) # SHOULD BE RELATIVE TO 1980-1999 MEAN !!!

    GrIS_SMB_u  = -1.15*1.49*GrIS_smb(T_1980_1999) # SHOULD BE RELATIVE TO 1980-1999 MEAN !!!
    GrIS_SMB_d  = -1*0.67*GrIS_smb(T_1980_1999) # SHOULD BE RELATIVE TO 1980-1999 MEAN !!!

    GrIS_SMB_rms_u = np.sqrt(np.square(GrIS_SMB-GrIS_SMB_u))
    GrIS_SMB_rms_d = np.sqrt(np.square(GrIS_SMB-GrIS_SMB_d))

    # GrIS_Discharge

    GrIS_D    = -discharge(T_1980_1999, 67.2, -919.9) # SHOULD BE RELATIVE TO 1980-1999 MEAN !!!
    GrIS_D_u  = -discharge(T_1980_1999, 148.7, -1067.9-126)
    GrIS_D_d  = -discharge(T_1980_1999, -14.4, -771.9+84)

    GrIS_D_rms_u = np.sqrt(np.square(GrIS_D-GrIS_D_u))
    GrIS_D_rms_d = np.sqrt(np.square(GrIS_D-GrIS_D_d))

    # AIS_SMB
    ref       = 1983 # +- 122
    amp       = 1.1  # +- 0.2
    p_inc     = 5.1  # +- 1.5

    AIS_SMB   = -AIS_smb(T_1980_1999, ref, amp, p_inc)
    AIS_SMB_u = -AIS_smb(T_1980_1999, ref+122, amp+0.2, p_inc+1.5)
    AIS_SMB_d = -AIS_smb(T_1980_1999, ref-122, amp-0.2, p_inc-1.5)

    AIS_SMB_rms_u = np.sqrt(np.square(AIS_SMB-AIS_SMB_u))
    AIS_SMB_rms_d = np.sqrt(np.square(AIS_SMB-AIS_SMB_d))
    # AIS_Discharge
    #a2        = -2046.6  # +- 7.3
    #b2        = -501.8 # +- 194

    AIS_D     = -discharge(T_1980_1999, -2154.6, -116.6)
    AIS_D_u   = -discharge(T_1980_1999, -2076, -235.9-42.4)
    AIS_D_d   = -discharge(T_1980_1999, -2233.3, 2.62+51.8)

    AIS_D_rms_u = np.sqrt(np.square(AIS_D-AIS_D_u))
    AIS_D_rms_d = np.sqrt(np.square(AIS_D-AIS_D_d))

    ## Plot everything

    #plt.figure()
    """
    plt.subplot(211)

    plt.plot(time_2000, T_cmip5[2000-1850:]- np.mean(T_cmip5[0:51]))

    plt.grid(True)
    plt.axhline(y=1.5, linestyle = '--')
    plt.xlabel('Year', fontsize=10, fontweight='bold')
    plt.ylabel('Temperature [K]', fontsize=12, fontweight='bold')
    #plt.legend(loc = 'best', fontsize=20)



    plt.subplot(212)
    """
    # steric

    #plt.plot(time_2000, steric[2000-1850:], label = 'steric')
    #plt.fill_between(time_2000, steric_u[2000-1765:], steric_d[2000-1765:])

    # ice sheets - SMB

    #plt.plot(time_2000, AIS_SMB, label = 'AIS_smb')
    #plt.fill_between(time_2000, AIS_SMB_u, AIS_SMB_d, alpha = 0.25)

    #plt.plot(time_2000, GrIS_SMB, label = 'GrIS_smb')

    # ice sheets - D

    #plt.plot(time_2000, AIS_D, label = 'AIS_D')
    #plt.fill_between(time_2000, AIS_D_u, AIS_D_d, alpha = 0.25)

    #plt.plot(time_2000, GrIS_D, label = 'GrIS_D')
    #plt.fill_between(time_2000, GrIS_D_u, GrIS_D_d, alpha = 0.25)

    # glaciers

    #plt.plot(time_2006, glac, label = 'Glaciers')
    #plt.fill_between(time_2006, glac_u, glac_d, alpha = 0.45)

    # Total

    total = np.multiply(steric[2000-1850:] + AIS_SMB  + GrIS_D + AIS_D + GrIS_SMB, 1)
    total[6:] += glac

    model_mean += np.multiply(total, 1)

    # COMBINE UNCERTAINTIES

    total_u = np.multiply(steric_u[2000-1850:] + AIS_SMB_u + GrIS_SMB_u  + GrIS_D_u + AIS_D_u, 1)
    total_u[6:] += glac_u

    total_d = np.multiply(steric_d[2000-1850:] + AIS_SMB_d + GrIS_SMB_d + GrIS_D_d + AIS_D_d, 1)
    total_d[6:] += glac_d

    ax.fill_between(time_2000,total_d+total_d[5], total_u+total_u[5], color = '#F8B787')

    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Sea-level rise [m]', fontsize=20)

ax.plot(time_2000, np.divide(model_mean+model_mean[5],35), color = 'brown', linestyle = '--', linewidth = 1)

ax.scatter(2100, np.divide(model_mean+model_mean[5],35)[99], marker = 'x', color = 'orange',label = 'Multi-model mean, this work')
ax.errorbar(2100, np.divide(model_mean+model_mean[5],35)[99], yerr=[[np.divide(model_mean-model_mean[5],35)[99]-1.14],[2.23 - np.divide(model_mean-model_mean[5],35)[99]]], color = 'orange')

ax.scatter(2103, 0.85, marker = 'x', color = 'blue', label = 'Grinsted et al. (2010) multi-model mean')
ax.errorbar(2103, 0.85, yerr=[[0.85-0.33],[1.58-0.85]], color = 'blue')

ax.scatter(2105, 0.76, marker = 'x', color = 'red', label = 'Kopp et al. (2016) multi-model mean')
ax.errorbar(2105, 0.76, yerr=[[0.76-0.59],[1.05-0.76]], color = 'red')

ax.scatter(2107, 0.73, marker = 'x', color = 'green', label = 'AR5 multi-model mean')
ax.errorbar(2107, 0.73, yerr=[[0.73-0.53],[0.97-0.73]], color = 'green')


plt.legend(loc = 'upper left', prop={'size': 15})
plt.show()
