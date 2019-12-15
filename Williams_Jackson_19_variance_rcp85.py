import numpy as np
from scipy import stats
from scipy import integrate
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import pandas as pd

import fair
from fair.forward import fair_scm
from fair.RCPs import rcp85

import sys
sys.path.append(r'\Users\Asus\Desktop\Uni_4th_year\MPhys_project')
from functions import running_mean, discharge, glaciers, fettweis, GrIS_smb, AIS_smb, thermosteric2

## 07/03/2019 --- Script to plot normalized variance timeseries for each component 

## Read in RCP2.6 temperature pathways from Excel file

df = pd.read_excel('tas.models.rcp85.xlsx')

m = np.shape(df)[1]
 
total_var_mean    = np.zeros(100)

steric_var_mean   = np.zeros(100)

glac_var_mean     = np.zeros(100)

GrIS_SMB_var_mean = np.zeros(100)

GrIS_D_var_mean   = np.zeros(100)

AIS_SMB_var_mean  = np.zeros(100)

AIS_D_var_mean    = np.zeros(100)


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

    steric_var_mean += np.multiply(np.divide(np.square(steric[2000-1850:]-steric_u[2000-1850:])+ np.square(steric[2000-1850:]-steric_d[2000-1850:]), 2), 1)
    
    # Glaciers
    f         = 4.2175 # [mm K^-1 yr^-1]
    p         = 0.709  # dimensionless
    
    glac      = glaciers(T_pre_ind[np.where(time_full == 2006)[0][0]:], f, p)
    glac_u    = glaciers(T_pre_ind[np.where(time_full == 2006)[0][0]:], f+1, p+0.02885)
    glac_d    = glaciers(T_pre_ind[np.where(time_full == 2006)[0][0]:], f-1, p-0.02885)

    glacier_rms_u = np.sqrt(np.square(glac-glac_u))
    glacier_rms_d = np.sqrt(np.square(glac-glac_d))

    glac_var_mean += np.multiply(np.divide(np.square(np.pad(glac, (6, 0), 'constant')-np.pad(glac_u, (6, 0), 'constant'))+ np.square(np.pad(glac, (6, 0), 'constant')-np.pad(glac_d, (6, 0), 'constant')), 2), 1)
    
    # GrIS_SMB

    ## --- TO-DO: Find upper/lower GrIS_smb bounds!
        # --> Implemented by taking 1-sigma from lognormal (mu=0, sigma = 0.4) and the 1-1.15 uniform prob function, see de vries et al

    GrIS_SMB  = -1.075*GrIS_smb(T_1980_1999) # SHOULD BE RELATIVE TO 1980-1999 MEAN !!!

    GrIS_SMB_u  = -1.15*1.49*GrIS_smb(T_1980_1999) # SHOULD BE RELATIVE TO 1980-1999 MEAN !!!
    GrIS_SMB_d  = -1*0.67*GrIS_smb(T_1980_1999) # SHOULD BE RELATIVE TO 1980-1999 MEAN !!!

    GrIS_SMB_rms_u = np.sqrt(np.square(GrIS_SMB-GrIS_SMB_u))
    GrIS_SMB_rms_d = np.sqrt(np.square(GrIS_SMB-GrIS_SMB_d))

    GrIS_SMB_var_mean += np.multiply(np.divide(np.square(GrIS_SMB-GrIS_SMB_u)+ np.square(GrIS_SMB-GrIS_SMB_d), 2), 1)

    
    # GrIS_Discharge
    
    GrIS_D    = -discharge(T_1980_1999, 67.2, -919.9) # SHOULD BE RELATIVE TO 1980-1999 MEAN !!!
    GrIS_D_u  = -discharge(T_1980_1999, 148.7, -1067.9-126)
    GrIS_D_d  = -discharge(T_1980_1999, -14.4, -771.9+84)

    GrIS_D_rms_u = np.sqrt(np.square(GrIS_D-GrIS_D_u))
    GrIS_D_rms_d = np.sqrt(np.square(GrIS_D-GrIS_D_d))

    GrIS_D_var_mean += np.multiply(np.divide(np.square(GrIS_D-GrIS_D_u)+ np.square(GrIS_D-GrIS_D_d), 2), 1)

    
    # AIS_SMB
    ref       = 1983 # +- 122
    amp       = 1.1  # +- 0.2
    p_inc     = 5.1  # +- 1.5
    
    AIS_SMB   = -AIS_smb(T_1980_1999, ref, amp, p_inc)
    AIS_SMB_u = -AIS_smb(T_1980_1999, ref+122, amp+0.2, p_inc+1.5)
    AIS_SMB_d = -AIS_smb(T_1980_1999, ref-122, amp-0.2, p_inc-1.5)

    AIS_SMB_rms_u = np.sqrt(np.square(AIS_SMB-AIS_SMB_u))
    AIS_SMB_rms_d = np.sqrt(np.square(AIS_SMB-AIS_SMB_d))

    AIS_SMB_var_mean += np.multiply(np.divide(np.square(AIS_SMB-AIS_SMB_u)+ np.square(AIS_SMB-AIS_SMB_d), 2), 1)

    
    # AIS_Discharge
    #a2        = -2046.6  # +- 7.3
    #b2        = -501.8 # +- 194
    
    AIS_D     = -discharge(T_1980_1999, -2154.6, -116.6)
    AIS_D_u   = -discharge(T_1980_1999, -2076, -235.9-42.4)
    AIS_D_d   = -discharge(T_1980_1999, -2233.3, 2.62+51.8)

    AIS_D_rms_u = np.sqrt(np.square(AIS_D-AIS_D_u))
    AIS_D_rms_d = np.sqrt(np.square(AIS_D-AIS_D_d))

    AIS_D_var_mean += np.multiply(np.divide(np.square(AIS_D-AIS_D_u)+ np.square(AIS_D-AIS_D_d), 2), 1)

    ## Plot everything
    
    #plt.figure()

    plt.subplot(211)

    plt.plot(time_2000, T_cmip5[2000-1850:]- np.mean(T_cmip5[0:51]))

    plt.grid(True)
    plt.axhline(y=1.5, linestyle = '--')
    plt.xlabel('Year', fontsize=10, fontweight='bold')
    plt.ylabel('Temperature [K]', fontsize=12, fontweight='bold')
    #plt.legend(loc = 'best', fontsize=20)

total_var_mean = np.multiply(steric_var_mean + glac_var_mean + GrIS_SMB_var_mean + GrIS_D_var_mean + AIS_SMB_var_mean + AIS_D_var_mean ,1)
total_var_mean = np.divide(total_var_mean, 28)

steric_var_mean = np.divide(steric_var_mean, 28)

glac_var_mean = np.divide(glac_var_mean, 28)

GrIS_SMB_var_mean = np.divide(GrIS_SMB_var_mean, 28)

GrIS_D_var_mean = np.divide(GrIS_D_var_mean, 28)

AIS_SMB_var_mean = np.divide(AIS_SMB_var_mean, 28)

AIS_D_var_mean = np.divide(AIS_D_var_mean, 28)


plt.subplot(212)

#TOTAL variance


    
# steric
    
plt.plot(time_2000, np.divide(steric_var_mean, total_var_mean), label = 'steric')
    
# ice sheets - SMB
    
plt.plot(time_2000, np.divide(AIS_SMB_var_mean, total_var_mean), label = 'AIS_smb')
    
plt.plot(time_2000, np.divide(GrIS_SMB_var_mean, total_var_mean), label = 'GrIS_smb')
    
# ice sheets - D
    
plt.plot(time_2000, np.divide(AIS_D_var_mean, total_var_mean), label = 'AIS_D')

plt.plot(time_2000, np.divide(GrIS_D_var_mean, total_var_mean), label = 'GrIS_D')
    
# glaciers
    
plt.plot(time_2000, np.divide(glac_var_mean, total_var_mean), label = 'Glaciers')
    
    

#print(np.divide(model_mean,28))
#plt.figure()
#plt.plot(time_2000, np.divide(model_mean,28), color = 'k', linewidth = 5,label = 'Multi-model mean')
#plt.scatter(2100, np.divide(model_mean,28)[99], color = 'orange',label = 'Multi-model mean')
#plt.errorbar(2100, np.divide(model_mean,28)[99], yerr=[[np.divide(model_mean,28)[99]-0.83],[1.45 - np.divide(model_mean,28)[99]]], color = 'orange')
plt.legend(loc = 'best')
plt.show()
    


