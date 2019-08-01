# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:09:16 2018

@author: schwarz
"""

import numpy as np
from copy import deepcopy
from scipy.constants import c


### SPS --- Impedance reduction definition
class ImpedanceReduction():
    def __init__(self, Ring, RFStation, 
                 InducedVoltageFreq, filter_type, center_frequency, bandwidth,
                 time_constant, cavity_length, start_time = -1, FB_strength = 1):
        self.time = Ring.cycle_time
        self.counter = RFStation.counter
        self.frequencies = InducedVoltageFreq.freq
        self.impedance = InducedVoltageFreq.total_impedance
        self.initial_impedance = deepcopy(InducedVoltageFreq.total_impedance)
        self.freq_center = center_frequency
        self.delta_f = bandwidth
        self.FB_time = time_constant
        self.FB_strength = FB_strength
        if start_time == -1:
            self.start_time = Ring.t_rev[0]
        else:
            self.start_time = start_time
        
        if filter_type == 'fb_reduction':
            self.v = -Ring.beta[0,0]*c # particle speed [m/s]
            self.L = cavity_length # cavity length [m]
            self.Z0 = 50        # characteristic impedance [Ohm]
            self.N = 924        # comb filter parameter, samplings per turn [1]
            self.a = 15/16      # comb filter parameter, bandwidth [1]
            self.f_clock = self.N * Ring.f_rev[0]  # filter clock [Hz]
            self.b = 1/2        # high-pass filter parameter [1]
            self.k = 1/120      # open-loop gain
            
            if int(np.rint(center_frequency / 200.222e6)) == 1:  # fundamental
                self.vg = 0.0946*c  # group velocity [m/s]
                self.R2 = 27.1e3    # series impedance [Ohm/m^2]
                # high-pass filter parameter [1] M=25 for 4-sections
                self.M = int(np.round(2*(self.L*(1+self.vg/c)/(2*self.vg))
                                        * self.f_clock))
                self.ghp = 3        # high-pass filter gain [1]
                
                if np.round(self.L/0.374) == 43.0:
                    self.lpf_par1 = 2.6
                    self.lpf_par2 = 0.7
                elif np.round(self.L/0.374) == 54.0:
                    self.lpf_par1 = 2.0
                    self.lpf_par2 = 0.66
                elif np.round(self.L/0.374) == 32.0:
                    self.lpf_par1 = 3.0
                    self.lpf_par2 = 0.66
                else:
                    raise RuntimeError('Cavity not recognized')
            
            elif int(np.rint(center_frequency / 200.222e6)) == 4:  # 800MHz
                self.vg = 0.0388574*c  # [P. Kramer, priv. comm. 2019]
                self.R2 = 0.647e6  # [Ohm/m^2]; T.Bohl Note-2004-06 (v01.12 2009-05-05)
                # -26dB reduction in first lope [P. Baudrenghien, priv. comm. 2019]
                # high-pass filter parameter, M=14 for 800MHz TWC
                self.M = int(np.round(2*(self.L*(1+self.vg/c)/(2*self.vg))
                                        * self.f_clock))
                self.ghp = 10
                self.lpf_par1 = 1.0
                self.lpf_par2 = 0.66
            
            else:
                raise RuntimeError('Cavity not recognized')
            
            self.glp = self.ghp/30  # low-pass filter gain [1]
            
        # set minimum and maximum of affected frequencies to be +/-delta_f/2
        self.min_freq = self.freq_center - self.delta_f/2
        self.max_freq = self.freq_center + self.delta_f/2
        self.cond1 = self.frequencies <= self.max_freq
        self.cond2 = self.frequencies >= self.min_freq
        self.affected_indices = np.where(self.cond1*self.cond2)
        
        if filter_type == 'none':
            self.affected_indices = np.where(False)
            self.filter_func =  self.none_filter(
                    self.frequencies[self.affected_indices],
                    self.freq_center, self.delta_f)
            self.reduction_factor = np.ones(len(self.time))
        elif filter_type == 'fb_reduction':
            self.filter_func = self.fb_filter(
                    self.frequencies[self.affected_indices])
            self.reduction_factor = \
                1-np.exp(-(self.time-start_time)/self.FB_time)
        elif filter_type == 'flat':
            self.filter_func =  10*self.none_filter(
                    self.frequencies[self.affected_indices],
                    self.freq_center, self.delta_f)
            self.reduction_factor = np.ones(len(self.time))
        else:
            raise RuntimeError('Filter type '+str(filter_type)+
                    ' not recognized! Must be: fb_reduction or none')
        
    #end __init__
    
    def none_filter(self,x,x0,delta_x):
        return np.ones(len(x))
    
    # contains filter and impedance reduction
    def fb_filter(self,f):
        z = np.exp(np.pi*2j*(f-self.freq_center)/self.f_clock)
        self.z = z
        
        # caution: np.sinc(x) = sin(pi*x)/(pi*x)
        Z_rf = self.L*(self.Z0*self.R2/2)**0.5 \
            * np.sinc(self.L * (f-self.freq_center) \
                      * (1-self.vg/self.v) / self.vg)
        self.Z_rf = Z_rf
        
#        H_comb = (1-self.a)/(1-self.a*z**-self.N) * z**-self.N
        H_comb = 1
        self.H_comb = H_comb
        
        H_lp = self.glp/self.M * (1-z**-self.M)/(1-z**(-1)) * z**((self.M-1)/2)
        self.H_lp = H_lp
        H_hp = -self.ghp * (1-self.b)/4 * (1-z**-self.M)/(1-self.b*z**-self.M)\
            * (1-z**-1) * z**((self.M+1)/2)
        self.H_hp = H_hp
        # fudge implementation of low-pass filter
        # achieves -3dB at 2MHz single-sided bandwidth
        lpf = self.lpf_par1*np.sinc((self.lpf_par2/1e6)*(f-self.freq_center)/np.pi)
        self.lpf = lpf
        
        return 1/(1 + 0.45*self.k * H_comb * (H_lp + lpf*H_hp) * Z_rf)
    #end fb_reduction
    
    def track(self):
        self.impedance[self.affected_indices] = \
            self.initial_impedance[self.affected_indices] \
            * self.filter_func**(self.reduction_factor[self.counter] \
                                 * self.FB_strength)
    #end track
#end definition ImpedanceReduction

### SPS --- Impedance reduction definition
class ImpedanceReduction2():
    def __init__(self, Ring, RFStation, 
                 TravelingWaveCavity, filter_type, center_frequency, bandwidth,
                 time_constant, cavity_length, start_time = -1, FB_strength = 1):
        self.time = Ring.cycle_time
        self.counter = RFStation.counter
        self.frequencies = np.copy(TravelingWaveCavity.frequency_array)
        self.impedance = np.copy(TravelingWaveCavity.impedance)
        self.initial_impedance = np.copy(TravelingWaveCavity.impedance)
        self.freq_center = center_frequency
        self.delta_f = bandwidth
        self.FB_time = time_constant
        self.FB_strength = FB_strength
        if start_time == -1:
            self.start_time = Ring.t_rev[0]
        else:
            self.start_time = start_time
        
        if filter_type == 'fb_reduction':
#            self.L = 43*0.374   # cavity length [m]
            self.L = cavity_length # cavity length [m]
            self.vg = 0.0946*c  # group velocity [m/s]
            self.v = -Ring.beta[0,0]*c # particle speed [m/s]
            self.R2 = 27.1e3    # series impedance [Ohm/m^2]
            self.Z0 = 50        # characteristic impedance [Ohm]
            self.N = 924        # comb filter parameter, samplings per turn [1]
            self.a = 15/16      # comb filter parameter, bandwidth [1]
            # filter clock [Hz]
            self.f_clock = self.N * Ring.f_rev[0]
            self.b = 1/2        # high-pass filter parameter [1]
            # high-pass filter parameter [1] M=25 for 4-sections
            self.M = int(np.round(2*(self.L*(1+self.vg/c)/(2*self.vg))
                                    * self.f_clock))
            self.ghp = 3        # high-pass filter gain [1]
            self.glp = self.ghp/30  # low-pass filter gain [1]
            self.k = 1/120      # open-loop gain
            
            if np.round(self.L/0.374) == 43.0:
                self.lpf_par1 = 2.6
                self.lpf_par2 = 0.7
            elif np.round(self.L/0.374) == 54.0:
                self.lpf_par1 = 2.0
                self.lpf_par2 = 0.66
            elif np.round(self.L/0.374) == 32.0:
                self.lpf_par1 = 3.0
                self.lpf_par2 = 0.66
            else:
                raise RuntimeError('Cavity not recognized')
            
        # set minimum and maximum of affected frequencies to be 5*delta_f/2
        # and test that you don't under-/overshoot
        if (self.freq_center - 5*self.delta_f/2) < 0: 
            self.min_freq = 0.0
        else:
            self.min_freq = self.freq_center - 5*self.delta_f/2 # [Hz]
        if (self.freq_center + 5*self.delta_f/2) > max(self.frequencies):
            self.max_freq = max(self.frequencies)
        else:
            self.max_freq = self.freq_center + 5*self.delta_f/2 # [Hz]
        self.cond1 = np.abs(self.frequencies) <= self.max_freq
        self.cond2 = np.abs(self.frequencies) >= self.min_freq
        self.affected_indices = np.where(self.cond1*self.cond2)
#        self.affected_indices = np.where(np.abs(self.frequencies) <=self.max_freq - self.min_freq)
        
        if filter_type == 'none':
            self.affected_indices = np.where(False)
            self.filter_func =  self.none_filter(
                    self.frequencies[self.affected_indices],
                    self.freq_center, self.delta_f)
        elif filter_type == 'fb_reduction':
            self.filter_func = self.fb_filter(
                    self.frequencies[self.affected_indices])
        else:
            raise RuntimeError('Filter type '+str(filter_type)+
                    ' not recognized! Must be: fb_reduction or none')
        
        if filter_type == 'none':
            self.reduction_factor = np.ones(len(self.time))
        elif filter_type =='fb_reduction':
            self.reduction_factor = \
                1-np.exp(-(self.time-start_time)/self.FB_time)
            self.reduction_factor[np.argwhere(self.time<self.start_time)] = 0
    #end __init__
    
    def none_filter(self,x,x0,delta_x):
        return np.ones(len(x))
    
    # contains filter and impedance reduction
    def fb_filter(self,f):
#        z_m = np.exp(np.pi*2j*(f-self.freq_center)/self.f_clock)
#        z_p = np.exp(np.pi*2j*(f+self.freq_center)/self.f_clock)
#        z = z_m+z_p
        z = np.exp(np.pi*2j*f/self.f_clock)
        self.z = z
        
        # caution: np.sinc(x) = sin(pi*x)/(pi*x)
        Z_rf_m = self.L*(self.Z0*self.R2/2)**0.5 \
            * np.sinc(self.L * (f-self.freq_center) \
                      * (1-self.vg/self.v) / self.vg)
        Z_rf_p = self.L*(self.Z0*self.R2/2)**0.5 \
            * np.sinc(self.L * (f+self.freq_center) \
                      * (1-self.vg/self.v) / self.vg)
        self.Z_rf = Z_rf_m + Z_rf_p
        
        H_comb = (1-self.a)/(1-self.a*z**-self.N) * z**-self.N
#        H_comb = 1
        self.H_comb = H_comb
        
        H_lp = self.glp/self.M * (1-z**-self.M)/(1-1/z) * z**((self.M-1)/2)
        self.H_lp = H_lp
        H_hp = -self.ghp * (1-self.b)/4 * (1-z**-self.M)/(1-self.b*z**-self.M)\
            * (1-1/z) * z**((self.M+1)/2)
        self.H_hp = H_hp
        # fudge implementation of low-pass filter
        # achieves -3dB at 2MHz single-sided bandwidth
        lpf_m = self.lpf_par1*np.sinc((self.lpf_par2/1e6)*(f-self.freq_center)/np.pi)
        lpf_p = self.lpf_par1*np.sinc((self.lpf_par2/1e6)*(f+self.freq_center)/np.pi)
        self.lpf = lpf_m + lpf_p
        
        return 1/(1 + 0.45*self.k * H_comb * (H_lp + self.lpf*H_hp) * self.Z_rf)
    #end fb_reduction
    
    def track(self):
        self.impedance[self.affected_indices] = \
            self.initial_impedance[self.affected_indices] \
            * self.filter_func**(self.reduction_factor[self.counter] \
                                 * self.FB_strength)
    #end track
#end definition ImpedanceReduction
