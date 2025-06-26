###
import pandas as pd
import numpy as np
import sys
###
from scipy import interpolate
from scipy import signal
from scipy.signal import fftconvolve, lfilter, cont2discrete
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.fft import rfft, irfft, rfftfreq
import scipy.fft as sfft 
###
from pyrocko import gf
from pyrocko import moment_tensor as pmt
###
from functools import partial
from multiprocessing import get_context
###
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
###
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.animation import FuncAnimation
###
import cmcrameri.cm as cmc
from pdb import set_trace as bp
###
fold ="./"
# sys.path.append('./Venus_Detectability/')
# sys.path.append(fold)
###


# =========================================================================================================
class Seismograms:
# =========================================================================================================

    def __init__(self, mw=6.5, strike=0., dip=45., rake=45., min_dist=500e3,
                        store_id= 'GF_venus_Cold100_qssp', depth = None,
                        north_shifts = None, east_shifts= None, gridded = True, 
                        base_folder='/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/', 
                        stf_type=None, effective_duration=25.):
        
        """
        Initialize the Seismogram class.

        This will set all parameters needed to:
        - Get seismograms from pyrocko 
        - Get seismograms from Normal Modes 
        - Read sampling rate, time, etc
        - Organise them in appropriate grids for location 
        """
        
        ### Remove sections that QSSP cannot calculate 
        self.north_shifts = north_shifts
        self.east_shifts = east_shifts
        self.north_shifts_valid = north_shifts[np.where(abs(north_shifts)>=min_dist)]
        self.east_shifts_valid = east_shifts[np.where(abs(north_shifts)>=min_dist)] 
        self.delta_dist = 10e3
        self.gridded=gridded
        ### NOTE: Add option to not calculate a grid ! 
        ### Definition of the grid 
        if self.gridded:
            self.Nn = north_shifts.size 
            self.Ne = east_shifts.size 
            self.EE, self.NN = np.meshgrid(east_shifts, north_shifts)
        else: 
            self.Nn = north_shifts.size  
            self.Ne = 1 
            self.EE, self.NN = north_shifts, east_shifts
        #if gridded: 

        self.synthetic_traces_v, self.synthetic_traces_u, self.targets_v, targets_u, iNN, self.iEE, shape_init = \
                            self._build_seismic_synthetics(mw, strike, dip, rake,
                                                          stf_type, effective_duration, depth, 
                                                          self.north_shifts_valid, self.east_shifts_valid,
                                                          store_id, base_folder, self.gridded)
        ### Useful variables
        self.dt = np.diff(self.synthetic_traces_u[0].get_xdata())[0]


    def _build_seismic_synthetics(self, mw, strike, dip, rake, stf_type, effective_duration, depth, 
                                 north_shifts, east_shifts, store_id, base_folder, gridded):

        

        ### Make a pseudo-grid of North/East 
        if gridded:
            iNN, iEE = np.meshgrid(range(north_shifts.size), range(east_shifts.size))
            shape_init = iNN.shape
            iNN, iEE = iNN.ravel(), iEE.ravel()
        else: 
            iNN, iEE = np.arange(north_shifts.size), np.arange(east_shifts.size)
            shape_init = iNN.shape
            iNN, iEE = iNN.ravel(), iEE.ravel()

        ### Source Time function properties 
        stf = dict()
        if stf_type is not None:
            if stf_type == 'boxcar':
                stf['stf'] = gf.BoxcarSTF(effective_duration=effective_duration)
            elif stf_type =="triangle":
                stf['stf'] = gf.TriangularSTF(duration=effective_duration)
            else:
                stf['stf'] = gf.HalfSinusoidSTF(duration=effective_duration)

        ### Moment Tensor definition
        scalar_moment = 10**(1.5 * mw + 9.1)
        mt_strike = pmt.MomentTensor(strike=strike, dip=dip, rake=rake, scalar_moment=scalar_moment).m6()
        mt = dict(mnn=mt_strike[0], mee=mt_strike[1], mdd=mt_strike[2], mne=mt_strike[3], mnd=mt_strike[4], med=mt_strike[5],)
        mt_source = gf.MTSource(lat=0., lon=0., depth=depth, **mt, **stf)

        ### Make velocity and displacement time series 
        waveform_targets = [
            gf.Target(
                quantity='velocity',
                lat = 0,
                lon = 0,
                north_shift=north_shift,
                east_shift=east_shift,
                store_id=store_id,
                interpolation='multilinear',
                codes=('NET', 'STA', 'LOC', 'Z'))
            for north_shift, east_shift in zip(north_shifts[iNN], east_shifts[iEE])
            ]

        waveform_targets_u = [
            gf.Target(
                quantity='displacement',
                lat = 0,
                lon = 0,
                north_shift=north_shift,
                east_shift=east_shift,
                store_id=store_id,
                interpolation='multilinear',
                codes=('NET', 'STA', 'LOC', 'Z'))
            for north_shift, east_shift in zip(north_shifts[iNN], east_shifts[iEE])
            ]

        engine = gf.LocalEngine(store_dirs=[f'{base_folder}{store_id}/'])
        response = engine.process(mt_source, waveform_targets)
        synthetic_traces = response.pyrocko_traces()

        response = engine.process(mt_source, waveform_targets_u)
        synthetic_traces_u = response.pyrocko_traces()

        return synthetic_traces, synthetic_traces_u, waveform_targets, waveform_targets_u, iNN, iEE, shape_init
    

    def arrange_interpolate_synthetics(self, tmax=2500):
        
        ### Define time array 
        t = self.synthetic_traces_v[0].get_xdata()
        self.t_new = np.arange(0., max(t.max(), tmax), self.dt)
        self.Nt = self.t_new.size

        ### Remake the pseudo grid 
        if self.gridded:
            iNN, iEE = np.meshgrid(range(self.north_shifts.size), range(self.east_shifts.size))
            shape_init = iNN.shape
            self.iNN, self.iEE = iNN.ravel(), iEE.ravel()
        else: 
            iNN, iEE = np.arange(self.north_shifts.size), np.arange(self.east_shifts.size)
            shape_init = iNN.shape
            self.iNN, self.iEE = iNN.ravel(), iEE.ravel()

        ### Where to store interpolated seismograms (grid)
        self.VEL = np.zeros((self.Ne, self.Nn, self.Nt), dtype=np.float64)

        ### Loop on calculated seismograms 
        for ii, (trace, target) in enumerate(zip(self.synthetic_traces_v, self.targets_v)):

            #print(f"Trace: {trace}, Location: N={target.north_shift/1e3}, E={target.east_shift/1e3}, depth={target.depth}")

            if self.gridded:
                iee, inn = np.where( (abs(self.NN-target.north_shift)<self.delta_dist/2) & (abs(self.EE-target.east_shift)<self.delta_dist/2))
            else: 
                inn = np.where( (abs(self.NN-target.north_shift)<self.delta_dist/2))
                iee = [0]

            t = trace.get_xdata()
            x = trace.get_ydata()

            ### Taper borders of signal before interpolating
            xt = x.copy()
            xt[:x.size//2] = x[:x.size//2]*signal.windows.tukey(x.size, alpha=0.1)[:x.size//2]
            xt[x.size//2:] = x[x.size//2:]*signal.windows.tukey(x.size, alpha=0.3)[x.size//2:]

            ### interpolate over full time range 
            xi = interpolate.interp1d(t, xt, bounds_error=False, fill_value=0.0)(self.t_new)
            
            # plot=True#False
            # if plot:
            #     if ii==0:
            #         fig, ax = plt.subplots() 
            #         ax.plot(t, x, c="k") 
            #         ax.plot(t_new, xi, c="r", ls ="--")
            #         break
                
            self.VEL[iee[0],inn[0],:] = xi

        return()


    def plot_traces(self, ns, es):

        idx = np.argmin(np.sqrt((self.north_shifts_valid/1e3-ns/1e3)**2+(self.east_shifts_valid/1e3-es/1e3)**2)) 
        # print(idx)
        
        fig = plt.figure(figsize=(8,6))
        grid = fig.add_gridspec(3, 1)

        ax = fig.add_subplot(grid[:-1,0])
        ax_t = fig.add_subplot(grid[-1,0])

        for entry in [self.synthetic_traces_v[idx], ]:
            t = entry.get_xdata()
            fs = 1./(t[1]-t[0])
            x = entry.get_ydata()

            t_new = np.arange(0., max(t.max(),2000), 1./fs)
            xi = interpolate.interp1d(t, x, bounds_error=False, fill_value=0.0)(t_new)
            t = t_new
            ax_t.plot(t_new, xi, c="k")
            
            # Compute rFFT and frequencies
            X = np.fft.rfft(x)
            freqs = np.fft.rfftfreq(len(x), 1/fs)
            magnitude = np.abs(X)*np.sqrt(1/fs/x.size)

            ax.plot(freqs, magnitude, c="k")

        ax.set_xlabel(r'Frequency / [$Hz$])')
        ax.set_ylabel(r'Amplitude spectrum / [$m/s/\sqrt{Hz}]$')
        ax.grid(True)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(freqs[1], freqs.max())
        ax.set_ylim(magnitude[-1]/100, magnitude.max()*10)
        ###
        ax_t.set_xlim(t_new.min(), t_new.max())
        ax_t.set_xlabel(r'Time since event / [$s$]')
        ax_t.set_ylabel(r'Vertical velocity / [$m/s$]')
        ###
        fig.align_labels()
        fig.tight_layout()

        return fig


# =========================================================================================================
### Defines module-level globals for parallelisation
# =========================================================================================================
list_of_locations = None
fft_vzs = None
att_exp = None
amplification= None
phase_shift_z = None
f_VER_1_27 = None
z_1_27_calc_m = None
fourier_filtering = None
b = None
a = None
loc_save = None
itime_save = None
gridded = None


# =========================================================================================================
### Function to define global parameters for parallelisation 
def init_worker(_list_of_locations, _fft_vzs, _att_exp, _amplification, _phase_shift_z,
                _f_VER_1_27, _z_1_27_calc_m, _fourier_filtering,
                _b, _a, _loc_save, _itime_save, _gridded):
    global fft_vzs, att_exp, amplification, phase_shift_z
    global f_VER_1_27, z_1_27_calc_m, fourier_filtering
    global b, a
    global list_of_locations, loc_save, itime_save, gridded

    list_of_locations = _list_of_locations
    fft_vzs = _fft_vzs
    att_exp = _att_exp
    amplification = _amplification
    phase_shift_z = _phase_shift_z
    f_VER_1_27 = _f_VER_1_27
    z_1_27_calc_m = _z_1_27_calc_m
    fourier_filtering = _fourier_filtering
    b = _b
    a = _a
    loc_save = _loc_save
    itime_save = _itime_save
    gridded = _gridded


# =========================================================================================================
### Function to propagate a group seismograms up in the atmosphere (simple model)
def propagate_attenuate(fft_vzs, i_east, i_north, att_exp, amplification, phase_shift_z):
    ### Apply attenuation and amplification at all z 
    ### shape: (Nz, Nw)
    att_vz = fft_vzs[i_east, i_north, np.newaxis, :] * att_exp
    ampl_vz_z = att_vz * amplification[:, np.newaxis]

    ### Delay at altitude z 
    fft_vz_z = ampl_vz_z *phase_shift_z

    ### Back to time domain: inverse FFT to get vz_z for these altitudes
    vz_z = np.real(sfft.ifft(fft_vz_z, axis=1))  # shape: (2, Nt)

    return(vz_z, fft_vz_z)


# =========================================================================================================
### Function to transform velocity at a specific altitude into VER perturbation 
def velocity_to_dVER_nightglow(vz_z, fft_vz_z, z_1_27_calc_m, f_VER_1_27, b,a, tf_phase_nightglow=None, fourier_filtering=False):

    z_1_27_calc_km = z_1_27_calc_m/1e3
    fver_alt = f_VER_1_27(z_1_27_calc_km)[:,np.newaxis]

    ### Filter at all altitudes 
    if not fourier_filtering:
        ### Compute VER and its vertical gradient (numpy gradient) TIME DOMAIN 
        ver_vz = fver_alt * vz_z  # shape: (Nz, Nt)
        ### VERSION OF PL 
        dver_vz_z = np.gradient(ver_vz, z_1_27_calc_m, axis=0)
        ### VERSION OF BK 
        #dver_vz_z = fver_alt * np.gradient(vz_z, self.z_1_27_calc_m, axis=0)

        dver_z = lfilter(b, a, dver_vz_z, axis=1)
    else:
        ### Compute VER and its vertical gradient (numpy gradient) FOURIER DOMAIN 
        ver_vz = fver_alt * fft_vz_z  # shape: (Nz, Nt)        
        print("here")
        ### VERSION OF PL 
        dver_vz_z = np.gradient(ver_vz, z_1_27_calc_m, axis=0)
        ### VERSION OF BK 
        # dver_vz_z = fver_alt * np.gradient(vz_z, self.z_1_27_calc_m, axis=0)

        dver_z = sfft.ifft(tf_phase_nightglow * dver_vz_z, axis=0).real


    return(dver_z)


# =========================================================================================================
### Function to integrate over line of sight (simple model)
def integrate_line_of_sight(dver_z, z_calc_m):
    ### For now, the LOS is a simple vertical line 
    #amp_dayglow = np.trapz((dVER_ad+1*dVER_tr), x=alts_dayglow, axis=1)/np.trapz(f_VER_dayglow(alts_dayglow), x=alts_dayglow,)

    ### Luminosity of airglow perturbation 
    ### dver_z shape (Nz,Nt)
    amp_airglow = np.trapz(dver_z, x=z_calc_m, axis=0) # /np.trapz(f_VER(alts_dayglow), x=alts_dayglow,)

    ### Can then be converted into total airglow luminosity by adding the integral of ver(z) 

    return(amp_airglow)


# =========================================================================================================
### Wrapper function for calculating airglow at one location (outside of class to be parallelisable)
def airglow_at_location(i_en, list_of_locations, fft_vzs, att_exp, amplification, phase_shift_z,f_VER_1_27,
                        z_1_27_calc_m, fourier_filtering, b,a, loc_save, itime_save, gridded):
    i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]

    vz_z, fft_vz_z = propagate_attenuate(fft_vzs, i_east, i_north, att_exp, amplification, phase_shift_z)
    dver_z = velocity_to_dVER_nightglow(vz_z, fft_vz_z, z_1_27_calc_m, f_VER_1_27, b, a, fourier_filtering= fourier_filtering)
    amp_airglow = integrate_line_of_sight(dver_z, z_1_27_calc_m)
            
    ### Store wavefield info 
    # if gridded:
    #     save_wavefield[i_east, i_north, :,:,0] = vz_z[:,itime_save]    ### Save velocity waveform 
    #     save_wavefield[i_east, i_north, :,:,1] = dver_z[:,itime_save]  ### Save dVER at altitude z 
    #     save_intensity_dver[i_east, i_north,:] = amp_airglow[itime_save]  ### Save dVER at altitude z 

    ### For specific location, save all times and altitudes.
    ### NOTE: Otherwise very heavy and very slow 
    if (i_east, i_north) in loc_save:
        np.save("./results/dver_z_{:d}_{:d}".format(i_east, i_north), dver_z)
        np.save("./results/vz_z_{:d}_{:d}".format(i_east, i_north), vz_z)
        np.save("./results/I_1_27_{:d}_{:d}".format(i_east, i_north), amp_airglow)

    return(vz_z[:,itime_save], dver_z[:,itime_save], amp_airglow[itime_save])


def worker_func(i):
    # location = list_of_locations[i]
    return airglow_at_location(i, list_of_locations, fft_vzs, att_exp, amplification, 
                               phase_shift_z, f_VER_1_27, z_1_27_calc_m, 
                               fourier_filtering, b, a, loc_save, itime_save, gridded)


# =========================================================================================================
class AirglowSignal:
# =========================================================================================================

    def __init__(self, SEISMO, Nz = 10 ):
        """
        Initialize the AirglowSignal model.

        Parameters:
        - The Pre-calculated seismograms and everything from Seismograms class
        - Parameters of the precision in altitude (Nz) for the airglow integrations 
        """

        ### READ WAVEFORM PARAMETERS 
        self.__dict__.update(SEISMO.__dict__)

        ### READ ATMOSPHERIC DATA. 
        folder_data            = fold + 'data/'
        file_atmos             = f'{folder_data}profile_VCD_for_scaling_pd.csv'
        file_1_27_airglow      = f'{folder_data}VER_profile_scaled.csv'
        file_4_28_airglow      = f'{folder_data}VER_profile_dayglow.csv'
        file_airglow_kenda     = f'{folder_data}VER_profiles_from_kenda.csv'
        file_attenuation_kenda = f'{folder_data}attenuation_kenda.csv'
        use_kenda_data         = False
        
        ### CONSTRUCT INTERPOLATED ATMOSPHERIC MODELS 
        self._load_atmosphere(file_atmos, file_1_27_airglow, file_4_28_airglow, file_airglow_kenda, use_kenda_data=use_kenda_data)

        ### CONSTRUCT INTERPOLATED ABSPORTION / AMPLIFICATION MODELS 
        self.f_alpha, self.f_alpha_2d, self.f_amplification = self._load_absorption_amplification(file_attenuation_kenda)#, do_plot=True)

        ### Definitions for the calculation of 1_27 micrometer nightglow 
        self.tau = 4460 # s
        self.b, self.a = self._def_filter_nightglow()
        self.Nz = Nz
        self.z_1_27_calc_km = np.linspace(self.z_1_27_min, self.z_1_27_max, self.Nz)  #km 
        self.z_1_27_calc_m = self.z_1_27_calc_km * 1e3        
        self.dz_1_27_m = np.diff(self.z_1_27_calc_m)[0]
        ### For calculation of cumulated attenuation: 
        self.z_att_km = np.concatenate((np.arange(self.z_1_27_min,0, -self.dz_1_27_m/1e3)[1:][::-1], self.z_1_27_calc_km)) 


    def _load_atmosphere(self, file_atmos, file_1_27_airglow, file_4_28_airglow, file_airglow_kenda,
                         use_kenda_data=False, gamma_kenda=11./9.):

        atm_profile = pd.read_csv(file_atmos)
        self.f_rho = interpolate.interp1d(atm_profile.altitude/1e3, atm_profile.rho, kind='quadratic', bounds_error=False, fill_value=(atm_profile.rho.min(), atm_profile.rho.max()))
        self.f_t = interpolate.interp1d(atm_profile.altitude/1e3, atm_profile.t, kind='quadratic')
        self.f_gamma = interpolate.interp1d(atm_profile.altitude/1e3, atm_profile.gamma, kind='quadratic')
        self.f_c = interpolate.interp1d(atm_profile.altitude/1e3, atm_profile.c, kind='quadratic')

        ### Read data for 1_27 micrometer Oxygen volume emission rate
        VER = pd.read_csv(file_1_27_airglow)
        VER.columns=['VER', 'alt']
        self.f_VER_1_27 = interpolate.interp1d(VER.alt, VER.VER, kind='quadratic', bounds_error=False, fill_value=0.)
        self.z_1_27_min = VER.alt.min()
        self.z_1_27_max = VER.alt.max()

        ### Read data for 4_32 micromter CO2 volume emission rate
        VER = pd.read_csv(file_4_28_airglow)
        VER.columns=['VER', 'alt']
        VER.to_csv(file_4_28_airglow.replace('.csv', '_scaled.csv'), index=False)
        self.f_VER_4_28 = interpolate.interp1d(VER.alt, VER.VER, kind='cubic', bounds_error=False, fill_value=(VER.VER.iloc[0], VER.VER.iloc[-1]))

        ### Use a different type of data 
        if use_kenda_data:
            VER_kenda = pd.read_csv(file_airglow_kenda)

            self.f_rho = interpolate.interp1d(VER_kenda.z, VER_kenda.rho, kind='quadratic', bounds_error=False, fill_value=(VER_kenda.rho.min(), VER_kenda.rho.max()))
            self.f_t = interpolate.interp1d(VER_kenda.z, VER_kenda['T'], kind='quadratic')
            self.f_gamma = interpolate.interp1d(atm_profile.altitude/1e3, atm_profile.gamma*0.+gamma_kenda, kind='linear')  ### NOTE: Constant ? 
            self.f_c = interpolate.interp1d(VER_kenda.z, VER_kenda.c, kind='quadratic')

            self.f_VER_1_27 = interpolate.interp1d(VER_kenda.z, VER_kenda.VER_127, kind='quadratic', bounds_error=False, fill_value=0.)
            self.f_VER_4_28 = interpolate.interp1d(VER_kenda.z, VER_kenda.VER_428, kind='cubic', bounds_error=False, fill_value=(VER_kenda.VER_428.iloc[0], VER_kenda.VER_428.iloc[-1]))
            
            self.z_1_27_min = VER_kenda.z.min()
            self.z_1_27_max = VER_kenda.z.max()

        return 

    
    def _load_absorption_amplification(self, file_atteunation_kenda, do_plot=False):
        ### Then define a function to go from vz_surface to vz_z (amplification + attenuation)
        atten = pd.read_csv(file_atteunation_kenda, header=[0])
        alts = atten.alt.unique()
        freq = atten.frequency.unique()
        FF, AA = np.meshgrid(freq, alts)

        alpha = atten.alpha.values.reshape((alts.size, freq.size))
        #print(alpha.shape)
        if do_plot:
            fig2, ax2 = plt.subplots()
            cols = plt.get_cmap("viridis")
            for i in range(0,800,80) :
               ax2.plot(freq, alpha[i,:], c=cols(i/799), label = "{:.1f}".format(alts[i])) 
            ax2.set_yscale("log")
            ax2.set_xscale("log")
            ax2.legend(title="Altitude / [$km$]", ncol=2, framealpha=0.5, edgecolor="none")
            ax2.set_xlabel("Frequency / [$Hz$]")
            ax2.set_ylabel("Attenuation / [??]")

        ### 1D interpolation for alpha (as a function of frequency / for each individual altitude)
        f_alpha = interpolate.interp1d(freq, alpha, axis=1, bounds_error=False, fill_value=0.0)
        ### 2D interpolation (as a function of frequency + altitude)
        f_alpha_2d = interpolate.RegularGridInterpolator((alts,freq), alpha, method='linear',fill_value=0, bounds_error=False)
        if do_plot:
            fig, (axa, axb) = plt.subplots(1,2, figsize=(8,4))
            axa.pcolormesh(freq, alts, np.log10(alpha), cmap="viridis", vmin = np.log10(alpha.min()), vmax = np.log10(alpha.max()))
            axa.set_xscale("log")
            axa.set_ylabel("Altitude / [$km$]")
            axa.set_xlabel("Frequency / [$Hz$]")
            axa.set_title("Raw attenuation (log)")
            ###
            f_test = 10**np.linspace(np.log10(freq.min()), np.log10(freq.max()), 200)
            z_test = np.linspace(0, alts.max(), 400)
            FF, ZZ = np.meshgrid(f_test, z_test)
            RES = f_alpha_2d((ZZ,FF)) 
            axb.pcolormesh(f_test, z_test, np.log10(RES), cmap="viridis", vmin = np.log10(alpha.min()), vmax = np.log10(alpha.max()))
            axb.set_xscale("log")
            axb.set_ylabel("Altitude / [$km$]")
            axb.set_xlabel("Frequency / [$Hz$]")
            axb.set_title("Interpolated attenuation (log)")
            fig.tight_layout()
            ###

        amplification = atten.amplification.values.reshape((alts.size, freq.size))
        if do_plot:
            fig2, (ax2, ax2b) = plt.subplots(1,2,figsize=(8,4))
            for i in range(0,800,80) :
               ax2.plot(freq, amplification[i,:], c=cols(i/799), label = "{:.1f}".format(alts[i])) 
            ax2b.plot(alts, amplification[:,0], c="k")
            ax2.set_yscale("log")
            ax2.set_xscale("log")
            ax2b.set_yscale("log")
            ax2.legend(title="Altitude / [$km$]", ncol=2, framealpha=0.5, edgecolor="none")
            # ax2b.legend(title="Frequency / [$Hz$]", ncol=2, framealpha=0.5, edgecolor="none")
            ax2.set_xlabel("Frequency / [$Hz$]")
            ax2b.set_xlabel("Altitude / [$km$]")
            ax2.set_ylabel("Amplification / [??]")
            ax2b.set_ylabel("Amplification / [??]")
            fig2.suptitle("Amplification with altitude")
            fig2.tight_layout()
        #f_amplification = interpolate.interp1d(freq, amplification, kind='quadratic', axis=1, bounds_error=False, fill_value=0.0)
        ### 1D interpolatiom (it doesn't depend on frequency)
        f_amplification = interpolate.interp1d(alts, amplification[:,0], kind='quadratic')
        
        return(f_alpha, f_alpha_2d, f_amplification)
    

    def _def_filter_nightglow(self):
        ### Transfer function for OPTION 2: scipy-defined filtering 
        #tau = 4460  # seconds
        num = [self.tau]
        den = [self.tau, 1]
        system_d = cont2discrete((num, den), self.dt, method='bilinear')
        b, a = system_d[0].flatten(), system_d[1].flatten()
        return(b,a)


    def calculate_1_27_airglow(self, list_ieast, list_inorth, loc_save=None, time_save = None, fourier_filtering=False, 
                               n_cpus=10, do_parallel=False, tmax=2500):
        ### Ensure we have a time series: 
        if self.t_new is None: 
            t = self.synthetic_traces_v[0].get_xdata()
            self.t_new = np.arange(0., max(t.max(), tmax), self.dt)
            self.Nt = self.t_new.size
        ### Save vertical profiles at 10 first locations by default
        if loc_save is None:
            loc_save = list(zip(list_ieast, list_inorth))[:10]
        ### Save wavefield every 10 timesteps by default
        if time_save is None: 
            time_save = self.t_new[::int(self.Nt//10)]
        itime_save = [np.where(abs(self.t_new-tw)<self.dt/2)[0][0] for tw in time_save]
        save_wavefield = np.zeros((self.Ne, self.Nn, self.Nz, len(time_save),2 ))
        save_intensity_dver = np.zeros((self.Ne, self.Nn,len(time_save)))

        ### Define frequencies 
        freqsi = sfft.fftfreq(d=self.dt, n=self.Nt)
        freqsp = abs(freqsi)

        ### Fourier transform of seismograms 
        fft_vzs = sfft.fft(self.VEL, axis=2)

        ### Pre-calculate the phase shift corresponding to the delay with altitude 
        self.phase_shift_z = np.zeros((self.Nz, self.Nt), dtype = np.complex64)
        for jz, zz in enumerate(self.z_1_27_calc_km):
            self.phase_shift_z[jz,:] = np.exp(-2 * np.pi * freqsi * 1j * zz / 0.3)

        ### If using fourier filtering, prepare the filter: 
        if fourier_filtering:
            self.tf_phase_nightglow = -(self.tau/(1+1j*2*np.pi*freqsi[None,:]*self.tau)) 


        ### Grid of amplification 
        self.amplification = self.f_amplification(self.z_1_27_calc_km)

        ### Grid of attenuation: OPTION WITHOUT CUMULATIVE SUM 
        # FFver, ZZver = np.meshgrid(freqsp, self.z_1_27_calc_km)
        # attenuation = self.f_alpha_2d((ZZver, FFver))
        ### Exponential of attenuation (supposing Np/km)
        # self.att_exp = np.exp(-self.z_1_27_calc_km[:,np.newaxis]*attenuation)   ### meter or kilometer ? Cumulative sum or not ? 

        ### NOTE: Cumulative sum works only if we are starting from z=0
        FFver, ZZver2 = np.meshgrid(freqsp, self.z_att_km)
        attenuation = self.f_alpha_2d((ZZver2, FFver))
        att_exp = np.exp(-self.dz_1_27_m*np.cumsum(attenuation, axis=0)/1e3)   ### SupposesNp/km 
        self.att_exp = att_exp[-self.Nz:]
        # fig, ax = plt.subplots() 
        # for i in range(self.Nz):
        #     ax.plot(freqsp, self.att_exp[i,:])        
        
        ### Prepare the loop 
        list_of_locations = list(zip(list_ieast, list_inorth))
        list_indices = range(len(list_of_locations))

        ### SERIAL VERSION OF THE CALCULATION 
        if not do_parallel: 
            # import time as ptime 
            # t1=ptime.time()
            for i_en in tqdm(list_indices, total=len(list_ieast), disable=False):
                # save_wavefield, save_intensity_dver = self.airglow_at_location(i_en, 
                #                                                                 fft_vzs, fourier_filtering, 
                #                                                                 save_wavefield, save_intensity_dver, 
                #                                                                 loc_save, itime_save)
                vz_z_it, dver_z_it, amp_airglow_it = airglow_at_location(i_en, list_of_locations, fft_vzs, self.att_exp, self.amplification, 
                                                                          self.phase_shift_z,self.f_VER_1_27,
                                                                        self.z_1_27_calc_m, fourier_filtering, self.b, self.a, 
                                                                        loc_save, itime_save, self.gridded)
                if self.gridded:
                    i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]
                    save_wavefield[i_east, i_north, :,:,0] = vz_z_it    ### Save velocity waveform 
                    save_wavefield[i_east, i_north, :,:,1] = dver_z_it  ### Save dVER at altitude z 
                    save_intensity_dver[i_east, i_north,:] = amp_airglow_it  ### Save dVER at altitude z

            # t2 = ptime.time()
            # print(t2-t1)
        ### PARALLEL WITH MULTIPROCESSING (requires all functions defined outside of class)
        else:
            # import time as ptime 
            # t0=ptime.time()
            ### NOTE: "fork" is much faster than "spawn" as spawn spends time copying array to different workers 
            ### However, fork doesn't work on windows/mac. To get a more flexible but fast parallelisation, 
            ### we need to move to joblib and memory maps. 
            with get_context("fork").Pool(processes=n_cpus,
                                            initializer=init_worker,
                                            initargs=(list_of_locations, fft_vzs, self.att_exp, self.amplification, self.phase_shift_z,
                                                    self.f_VER_1_27, self.z_1_27_calc_m, fourier_filtering,
                                                    self.b, self.a, 
                                                    loc_save, itime_save, self.gridded)
                                                ) as p:
                # t1 = ptime.time()
                results = p.map(worker_func, list_indices)
                
                # t2 = ptime.time()
            # print(t1-t0)
            # print(t2-t1)

            ### Store wavefields 
            if self.gridded: 
                for i_en ,r in enumerate(results):
                    i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]
                        
                    vz_z_it = r[0]
                    dver_z_it = r[1]
                    amp_airglow_it = r[2]
                    save_wavefield[i_east, i_north, :,:,0] = vz_z_it         ### Save velocity waveform 
                    save_wavefield[i_east, i_north, :,:,1] = dver_z_it       ### Save dVER at altitude z 
                    save_intensity_dver[i_east, i_north,:] = amp_airglow_it  ### Save dVER at altitude z

            
        ### Save the full wavefield, but only at certain times 
        np.save("./results/dver_t", save_wavefield)
        np.save("./results/I_t", save_intensity_dver)


    
    def plot_nightglow_traces(self, i_east, i_north, z1=92, z2=112, photons_nightglow=2e4):
    
        ### Find the right altitude
        alts_airglow = self.z_1_27_calc_km
        iz1 = np.where( np.abs(alts_airglow-z1) == np.min(np.abs(alts_airglow-z1)))[0][0]
        iz2 = np.where( np.abs(alts_airglow-z2) == np.min(np.abs(alts_airglow-z2)))[0][0]
        print(iz1, iz2)
        ### Get appropriate signals 
        dver_z_loc = np.load("./results/dver_z_{:d}_{:d}.npy".format(i_east, i_north))
        dver_z1 = dver_z_loc[iz1,:]
        dver_z2 = dver_z_loc[iz2,:]
        vz_z_loc = np.load("./results/vz_z_{:d}_{:d}.npy".format(i_east, i_north))
        vz_z1 = vz_z_loc[iz1,:]
        vz_z2 = vz_z_loc[iz2,:]
        I_dver_nightglow = np.load("./results/I_1_27_{:d}_{:d}.npy".format(i_east, i_north))
        I_background_nightglow = np.trapz(self.f_VER_1_27(alts_airglow), alts_airglow*1e3 )

        ####################################################################################
        fig, (axt,axm, axb, axbb) = plt.subplots(4,1,figsize=(6,6) )
        axt.plot(self.t_new, self.VEL[i_east, i_north,:], c="k", lw=1) 
        axt.set_title("Ground signal")
        axt.set_ylabel("Velocity / [$m/s$]")
        ###
        axm.plot(self.t_new, vz_z1, c="navy", lw=1, label="z={:.1f} km".format(alts_airglow[iz1]))
        axm.plot(self.t_new, vz_z2, c="purple", lw=1, label="z={:.1f} km".format(alts_airglow[iz2]))
        axm.set_title("Surface signal")
        axm.set_title("Amplified, propagated, attenuated")
        axm.set_ylabel("Velocity / [$m/s$]")
        axm.legend(loc=2, frameon=False)
        ####################################################################################
        axb.plot(self.t_new, dver_z1/self.f_VER_1_27(alts_airglow[iz1])*100, c="navy", ls="-", label="z={:.1f} km".format(alts_airglow[iz1]), lw=1)
        axb.plot(self.t_new, dver_z2/self.f_VER_1_27(alts_airglow[iz1])*100, c="purple", ls="-", label="z={:.1f} km".format(alts_airglow[iz2]), lw=1)
        axb.set_title("Percentage of perturbation of VER")
        axb.legend(loc=2, frameon=False)
        axb.set_ylabel("dVER/VER / [%]")
        axb.set_xlabel("Time / [$s$]")
        ####################################################################################
        ### VERTICAL INTEGRATION 
        axbb.plot(self.t_new, I_dver_nightglow + I_background_nightglow, c="k", ls="-", lw=1)
        axbb.set_ylabel("Intensity")
        axbb.set_xlabel("Time / [$s$]")
        axbb.set_title("Integration, vertical line of sight")
        for ax in [axt, axm, axb, axbb]:
            ax.set_xlim(self.t_new.min(), self.t_new.max())
        ###
        fig.align_labels()
        fig.tight_layout()
        

    def plot_nightglow_images(self, time_save=None):
        if time_save ==None: 
            time_save = self.t_new[::int(self.Nt//10)]
            
        #wf = np.load("./results/dver_t.npy")
        wf = np.load("./results/I_t.npy")

        
        Nrow = int(np.ceil(len(time_save[1:])/3))
        Ncol = 4
        fig, axes = plt.subplots(ncols=Ncol, nrows=Nrow, 
                                 gridspec_kw= dict(width_ratios=[1,1,1,0.05]), 
                                 figsize=(6,int(6*(Nrow/Ncol))) )
        
        for it, tw  in enumerate(time_save[1:]):

            ###
            u = it%3
            v = it//3
            im = axes[v][u].pcolormesh(self.EE/1e3, self.NN/1e3, wf[:,:,it], cmap="Greys_r",vmin=np.mean(wf)-1*np.std(wf), vmax=np.mean(wf)+1*np.std(wf))
        #                        norm=colors.SymLogNorm(linthresh=np.mean(I), linscale=1,
        #                                              vmin=np.min(I), vmax=np.max(I), base=10))
            ###
            if u==2:
                fig.colorbar(im, cax = axes[v][3], label=r"I / Rayleighs ? ", fraction=0.3, pad=-60.)
            if u ==0:
                axes[v][u].set_ylabel(r"North distance / [$km$]")
            if v==Nrow-1:
                axes[v][u].set_xlabel(r"East distance / [$km$]")
            axes[v][u].set_aspect('equal', adjustable="box")
        fig.suptitle("Vertically integrated VER, {:.0f} s".format(tw))
        fig.subplots_adjust(wspace=-0.3, right=0.85, left =0.05)




# =========================================================================================================
### QUENTIN's functions 
# =========================================================================================================
def get_inputs_for_airglow(amps, amps_u, dt, f_rho, f_c, f_VER_dayglow, f_VER, north_shifts, iNN, east_shifts, iEE, ns, es, normalize_w_amplitude_at_90=False, n_add=1000, use_theoretical=False, freq_target=1./25., do_density_scaling=False, data_file='./data/attenuation_kenda.csv'):

    idx = np.argmin(np.sqrt((north_shifts[iNN]/1e3-ns)**2+(east_shifts[iEE]/1e3-es)**2))
    #dt = times[1]-times[0]
    
    GF_f0 = amps[idx,:]
    if n_add > 0:
        #GF_f0 = np.r_[np.zeros(n_add), GF_f0, np.zeros(n_add)]
        GF_f0 = np.r_[GF_f0, np.zeros(n_add)]

    GF_f0_u  = amps_u[idx,:]
    if n_add > 0:
        #GF_f0_u = np.r_[np.zeros(n_add), GF_f0_u, np.zeros(n_add)]
        GF_f0_u = np.r_[GF_f0_u, np.zeros(n_add)]

    alpha = 1e-2
    tau = 0.5*1e4 # s, after eq. 23 in Lognonne, 2016
    times_loc = np.arange(0, GF_f0_u.size)*dt
    #alts = np.linspace(90., 120., 100)
    alts_dayglow = np.linspace(90., 150., 400)
    c = f_c(alts_dayglow).mean()
    #ALTS, TIMES = np.meshgrid(alts, times_loc)
    ALTS_DAYGLOW, TIMES_DAYGLOW = np.meshgrid(alts_dayglow, times_loc)
    #dz = alts[1]-alts[0]
    amplification, Az, dzAz, dzrho, dVERdz, dVERnightglowdz = return_gradients_and_properties(alts_dayglow, f_VER_dayglow, f_VER, f_rho, do_density_scaling=do_density_scaling, freq_target=freq_target, data_file=data_file)

    amp_at_90 = 1.
    if use_theoretical:
        amp_at_90 = amplification[np.argmin(abs(alts_dayglow-90.))]
        #f0, df0dt = ar.return_stf(times_loc, A0_v=4e-2/amp_at_90, std_t=25., displacement=False, GF_f0=None, u_is_gaussian=True)
        f0, df0dt = return_stf(times_loc, A0_v=9./amp_at_90, std_t=1./freq_target, displacement=False, GF_f0=None, u_is_gaussian=True)
        GF_f0 = df0dt(times_loc)
        GF_f0_u = f0(times_loc)

    if normalize_w_amplitude_at_90:
        amp_at_90 = amplification[np.argmin(abs(alts_dayglow-90.))]
        maxval = GF_f0.max()
        GF_f0 /= maxval*amp_at_90
        GF_f0_u /= maxval*amp_at_90
    
    return TIMES_DAYGLOW, ALTS_DAYGLOW, tau, c, amplification, Az, dzAz, dzrho, alpha, dVERdz, GF_f0, GF_f0_u, amp_at_90


def load_atmosphere(folder_data=fold + 'data/', use_kenda_data=False, gamma_kenda=11./9., rel_path_to_kenda=fold + 'data/VER_profiles_from_kenda.csv'):

    #folder_data = './Venus_Detectability/data/'
    file_atmos = f'{folder_data}profile_VCD_for_scaling_pd.csv'
    profile = pd.read_csv(file_atmos)

    f_rho = interpolate.interp1d(profile.altitude/1e3, profile.rho, kind='quadratic', bounds_error=False, fill_value=(profile.rho.min(), profile.rho.max()))
    f_t = interpolate.interp1d(profile.altitude/1e3, profile.t, kind='quadratic')
    f_gamma = interpolate.interp1d(profile.altitude/1e3, profile.gamma, kind='quadratic')
    f_c = interpolate.interp1d(profile.altitude/1e3, profile.c, kind='quadratic')

    file_airglow = f'{folder_data}VER_profile_scaled.csv'
    VER = pd.read_csv(file_airglow)
    VER.columns=['VER', 'alt']
    f_VER = interpolate.interp1d(VER.alt, VER.VER, kind='quadratic', bounds_error=False, fill_value=0.)

    file_airglow = f'{folder_data}VER_profile_dayglow.csv'
    VER = pd.read_csv(file_airglow)
    VER.columns=['VER', 'alt']
    VER.to_csv(file_airglow.replace('.csv', '_scaled.csv'), index=False)
    f_VER_dayglow = interpolate.interp1d(VER.alt, VER.VER, kind='cubic', bounds_error=False, fill_value=(VER.VER.iloc[0], VER.VER.iloc[-1]))

    if use_kenda_data:
        file_airglow = f'{folder_data}{rel_path_to_kenda}'
        VER_kenda = pd.read_csv(file_airglow)

        f_rho = interpolate.interp1d(VER_kenda.z, VER_kenda.rho, kind='quadratic', bounds_error=False, fill_value=(VER_kenda.rho.min(), VER_kenda.rho.max()))
        f_t = interpolate.interp1d(VER_kenda.z, VER_kenda['T'], kind='quadratic')
        f_gamma = interpolate.interp1d(profile.altitude/1e3, profile.gamma*0.+gamma_kenda, kind='quadratic')
        f_c = interpolate.interp1d(VER_kenda.z, VER_kenda.c, kind='quadratic')

        f_VER = interpolate.interp1d(VER_kenda.z, VER_kenda.VER_127, kind='quadratic', bounds_error=False, fill_value=0.)
        f_VER_dayglow = interpolate.interp1d(VER_kenda.z, VER_kenda.VER_428, kind='cubic', bounds_error=False, fill_value=(VER_kenda.VER_428.iloc[0], VER_kenda.VER_428.iloc[-1]))

    return f_rho, f_t, f_gamma, f_c, f_VER, f_VER_dayglow


def build_seismic_synthetics(mw, depth, strike, dip, rake, store_id, north_shifts, east_shifts, base_folder='/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/', stf_type=None, effective_duration=25.):

    scalar_moment = 10**(1.5 * mw + 9.1)

    iNN, iEE = np.meshgrid(range(north_shifts.size), range(east_shifts.size))
    shape_init = iNN.shape
    iNN, iEE = iNN.ravel(), iEE.ravel()

    stf = dict()
    if stf_type is not None:
        if stf_type == 'boxcar':
            stf['stf'] = gf.BoxcarSTF(effective_duration=effective_duration)
        else:
            stf['stf'] = gf.TriangularSTF(duration=effective_duration)

    mt_strike = pmt.MomentTensor(strike=strike, dip=dip, rake=rake, scalar_moment=scalar_moment).m6()
    mt = dict(mnn=mt_strike[0], mee=mt_strike[1], mdd=mt_strike[2], mne=mt_strike[3], mnd=mt_strike[4], med=mt_strike[5],)
    mt_source = gf.MTSource(lat=0., lon=0., depth=depth, **mt, **stf)

    waveform_targets = [
        gf.Target(
            quantity='velocity',
            lat = 0,
            lon = 0,
            north_shift=north_shift,
            east_shift=east_shift,
            store_id=store_id,
            interpolation='multilinear',
            codes=('NET', 'STA', 'LOC', 'Z'))
        for north_shift, east_shift in zip(north_shifts[iNN], east_shifts[iEE])
        ]

    waveform_targets_u = [
        gf.Target(
            quantity='displacement',
            lat = 0,
            lon = 0,
            north_shift=north_shift,
            east_shift=east_shift,
            store_id=store_id,
            interpolation='multilinear',
            codes=('NET', 'STA', 'LOC', 'Z'))
        for north_shift, east_shift in zip(north_shifts[iNN], east_shifts[iEE])
        ]

    engine = gf.LocalEngine(store_dirs=[f'{base_folder}{store_id}/'])
    response = engine.process(mt_source, waveform_targets)
    synthetic_traces = response.pyrocko_traces()

    response = engine.process(mt_source, waveform_targets_u)
    synthetic_traces_u = response.pyrocko_traces()

    return synthetic_traces, synthetic_traces_u, waveform_targets, waveform_targets_u, iNN, iEE, shape_init


def build_amps_matrix(synthetic_traces, synthetic_traces_u, times, disable_bar=False):

    size_times = times.size

    amps = np.zeros((len(synthetic_traces), size_times))
    amps_u = np.zeros((len(synthetic_traces), size_times))
    for itrace, (trace, trace_u) in tqdm(enumerate(zip(synthetic_traces, synthetic_traces_u)), total=len(synthetic_traces), disable=disable_bar):
        trace_times = trace.get_xdata()
        #dt = trace_times[1]-trace_times[0]
        #window = signal.windows.tukey(trace.get_ydata().size, alpha=0.4)
        trace_resampled, trace_u_resampled = trace.get_ydata(), trace_u.get_ydata()

        trace_resampled -= trace_resampled.mean()
        #trace_resampled -= trace_resampled[0]
        trace_u_resampled -= trace_u_resampled.mean()
        #trace_u_resampled -= trace_u_resampled[0]

        #times_resampled_loc = trace_resampled.times()+trace_times[0]
        itime = np.argmin(abs(times-trace_times[0]))
        size_left = amps[itrace:itrace+1, itime:].size
        data = trace_resampled[:size_left] #*window[:size_left]
        data_u = trace_u_resampled[:size_left] #*window[:size_left]
        #print(size_left, trace_resampled.data[:size_left].shape)
        amps[itrace:itrace+1, itime:itime+data.size] = data
        amps_u[itrace:itrace+1, itime:itime+data_u.size] = data_u

    return amps, amps_u


def return_stf(times, A0_v=1., std_t=25., displacement=False, GF_f0=None, u_is_gaussian=True):
    
    if GF_f0 is None:
        t0 = 20*std_t
        t0 = 5*std_t
        std_peak = std_t/(2*np.pi)
        
        f0_u = np.exp(-((times-t0)/(2*std_peak))**2) 
        f0_v = -((times-t0)/std_peak**2)*f0_u
        if not u_is_gaussian:
            f0_temp = (((times-t0)**2 - std_peak**2) / std_peak**4)*f0_u
            f0_u = f0_v[:]
            f0_v = f0_temp
            
        #max_amp = abs(f0_v).max()
        f0 = f0_u/abs(f0_u).max() if displacement else f0_v/abs(f0_v).max()
        f0 *= A0_v
        
    else:
        f0 = GF_f0
        
    f0 = interpolate.interp1d(times, f0, kind='quadratic', bounds_error=False, fill_value=0.)
    
    df0dt = np.gradient(f0(times), times)
    df0dt = interpolate.interp1d(times, df0dt, kind='quadratic', bounds_error=False, fill_value=0)
    
    return f0, df0dt


def get_amplification(alts_dayglow):
    return np.exp(-((alts_dayglow-145.)/10.)**2)*1200.+1.


def get_amplification_freq_kenda(alts_dayglow, data_file, freq_target=1./25.):
    atten = pd.read_csv(data_file, header=[0])
    alts = atten.alt.unique()
    freq = atten.frequency.unique()
    FF, AA = np.meshgrid(freq, alts)

    alpha = atten.alpha.values.reshape((alts.size, freq.size))
    alpha = interpolate.interp1d(freq, alpha, axis=1, bounds_error=False, fill_value=0.0)
    amplification = atten.amplification.values.reshape((alts.size, freq.size))
    amplification = interpolate.interp1d(freq, amplification, axis=1, bounds_error=False, fill_value=0.0)
    #print(alpha.shape, freq.shape)
    #print(interpolate.interp1d(freq, alpha, axis=1, bounds_error=False, fill_value=0.0)(freq_target))
    #diff = abs(atten.frequency.values-freq_target)
    #freq_selected = atten.loc[diff==diff.min(), 'frequency'].values[0]
    #alpha = atten.loc[diff==diff.min(), 'alpha'].values
    #amplification = atten.loc[diff==diff.min(), 'amplification'].values
    dz = alts[1] - alts[0]
    amplification = amplification(freq_target)*np.exp(-dz*np.cumsum(alpha(freq_target), ))
    #atten[atten==0.] = atten[atten>0.].min()

    #print(alpha([1., 10.]))

    amplification = interpolate.interp1d(alts, amplification, kind='quadratic', bounds_error=False, fill_value=(amplification[0], amplification[-1]))(alts_dayglow)

    #plt.figure()
    #plt.plot(amplification, alts_dayglow,)
    #plt.legend()
    #plt.xscale('log')

    return amplification


def get_amplification_spectrum_kenda(alts_dayglow, freq_targets, deactivate_bar=True):
    atten = pd.read_csv('./data/attenuation_kenda.csv', header=[0])
    alts = atten.alt.unique()
    freq = atten.frequency.unique()
    FF, AA = np.meshgrid(freq, alts)

    alpha = atten.alpha.values.reshape((alts.size, freq.size))
    alpha = interpolate.interp1d(freq, alpha, axis=1, bounds_error=False, fill_value=0.0)
    amplification = atten.amplification.values.reshape((alts.size, freq.size))
    amplification = interpolate.interp1d(freq, amplification, kind='quadratic', axis=1, bounds_error=False, fill_value=0.0)

    #print(alpha.shape, freq.shape)
    #print(interpolate.interp1d(freq, alpha, axis=1, bounds_error=False, fill_value=0.0)(freq_target))
    #diff = abs(atten.frequency.values-freq_target)
    #freq_selected = atten.loc[diff==diff.min(), 'frequency'].values[0]
    #alpha = atten.loc[diff==diff.min(), 'alpha'].values
    #amplification = atten.loc[diff==diff.min(), 'amplification'].values
    dz = alts[1] - alts[0]
    #amplification = amplification(freq_target)*np.exp(-dz*np.cumsum(alpha(freq_target), ))
    #atten[atten==0.] = atten[atten>0.].min()
    amplification_output = np.ones((freq_targets.size, alts_dayglow.size))
    for ifreq, freq_target in tqdm(enumerate(freq_targets), total=freq_targets.size, disable=deactivate_bar):
        amp_loc = amplification(freq_target)*np.exp(-dz*np.cumsum(alpha(freq_target), ))
        amplification_output[ifreq,:] = interpolate.interp1d(alts, amp_loc, kind='quadratic', bounds_error=False, fill_value=(amp_loc[0], amp_loc[-1]))(alts_dayglow)
    #print(interpolate.interp1d(freq, amplification, axis=1, bounds_error=False, fill_value=0.0))

    #print(alpha([1., 10.]))

    #amplification = interpolate.interp1d(alts, amplification, kind='quadratic', bounds_error=False, fill_value=(amplification[0], amplification[-1]))(alts_dayglow)

    #plt.figure()
    #plt.plot(amplification, alts_dayglow,)
    #plt.legend()
    #plt.xscale('log')

    return amplification_output


def return_gradients_and_properties(alts_dayglow, f_VER_dayglow, f_VER, f_rho, do_density_scaling=False, freq_target=1./25., data_file='./data/attenuation_kenda.csv'):
    
    #amplification = get_amplification(alts_dayglow)
    amplification = get_amplification_freq_kenda(alts_dayglow, data_file, freq_target=freq_target)

    density_scaling = 1.
    if do_density_scaling:
        density_scaling = np.sqrt(f_rho(0.)/f_rho(alts_dayglow.min()))
    Az = density_scaling*amplification
    Az = interpolate.interp1d(alts_dayglow, Az, kind='quadratic', bounds_error=False, fill_value=0.)
    dzAz = np.gradient(Az(alts_dayglow), alts_dayglow)
    dzAz = interpolate.interp1d(alts_dayglow, dzAz, kind='quadratic', bounds_error=False, fill_value=0.)
    dzrho = np.gradient(f_rho(alts_dayglow), alts_dayglow)
    dzrho = interpolate.interp1d(alts_dayglow, dzrho, kind='quadratic', bounds_error=False, fill_value=0.)
                                 
    dVERdz = np.gradient(f_VER_dayglow(alts_dayglow), alts_dayglow)
    dVERdz = interpolate.interp1d(alts_dayglow, dVERdz, kind='quadratic', bounds_error=False, fill_value=(0., dVERdz[-1]))
    
    dVERnightglowdz = np.gradient(f_VER(alts_dayglow), alts_dayglow)
    dVERnightglowdz = interpolate.interp1d(alts_dayglow, dVERnightglowdz, kind='quadratic', bounds_error=False, fill_value=(0., dVERnightglowdz[-1]))
    
    return amplification, Az, dzAz, dzrho, dVERdz, dVERnightglowdz


def get_dVER_dayglow(TIMES_DAYGLOW, ALTS_DAYGLOW, c, Az, dzAz, dzrho, f0, df0dt, alpha, f_rho, f_gamma, f_t, f_VER_dayglow, dVERdz, kendas_eq=False, uz_and_dzu=None):
    
    coef = 1.
    if kendas_eq:
        coef = 0.
    
    alts_dayglow_min = ALTS_DAYGLOW[0,:].min()
    times_rescaled = TIMES_DAYGLOW - (ALTS_DAYGLOW-alts_dayglow_min)*1e3/c
    
    #uz = Az(ALTS_DAYGLOW)*f0(times_rescaled)
    #dzu = -(1/c)*df0dt(times_rescaled)*Az(ALTS_DAYGLOW) + dzAz(ALTS_DAYGLOW)*f0(times_rescaled)
    if uz_and_dzu is None:
        uz = Az(ALTS_DAYGLOW)*f0(times_rescaled)
        dzu = -(1/c)*df0dt(times_rescaled)*Az(ALTS_DAYGLOW) + dzAz(ALTS_DAYGLOW)*f0(times_rescaled)
    else:
        f_uz, f_duz = uz_and_dzu
        #print('times_rescaled', times_rescaled.shape)
        #uz = f_uz(times_rescaled)
        #dzu = f_duz(times_rescaled)
        shape_init = times_rescaled.shape
        #uz = f_uz(times_rescaled.ravel(), ALTS_DAYGLOW.ravel(), grid=False).reshape(shape_init)
        #dzu = f_duz(times_rescaled.ravel(), ALTS_DAYGLOW.ravel(), grid=False).reshape(shape_init)
        pts = np.stack([times_rescaled.ravel(), ALTS_DAYGLOW.ravel()], axis=-1)
        #print(times_rescaled.shape, pts.shape)
        uz = f_uz(pts, ).reshape(shape_init)
        dzu = f_duz(pts,).reshape(shape_init)
    udzrhodivrho = (1/f_rho(ALTS_DAYGLOW))*dzrho(ALTS_DAYGLOW)*uz

    dVER_ad = alpha*f_VER_dayglow(ALTS_DAYGLOW)*f_t(ALTS_DAYGLOW)*(f_gamma(ALTS_DAYGLOW)-1.)*(dzu + coef*udzrhodivrho)
    dVER_tr = -uz*dVERdz(ALTS_DAYGLOW)
    
    return dVER_ad, dVER_tr


def get_dVER_nightglow(TIMES_DAYGLOW, ALTS_DAYGLOW, tau, c, Az, dzAz, f0, df0dt, f_VER, vz_and_dzv=None):
    
    #coef = 1.
    #if kendas_eq:
    #    coef = 0.
        
    dt = TIMES_DAYGLOW[1,0] - TIMES_DAYGLOW[0,0]
    omega = 2 * np.pi * np.fft.fftfreq(TIMES_DAYGLOW.shape[0], d=dt)
    alts_dayglow_min = ALTS_DAYGLOW[0,:].min()
    times_rescaled = TIMES_DAYGLOW - (ALTS_DAYGLOW-alts_dayglow_min)*1e3/c
    
    #vz = Az(ALTS_DAYGLOW)*f0(times_rescaled)
    if vz_and_dzv is None:
        dzv = -(1/c)*df0dt(times_rescaled)*Az(ALTS_DAYGLOW) + dzAz(ALTS_DAYGLOW)*f0(times_rescaled)
    else:
        _, f_dvz = vz_and_dzv
        shape_init = times_rescaled.shape
        #dzv = f_dvz(times_rescaled.ravel(), ALTS_DAYGLOW.ravel(), grid=False).reshape(shape_init)
        pts = np.stack([times_rescaled.ravel(), ALTS_DAYGLOW.ravel()], axis=-1)
        dzv = f_dvz(pts).reshape(shape_init)
        
    #vzdzver = vz*dVERnightglowdz(ALTS_DAYGLOW)
    
    tf_phase_nightglow = -(tau/(1+1j*omega[:,None]*tau)) 
    signal = f_VER(ALTS_DAYGLOW)*dzv
    
    fourier_filtering = False
    if fourier_filtering:
        signal_fft = np.fft.fft(signal, axis=0)
        signal_mod_fft = tf_phase_nightglow * signal_fft
        dVER_nightglow = np.fft.ifft(signal_mod_fft, axis=0).real
        
    else:
        #h = (1.0 / tau) * np.exp(-TIMES_DAYGLOW / tau)
        #print('signal', signal.shape)
        #dVER_nightglow = np.convolve(signal, h, mode='full', axis=0)
        #print('dVER_nightglow', dVER_nightglow.shape)
        #dVER_nightglow = dVER_nightglow[:signal.shape[0],:]
        dVER_nightglow = np.empty_like(TIMES_DAYGLOW)
        t = np.arange(0, 10*tau, dt)
        h = np.exp(-t / tau)*dt
        
        #dt  = 0.01      # s,  set from your data
        #tau = 2.0       # s,  choose
        a   = np.exp(-dt/tau)      # pole
        b   = tau * (1.0 - a)              # zero
        
        for i in range(TIMES_DAYGLOW.shape[1]):
            #y = np.convolve(signal[:,i], h, mode='full')
            #dVER_nightglow[:,i] = y[:TIMES_DAYGLOW.shape[0]]
            #y = fftconvolve(signal[:,i], h, mode='same')
            y = lfilter([b], [1, -a], signal[:,i])
            dVER_nightglow[:,i] = y
    
    ## Advection term
    #dVER_nightglow += coef*vzdzver
    
    return dVER_nightglow 


def produce_one_estimate(TIMES_DAYGLOW, ALTS_DAYGLOW, tau, c, Az, dzAz, dzrho, alpha, f_rho, f_gamma, f_t, f_VER_dayglow, dVERdz, f_VER, GF_f0=None, GF_f0_u=None, std_t=10., A0_v=1., u_is_gaussian=True, use_direct_deriv=False, uz_and_dzu=None, vz_and_dzv=None):

    alts_dayglow = ALTS_DAYGLOW[0,:]
    times = TIMES_DAYGLOW[:,0]

    if use_direct_deriv and (uz_and_dzu is None):
        dt = times[1] - times[0]
        freq_targets = rfftfreq(GF_f0.size, dt)  # shape: (n_freqs_fft,)
        
        amplification_output = get_amplification_spectrum_kenda(alts_dayglow, freq_targets, deactivate_bar=True)
        #amplification_output[amplification_output<amplification_output[amplification_output>0].min()] = amplification_output[amplification_output>0].min()
        f_GF_f0 = rfft(GF_f0)[:, np.newaxis] * amplification_output
        m_GF_f0 = irfft(f_GF_f0, n=GF_f0.size, axis=0)  # shape: (n_times, n_altitudes)
        dz_m_GF_f0 = np.gradient(m_GF_f0, alts_dayglow, axis=1)
        #m_GF_f0 = interpolate.interp1d(times, m_GF_f0, axis=0, kind='quadratic', bounds_error=False, fill_value=0.)
        #dz_m_GF_f0 = interpolate.interp1d(times, dz_m_GF_f0, axis=0, kind='quadratic', bounds_error=False, fill_value=0.)
        #m_GF_f0 = RectBivariateSpline(times, alts_dayglow, m_GF_f0)
        #dz_m_GF_f0 = RectBivariateSpline(times, alts_dayglow, dz_m_GF_f0)
        m_GF_f0 = RegularGridInterpolator((times, alts_dayglow), m_GF_f0, method='linear', bounds_error=False, fill_value=0.)
        dz_m_GF_f0 = RegularGridInterpolator((times, alts_dayglow), dz_m_GF_f0, method='linear', bounds_error=False, fill_value=0.)

        #tt, aa = np.meshgrid(times, alts_dayglow)
        #tt, aa = tt.ravel(), aa.ravel()
        #print(dz_m_GF_f0(tt, aa, grid=False).shape, tt.shape)

        f_GF_f0_u = rfft(GF_f0_u)[:, np.newaxis] * amplification_output
        m_GF_f0_u = irfft(f_GF_f0_u, n=GF_f0_u.size, axis=0)  # shape: (n_times, n_altitudes)
        dz_m_GF_f0_u = np.gradient(m_GF_f0_u, alts_dayglow, axis=1)
        #m_GF_f0_u = interpolate.interp1d(times, m_GF_f0_u, axis=0, kind='quadratic', bounds_error=False, fill_value=0.)
        #dz_m_GF_f0_u = interpolate.interp1d(times, dz_m_GF_f0_u, axis=0, kind='quadratic', bounds_error=False, fill_value=0.)
        #m_GF_f0_u = RectBivariateSpline(times, alts_dayglow, m_GF_f0_u)
        #dz_m_GF_f0_u = RectBivariateSpline(times, alts_dayglow, dz_m_GF_f0_u)
        m_GF_f0_u = RegularGridInterpolator((times, alts_dayglow), m_GF_f0_u, method='linear', bounds_error=False, fill_value=0.)
        dz_m_GF_f0_u = RegularGridInterpolator((times, alts_dayglow), dz_m_GF_f0_u, method='linear', bounds_error=False, fill_value=0.)

        vz_and_dzv = (m_GF_f0, dz_m_GF_f0)
        uz_and_dzu = (m_GF_f0_u, dz_m_GF_f0_u)
    else:
        uz_and_dzu = None
        vz_and_dzv = None

    #GF_f0_u = amps_u[iloc,:]*window
    f0_u, df0dt = return_stf(times, A0_v=A0_v, std_t=std_t, displacement=True, GF_f0=GF_f0_u, u_is_gaussian=u_is_gaussian)
    dVER_ad, dVER_tr = get_dVER_dayglow(TIMES_DAYGLOW, ALTS_DAYGLOW, c, Az, dzAz, dzrho, f0_u, df0dt, alpha, f_rho, f_gamma, f_t, f_VER_dayglow, dVERdz, uz_and_dzu=uz_and_dzu)
    #GF_f0 = amps[iloc,:]*window
    f0, df0dt = return_stf(times, A0_v=A0_v, std_t=std_t, displacement=False, GF_f0=GF_f0, u_is_gaussian=u_is_gaussian)
    dVER_nightglow = get_dVER_nightglow(TIMES_DAYGLOW, ALTS_DAYGLOW, tau, c, Az, dzAz, f0, df0dt, f_VER, vz_and_dzv=vz_and_dzv)
    
    amp_dayglow = np.trapz((dVER_ad+1*dVER_tr), x=alts_dayglow, axis=1)/np.trapz(f_VER_dayglow(alts_dayglow), x=alts_dayglow,)
    amp_nightglow = np.trapz(dVER_nightglow, x=alts_dayglow, axis=1)/np.trapz(f_VER(alts_dayglow), x=alts_dayglow,)
    
    return amp_dayglow, amp_nightglow, dVER_nightglow, uz_and_dzu, vz_and_dzv


def build_amps_airglow_matrix(times, alts_dayglow,  tau, c, Az, dzAz, dzrho, alpha, f_rho, f_gamma, f_t, f_VER_dayglow, dVERdz, f_VER, factor_padding, use_direct_deriv, inputs):

    amps, amps_u, icpu = inputs

    dt_new = times[1] - times[0]
    amps_dayglow = np.zeros(amps.shape)
    amps_nightglow = np.zeros(amps_dayglow.shape)
    opt_computation = {}
    for iloc in tqdm(range(amps.shape[0]), total=amps.shape[0], disable=not icpu==0):
        
        GF_f0_u = amps_u[iloc,:]#*window
        GF_f0 = amps[iloc,:]#*window
        GF_f0_u = (GF_f0_u-GF_f0_u.mean())*1.
        GF_f0 = (GF_f0-GF_f0.mean())*1.
        
        n_add = int(GF_f0.size*factor_padding)
        GF_f0 = np.r_[GF_f0, np.zeros(n_add)]
        GF_f0_u = np.r_[GF_f0_u, np.zeros(n_add)]
        times_loc = np.arange(0, GF_f0_u.size)*dt_new
        ALTS_DAYGLOW, TIMES_DAYGLOW = np.meshgrid(alts_dayglow, times_loc)
        
        amp_dayglow, amp_nightglow, _, uz_and_dzu, vz_and_dzv = produce_one_estimate(TIMES_DAYGLOW, ALTS_DAYGLOW, tau, c, Az, dzAz, dzrho, alpha, f_rho, f_gamma, f_t, f_VER_dayglow, dVERdz, f_VER, GF_f0=GF_f0, GF_f0_u=GF_f0_u, use_direct_deriv=use_direct_deriv, **opt_computation)
        if iloc == 0:
            opt_computation = dict(uz_and_dzu=uz_and_dzu, vz_and_dzv=vz_and_dzv)

        amps_dayglow[iloc:iloc+1, :] = amp_dayglow[:-n_add]
        amps_nightglow[iloc:iloc+1, :] = amp_nightglow[:-n_add]

    return amps_dayglow, amps_nightglow


def build_amps_airglow_matrix_CPUs(amps, amps_u, times, alts_dayglow,  tau, c, Az, dzAz, dzrho, alpha, f_rho, f_gamma, f_t, f_VER_dayglow, dVERdz, f_VER, factor_padding=1.25, use_direct_deriv=False, nb_CPU=12):

    nb_chunks = amps.shape[0]
    partial_build_amps_airglow_matrix = partial(build_amps_airglow_matrix, times, alts_dayglow,  tau, c, Az, dzAz, dzrho, alpha, f_rho, f_gamma, f_t, f_VER_dayglow, dVERdz, f_VER, factor_padding, use_direct_deriv)
        
    N = min(nb_CPU, nb_chunks)
    ## If one CPU requested, no need for deployment
    if N == 1:
        print('Running serial')
        amps_dayglow, amps_nightglow = partial_build_amps_airglow_matrix( (amps, amps_u, 0) )

    ## Otherwise, we pool the processes
    else:
    
        step_idx =  nb_chunks//N
        list_of_lists = []
        idxs = []
        for i in range(N):
            idx = np.arange(i*step_idx, (i+1)*step_idx)
            if i == N-1:
                idx = np.arange(i*step_idx, nb_chunks)
            idxs.append(idx)
            list_of_lists.append( (amps[idx,:], amps_u[idx,:], i) )

        with get_context("spawn").Pool(processes = N) as p:
            print(f'Running across {N} CPU')
            results = p.map(partial_build_amps_airglow_matrix, list_of_lists)
            p.close()
            p.join()

        amps_dayglow = np.zeros(amps.shape)
        amps_nightglow = np.zeros(amps_dayglow.shape)
        for idx, result in zip(idxs, results):
            amps_dayglow_loc, amps_nightglow_loc = result
            amps_dayglow[idx,:] = amps_dayglow_loc[:]
            amps_nightglow[idx,:] = amps_nightglow_loc[:]

    return amps_dayglow, amps_nightglow


def get_idx_time(time):
    return np.argmin(abs(times-time))


def detrend_simple_2d(data):
    n_samples, n_time = data.shape
    start = data[:, 0][:, None]         # shape (n_samples, 1)
    end = data[:, -1][:, None]          # shape (n_samples, 1)
    trend = np.linspace(0, 1, n_time)   # shape (n_time,)
    trend = start + (end - start) * trend  # shape (n_samples, n_time)
    #print(trend)
    return data - trend


def interpolate_map(east_shifts, north_shifts, amps_in, shape_init): #amps_u[:,820]
    
    interp = RegularGridInterpolator((east_shifts, north_shifts), amps_in.reshape(shape_init), method='linear')

    xf = np.linspace(east_shifts.min(), east_shifts.max(), east_shifts.size*2)
    yf = np.linspace(north_shifts.min(), north_shifts.max(), north_shifts.size*2)
    Xf, Yf = np.meshgrid(xf, yf, indexing='ij')
    points = np.stack([Xf.ravel(), Yf.ravel()], axis=-1)

    Zf = interp(points).reshape(xf.size, yf.size)
    
    return xf, yf, Zf


def plot_maps(times, north_shifts, east_shifts, iNN, iEE, amps_dayglow, amps_nightglow, loc_stat, loc_time, shape_init, use_SNR=True):

    photons_dayglow = 3.5e5
    photons_nightglow = 2e4

    #loc_stat = [(0., 2000.), (1000., 1000.), (2000., 0.)]
    idx_time = np.argmin(abs(times-loc_time))
    density_scaling = 1.#np.sqrt(f_rho(0.)/f_rho(90.))
    #window = signal.windows.tukey(amps_dayglow.shape[1], alpha=0.2)
    window = 1.

    cmap = plt.get_cmap('viridis')  # or 'plasma', 'inferno', 'coolwarm', etc.
    n = len(loc_stat)
    colors = [cmap(i / (n - 1)) for i in range(n)]

    fig = plt.figure(figsize=(10,7))
    grid = fig.add_gridspec(5, 2)

    for itype, type_unknown in enumerate(['dayglow', 'nightglow']):

        if use_SNR:
            label = 'SNR'
            unknown = density_scaling*np.sqrt(photons_dayglow)*amps_dayglow
            if type_unknown == 'nightglow':
                unknown = density_scaling*np.sqrt(photons_nightglow)*amps_nightglow
        else:
            label = 'Photons'
            #noise = np.sqrt(photons_dayglow)*np.random.rand(*amps_dayglow.shape) # White noise distribution
            noise = np.random.poisson(np.sqrt(photons_dayglow), amps_dayglow.shape) # Poisson noise distribution
            unknown = density_scaling*photons_dayglow*amps_dayglow + noise
            if type_unknown == 'nightglow':
                unknown = density_scaling*photons_nightglow*amps_nightglow + noise
                
            
        amps_loc = detrend_simple_2d(unknown*window)
        vv = np.quantile(abs(amps_loc[amps_loc>0]), q=0.999)
        max_SNR = vv
        opt_vmin = dict(vmin=-vv, vmax=vv)

        xf, yf, amps_interp = interpolate_map(east_shifts, north_shifts, amps_loc[:,idx_time].reshape(shape_init), shape_init)

        ax = fig.add_subplot(grid[:2,itype])
        for iloc, (ns, es) in enumerate(loc_stat):
            idx = np.argmin(np.sqrt((north_shifts[iNN]/1e3-ns)**2+(east_shifts[iEE]/1e3-es)**2))
            ax.plot(times[:amps_loc.shape[1]], amps_loc[idx,:]+iloc*max_SNR, color=colors[iloc])
        ax.axvline(times[:amps_loc.shape[1]][idx_time], color='black', ls='--')
        if itype == 0:
            ax.set_ylabel(label)

        ax = fig.add_subplot(grid[2:,itype])
        sc = ax.pcolormesh(xf/1e3, yf/1e3, amps_interp, shading='auto', cmap='coolwarm', **opt_vmin)

        axins0 = inset_axes(ax, width="2%", height="100%", loc='lower left', bbox_to_anchor=(1.02, 0., 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
        axins0.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
        cbar0 = plt.colorbar(sc, cax=axins0, extend='both')
        if itype == 1:
            cbar0.ax.set_ylabel(label, rotation=270, labelpad=16)

        ax.scatter(0., 0., marker='*', edgecolor='black', color='yellow', s=200)
        for iloc, (ns, es) in enumerate(loc_stat):
            ax.scatter(es, ns, marker='^', edgecolor='black', color=colors[iloc], s=100)
        ax.set_xlabel("East (km)")
        if itype == 0:
            ax.set_ylabel("North (km)")

    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.75)

    """
    def update(itime):
        sc.set_array(amps_loc[:, itime].reshape(shape_init).ravel())
        ax.set_title(f"Time index: {itime}")
        return sc,
    """
    #ani = FuncAnimation(fig, update, frames=amps_loc.shape[1], interval=100, blit=False)
    #ani.save("../animation_v.mp4", fps=40)


def filter_wave(waveform, f1, f2, dt):

    #b, a = signal.butter(N=10, Wn=[f1, f2], btype='bandpass', analog=False, fs=1./dt, output='ba')
    #y_tf = signal.lfilter(b, a, dirac)
    #sos = signal.butter(N=10, Wn=[f1, f2], btype='bandpass', analog=False, fs=1./dt, output='sos')
    if f1 is None:
        sos = signal.butter(N=10, Wn=f2, btype='lowpass', analog=False, fs=1./dt, output='sos')
    elif f2 is None:
        sos = signal.butter(N=10, Wn=f1, btype='highpass', analog=False, fs=1./dt, output='sos')
    else:
        sos = signal.butter(N=10, Wn=[f1, f2], btype='bandpass', analog=False, fs=1./dt, output='sos')
    return signal.sosfilt(sos, waveform)


def compute_airglow_scaler(freq_bins, store_id = 'GF_venus_Cold100_qssp', strike=45., dip=45., rake=45., ns=2500., es=2500., mw=6.5):

    f_rho, f_t, f_gamma, f_c, f_VER, f_VER_dayglow = load_atmosphere(folder_data=fold + 'data/')

    ## Construct seismic sources and stations
    epsilon = 5e3
    delta_dist = 50e3
    dists = np.arange(50.e3+epsilon*0, 8000.e3+epsilon*0, delta_dist)
    delta_depth = 5e3
    depths = np.arange(5e3, 50e3+delta_depth, delta_depth)
    offset = 3000e3
    north_shifts = np.linspace(-dists.max()+offset, dists.max()-offset, 50)[::10]
    east_shifts = np.linspace(-dists.max()+offset, dists.max()-offset, 50)[::10]

    #mw = 6.5
    depth = depths[5]
    #strike, dip, rake = 45., 90., 0. # strike slip
    #strike, dip, rake = 45., 45., 90. # reverse
    #store_id = 'GF_venus_Cold100_qssp'
    stf_type = 'triangle'
    stf_type = None
    
    ## Build seismograms over grid
    synthetic_traces, synthetic_traces_u, target, target_u, iNN, iEE, shape_init = \
                                            build_seismic_synthetics(mw, depth, strike, dip, rake, store_id, 
                                                                     north_shifts, east_shifts, 
                                                                     base_folder='/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/', stf_type=stf_type)
    
    times = np.linspace(0., 2000., 4000)
    amps, amps_u = build_amps_matrix(synthetic_traces, synthetic_traces_u, times, disable_bar=True)

    n_add = 1000
    NN, EE = np.meshgrid(north_shifts, east_shifts)

    scaling_airglow = pd.DataFrame()
    for ns, es in tqdm(zip(NN.ravel(), EE.ravel()), total=NN.ravel().size):

        idx = np.argmin(np.sqrt((north_shifts[iNN]/1e3-ns)**2+(east_shifts[iEE]/1e3-es)**2))
        dt = times[1]-times[0]

        GF_f0 = amps[idx,:]
        if n_add > 0:
            #GF_f0 = np.r_[np.zeros(n_add), GF_f0, np.zeros(n_add)]
            GF_f0 = np.r_[GF_f0, np.zeros(n_add)]

        GF_f0_u  = amps_u[idx,:]
        if n_add > 0:
            #GF_f0_u = np.r_[np.zeros(n_add), GF_f0_u, np.zeros(n_add)]
            GF_f0_u = np.r_[GF_f0_u, np.zeros(n_add)]

        alpha = 1e-2
        tau = 0.5*1e4 # s, after eq. 23 in Lognonne, 2016
        times_loc = np.arange(0, GF_f0_u.size)*dt
        alts_dayglow = np.linspace(90., 150., 400)
        c = f_c(alts_dayglow).mean()
        ALTS_DAYGLOW, TIMES_DAYGLOW = np.meshgrid(alts_dayglow, times_loc)
        do_density_scaling = False
        freq_target = 1./25.
        amplification, Az, dzAz, dzrho, dVERdz, _ = return_gradients_and_properties(alts_dayglow, f_VER_dayglow, f_VER, f_rho, do_density_scaling=do_density_scaling, freq_target=freq_target,)
            
        amp_at_90 = amplification[np.argmin(abs(alts_dayglow-90.))]
        maxval = GF_f0.max()
        GF_f0 /= maxval*amp_at_90
        GF_f0_u /= maxval*amp_at_90

        use_direct_deriv = True
        amp_dayglow, amp_nightglow, _, _, _ = produce_one_estimate(TIMES_DAYGLOW, ALTS_DAYGLOW, 
                                                                tau, c, Az, dzAz, dzrho, alpha, f_rho, f_gamma, f_t, 
                                                                f_VER_dayglow, dVERdz, f_VER, 
                                                                GF_f0=GF_f0, GF_f0_u=GF_f0_u, 
                                                                use_direct_deriv=use_direct_deriv)
        bins = freq_bins#=np.logspace(np.log10(1e-2), np.log10(1), 4)
        f_targets = []
        for ibin, (binleft, binright) in enumerate(zip(bins[:-1], bins[1:])):
            if ibin == 0:
                binleft = None
            if ibin == len(bins)-2:
                binright = None
            f_targets += [[binleft, binright]]

        #amps_dayglow, amps_nightglow = [abs(amp_dayglow).max()], [abs(amp_nightglow).max()]
        #loc_dict = dict(f1=0., f2=1., dayglow=abs(amp_dayglow).max(), nightglow=abs(amp_nightglow).max())
        #scaling_airglow = pd.concat([scaling_airglow, pd.DataFrame([loc_dict])])
        for f1, f2 in tqdm(f_targets, disable=True):

            waveform_dayglow = filter_wave(amp_dayglow, f1, f2, dt)
            #amps_dayglow.append(abs(waveform).max())

            waveform_nightglow = filter_wave(amp_nightglow, f1, f2, dt)
            #amps_nightglow.append(abs(waveform).max())

            loc_dict = dict(ns=ns, es=es, f1=f1 if f1 is not None else 0, f2=f2 if f2 is not None else 1., dayglow=abs(waveform_dayglow).max(), nightglow=abs(waveform_nightglow).max())
            scaling_airglow = pd.concat([scaling_airglow, pd.DataFrame([loc_dict])])

    scaling_airglow.reset_index(drop=True, inplace=True)
    return scaling_airglow


def plot_QSSP_traces(synthetic_traces, ns, es, north_shifts, iNN, east_shifts, iEE):

    idx = np.argmin(np.sqrt((north_shifts[iNN]/1e3-ns)**2+(east_shifts[iEE]/1e3-es)**2))
    print(idx, synthetic_traces[idx])

    fig = plt.figure(figsize=(8,6))
    grid = fig.add_gridspec(3, 1)

    ax = fig.add_subplot(grid[:-1,0])
    ax_t = fig.add_subplot(grid[-1,0])
    ax_t.set_xlabel(r'Time since event / [$s$]')
    ax_t.set_ylabel(r'Vertical velocity / [$m/s$]')
    for entry in [synthetic_traces[idx], ]:
        t = entry.get_xdata()
        fs = 1./(t[1]-t[0])
        x = entry.get_ydata()

        t_new = np.arange(0., max(t.max(),2500), 1./fs)
        xi = interpolate.interp1d(t, x, bounds_error=False, fill_value=0.0)(t_new)
        t = t_new
        
        # Compute rFFT and frequencies
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), 1/fs)
        magnitude = np.abs(X)*np.sqrt(1/fs/x.size)

        ax.plot(freqs, magnitude, c="k")
    ax.set_xlabel(r'Frequency / [$Hz$])')
    ax.set_ylabel(r'Amplitude spectrum / [$m/s/\sqrt{Hz}]$')
    ax.grid(True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(freqs[1], freqs.max())
    ax.set_ylim(magnitude.max()/1e9, magnitude.max()*10)
    ax_t.plot(t_new, xi, c="k")
    ax_t.set_xlim(t_new.min(), t_new.max())
    fig.align_labels()
    fig.tight_layout()

    return fig


def plot_airglow_traces(amp_dayglow, amp_nightglow, GF_f0, GF_f0_u, amp_at_90, f_VER_dayglow, f_VER, TIMES_DAYGLOW, ALTS_DAYGLOW, plot_SNR=False, cut_off_stf=[300., 700.], photons_dayglow=3.5e5, photons_nightglow=2e4):

    times_loc = TIMES_DAYGLOW[:,0]
    alts_dayglow = ALTS_DAYGLOW[0,:]
    dayglow_color = 'tab:orange'

    fig = plt.figure(figsize=(9,4))
    grid = fig.add_gridspec(2, 5)

    ax = fig.add_subplot(grid[0,0])
    ax.plot(times_loc, GF_f0_u*amp_at_90, color=dayglow_color)
    ax.set_ylabel('Ground\ndisplacement (m)')
    #maxval = abs(GF_f0_u*amp_at_90).max()
    #ax.set_ylim([-maxval, maxval])
    ax.tick_params(axis='both', which='both', labelbottom=False)

    ax = fig.add_subplot(grid[1,0], sharex=ax, sharey=ax)
    ax.plot(times_loc, GF_f0*amp_at_90,)
    ax.set_ylabel('Ground\nvelocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(cut_off_stf)

    ax = fig.add_subplot(grid[0,1:-1])
    label = 'dayglow'
    if plot_SNR:
        ax.plot(times_loc, amp_dayglow*np.sqrt(photons_dayglow), label=label, color=dayglow_color)
    else:
        ax.plot(times_loc, amp_dayglow, label=label, color=dayglow_color)
    #ax.plot(times, amp_dayglow)
    ax.grid()
    ax.legend(frameon=False, loc='lower right')
    ax.tick_params(axis='both', which='both', labelbottom=False)

    ax = fig.add_subplot(grid[1,1:-1], sharex=ax)
    label = 'nightglow'
    #ax.axhline(0., color='black')
    if plot_SNR:
        ax.plot(times_loc, amp_nightglow*np.sqrt(photons_nightglow), label=label)
    else:
        ax.plot(times_loc, amp_nightglow, label=label)
    ax.grid()
    ax.legend(frameon=False, loc='lower right')
    ax.set_xlabel('Time (s)')

    ax = fig.add_subplot(grid[0,-1])
    ax.plot(f_VER_dayglow(alts_dayglow), alts_dayglow, color=dayglow_color)
    ax.tick_params(axis='both', which='both', labelright=True, right=True, labelleft=False, left=False, top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xlabel(f'VER dayglow')

    ax = fig.add_subplot(grid[1,-1], sharey=ax)
    ax.plot(f_VER(alts_dayglow), alts_dayglow)
    ax.tick_params(axis='both', which='both', labelright=True, right=True, labelleft=False, left=False)
    ax.set_xlabel(f'VER nightglow')
    ax.set_ylabel(f'Altitude (km)')

    fig.subplots_adjust(wspace=0.55)


# =========================================================================================================
if __name__ == '__main__':
# =========================================================================================================

    ## Load atmosphere
    f_rho, f_t, f_gamma, f_c, f_VER, f_VER_dayglow = load_atmosphere(folder_data=fold+'data/')

    ## Construct seismic sources and stations
    epsilon = 5e3
    delta_dist = 50e3
    dists = np.arange(50.e3+epsilon*0, 8000.e3+epsilon*0, delta_dist)
    delta_depth = 5e3
    depths = np.arange(5e3, 50e3+delta_depth, delta_depth)
    offset = 3000e3
    north_shifts = np.linspace(-dists.max()+offset, dists.max()-offset, 100)
    east_shifts = np.linspace(-dists.max()+offset, dists.max()-offset, 100)

    mw = 6.5
    depth = depths[5]
    strike, dip, rake = 45., 90., 0. # strike slip
    strike, dip, rake = 45., 45., 90. # reverse
    store_id = 'GF_venus_Cold100_qssp'
    base_folder = '/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/'

    ## Build seismograms over grid
    synthetic_traces, synthetic_traces_u, iNN, iEE, shape_init = build_seismic_synthetics(mw, depth, strike, dip, rake, store_id, north_shifts, east_shifts, base_folder=base_folder, stf_type=None, effective_duration=25.)

    times = np.linspace(0., 2000., 4000)
    amps, amps_u = build_amps_matrix(synthetic_traces, synthetic_traces_u, times)

    ## Build airglowgrams over grid
    alpha = 1e-2
    tau = 0.5*1e4 # s, after eq. 23 in Lognonne, 2016
    alts_dayglow = np.linspace(90., 150., 400)
    c = f_c(alts_dayglow).mean()
    do_density_scaling = False
    use_direct_deriv = True
    nb_CPU = 12
    amplification, Az, dzAz, dzrho, dVERdz, dVERnightglowdz = return_gradients_and_properties(alts_dayglow, f_VER_dayglow, f_VER, f_rho, do_density_scaling=do_density_scaling)
    amps_dayglow, amps_nightglow = build_amps_airglow_matrix_CPUs(amps, amps_u, times, alts_dayglow,  tau, c, Az, dzAz, dzrho, alpha, f_rho, f_gamma, f_t, f_VER_dayglow, dVERdz, f_VER, factor_padding=1.25, use_direct_deriv=use_direct_deriv, nb_CPU=nb_CPU)
    bp()

    folder_wavefield = './data/airglow_wavefield/'
    np.save(f'{folder_wavefield}amps_dayglow_mw{mw:.1f}_d{depth/1e3:.0f}_st{strike:.0f}_di{dip:.0f}_ra{rake:.0f}_updated.npy', amps_dayglow)
    np.save(f'{folder_wavefield}amps_nightglow_mw{mw:.1f}_d{depth/1e3:.0f}_st{strike:.0f}_di{dip:.0f}_ra{rake:.0f}_updated.npy', amps_nightglow)
    bp()