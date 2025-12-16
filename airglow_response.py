###
import pandas as pd
import numpy as np
import sys
###
from scipy import interpolate, integrate
from scipy import signal
from scipy.signal import fftconvolve, lfilter, cont2discrete, butter, sosfilt 
from scipy.signal.windows import tukey
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.fft import rfft, irfft, rfftfreq
import scipy.fft as sfft 
###
from pyrocko import gf
from pyrocko import cake
from pyrocko import moment_tensor as pmt
from disba import PhaseDispersion, GroupDispersion
import os
from obspy.taup import TauPyModel
from obspy.taup.velocity_model import VelocityModel
from obspy.taup.taup_create import build_taup_model
###
from functools import partial
from multiprocessing import get_context
###
from tqdm import tqdm
import time as ptime      
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
###
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import pycwt as wavelet
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

    def __init__(self, mw=6.5, strike=45., dip=45., rake=45., depth = None, stf_type=None, effective_duration=25., 
                        store_ids_dists=[('GF_venus_Cold100_qssp',50e3,8000e3)], 
                        north_shifts = None, east_shifts= None, gridded = True, 
                        base_folder='/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/'):
        
        """
        Initialize the Seismogram class.

        This will set all parameters needed to:
        - Get seismograms from pyrocko 
        - Get seismograms from Normal Modes 
        - Read sampling rate, time, etc
        - Organise them in appropriate grids for location 
        """

        self.depth = depth
        
        ### Remove sections that QSSP cannot calculate 
        self.north_shifts = north_shifts
        self.east_shifts = east_shifts
        #self.north_shifts_valid = north_shifts[np.where(np.sqrt(north_shifts**2+east_shifts**2)>=min_dist)]
        #self.east_shifts_valid = east_shifts[np.where(np.sqrt(north_shifts**2+east_shifts**2)>=min_dist)] 
        self.delta_dist = 10e3
        self.gridded=gridded
        ### NOTE: Add option to not calculate a grid ! 
        ### Definition of the grid 
        if self.gridded:
            self.Nn = north_shifts.size 
            self.Ne = east_shifts.size 
            self.EE, self.NN = np.meshgrid(east_shifts, north_shifts)       ### Complete grid 
            iEE, iNN = np.meshgrid(range(north_shifts.size), range(east_shifts.size))   ### Indices of grid 
            self.iEE, self.iNN = iEE.ravel(), iNN.ravel()
        else: 
            self.Nn = north_shifts.size  
            self.Ne = 1 
            self.EE, self.NN = east_shifts, north_shifts
            self.iEE = range(east_shifts.size)
            self.iNN = range(north_shifts.size)


        ### Build waveform from Green's functions, using several stores if needed. 
        self.synthetic_traces_v, self.synthetic_traces_u, self.targets_v, self.targets_u = [], [], [], []
        self.lNN, self.lEE = [],[]
        for (store_id, min_dist, max_dist) in store_ids_dists:
            synthetic_traces_v, synthetic_traces_u, targets_v, targets_u, lNN, lEE = \
                                self._build_seismic_synthetics(mw, strike, dip, rake,
                                                            stf_type, effective_duration, depth, 
                                                            self.north_shifts, self.east_shifts,
                                                            store_id, min_dist, max_dist, 
                                                            base_folder, self.gridded)
                                                            
            self.synthetic_traces_u += synthetic_traces_u
            self.synthetic_traces_v += synthetic_traces_v
            self.targets_u += targets_u
            self.targets_v += targets_v
            self.lEE = np.concatenate((self.lEE, lEE))
            self.lNN = np.concatenate((self.lNN, lNN))

        ### Useful variables
        self.dt = np.diff(self.synthetic_traces_u[0].get_xdata())[0]


    def _build_seismic_synthetics(self, mw, strike, dip, rake, stf_type, effective_duration, depth, 
                                 north_shifts, east_shifts, store_id, min_dist, max_dist, base_folder, gridded):

        

        ### Make a pseudo-grid of North/East 
        if gridded:
            # iNN, iEE = np.meshgrid(range(north_shifts.size), range(east_shifts.size))
            # shape_init = iNN.shape
            # iNN, iEE = iNN.ravel(), iEE.ravel()
            lNN, lEE = np.meshgrid(north_shifts, east_shifts)
            lNNo, lEEo = lNN.ravel(), lEE.ravel()
            lNN = lNNo[(np.sqrt(lNNo**2+lEEo**2)>=min_dist) & (np.sqrt(lNNo**2+lEEo**2)<max_dist)]
            lEE = lEEo[(np.sqrt(lNNo**2+lEEo**2)>=min_dist) & (np.sqrt(lNNo**2+lEEo**2)<max_dist)]
        else: 
            lNNo, lEEo = north_shifts, east_shifts
            lNN = lNNo[(np.sqrt(lNNo**2+lEEo**2)>=min_dist) & (np.sqrt(lNNo**2+lEEo**2)<max_dist)]
            lEE = lEEo[(np.sqrt(lNNo**2+lEEo**2)>=min_dist) & (np.sqrt(lNNo**2+lEEo**2)<max_dist)]

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
        if mw ==None: 
            mt_strike = [1,1,1,1,1,1]
        else:
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
            # for north_shift, east_shift in zip(north_shifts[iNN], east_shifts[iEE])
            for north_shift, east_shift in zip(lNN, lEE)
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
            # for north_shift, east_shift in zip(north_shifts[iNN], east_shifts[iEE])
            for north_shift, east_shift in zip(lNN, lEE)
            ]

        engine = gf.LocalEngine(store_dirs=[f'{base_folder}{store_id}/'])
        response = engine.process(mt_source, waveform_targets)
        synthetic_traces = response.pyrocko_traces()

        response = engine.process(mt_source, waveform_targets_u)
        synthetic_traces_u = response.pyrocko_traces()

        return synthetic_traces, synthetic_traces_u, waveform_targets, waveform_targets_u, lNN, lEE
    

    def arrange_interpolate_synthetics(self, tmax=2500, dt=0.5):
        
        ### Define time array 
        t = self.synthetic_traces_v[0].get_xdata()
        self.dt = max(self.dt, dt)
        self.t_new = np.arange(0., max(t.max(), tmax), self.dt)
        self.Nt = self.t_new.size

        ### Where to store interpolated seismograms (grid)
        self.VEL = np.zeros((self.Ne, self.Nn, self.Nt), dtype=np.float64)
        self.DIS = np.zeros((self.Ne, self.Nn, self.Nt), dtype=np.float64)

        ### Loop on calculated seismograms 
        for ii, (trace, trace_u, target) in enumerate(tqdm(zip(self.synthetic_traces_v,self.synthetic_traces_u, self.targets_v), total=len(self.targets_v), 
                                                           bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}' )):
            #print(f"Trace: {trace}, Location: N={target.north_shift/1e3}, E={target.east_shift/1e3}, depth={target.depth}")

            if self.gridded:
                iee, inn = np.where( (abs(self.NN-target.north_shift)<self.delta_dist/2) & (abs(self.EE-target.east_shift)<self.delta_dist/2))
            else: 
                inn = np.where( (abs(self.NN-target.north_shift)<self.delta_dist/2))
                iee = [0]

            t = trace.get_xdata()
            tu = trace_u.get_xdata()
            x = trace.get_ydata()
            xu = trace_u.get_ydata()

            ### Attempt: Make a traveling gaussian 
            #r = np.sqrt(target.north_shift**2+target.east_shift**2)
            #xi = np.exp(-(self.t_new-r/1e3)**2/50**2)*1/(r+100)

            ### Taper borders of signal before interpolating
            xt = x.copy()
            xt[:x.size//2] = x[:x.size//2]*signal.windows.tukey(x.size, alpha=0.1)[:x.size//2]
            xt[x.size//2:] = x[x.size//2:]*signal.windows.tukey(x.size, alpha=0.3)[x.size//2:]
            ### Same for displacement traces
            xtu = xu.copy()
            # xtu[:xu.size//2] = xu[:xu.size//2]*signal.windows.tukey(xu.size, alpha=0.1)[:xu.size//2]
            # xtu[xu.size//2:] = xu[xu.size//2:]*signal.windows.tukey(xu.size, alpha=0.3)[xu.size//2:]

            ### interpolate over full time range 
            xi = interpolate.interp1d(t, xt, bounds_error=False, fill_value=0.0)(self.t_new)
            xiu = interpolate.interp1d(tu, xtu, bounds_error=False, fill_value=(0.0, xtu[-1]))(self.t_new)
            ### Displacement
            #xi_u = interpolate.interp1d(trace_u.get_xdata(), trace.get_ydata(), bounds_error=False, fill_value=(0.0, trace.get_ydata()[-1]))(self.t_new)
            
            # plot=True#False
            # if plot:
            #     if ii==30:
            #         fig, ax = plt.subplots() 
            #         #ax.plot(t, x, c="k") 
            #         ax.plot(self.t_new, xi, c="r", ls ="--")
            #         break
                
            self.VEL[iee[0],inn[0],:] = xi
            self.DIS[iee[0],inn[0],:] = xiu
            # self.VEL[iee[0],inn[0],:] = xi_u
        print("Size of VEL, DIS (bytes): ", self.VEL.nbytes)

        return()


    def plot_traces(self, ns, es, do_interpolate=False):

        idx = np.argmin(np.sqrt((self.lNN-ns)**2+(self.lEE-es)**2)) 
        # print(idx)
        
        fig = plt.figure(figsize=(8,8))
        grid = fig.add_gridspec(4, 1)

        ax = fig.add_subplot(grid[:-2,0])
        ax_t = fig.add_subplot(grid[-2,0])
        ax_tu = fig.add_subplot(grid[-1,0])

        entry = self.synthetic_traces_v[idx]
        entry_u = self.synthetic_traces_u[idx]
        t = entry.get_xdata()
        fs = 1./(t[1]-t[0])
        x = entry.get_ydata()
        tu = entry_u.get_xdata()
        xu = entry_u.get_ydata()

        if do_interpolate:
            t_new = np.arange(0., max(t.max(),2000), 1./fs)
            ### Taper borders of signal before interpolating
            xt = x.copy()
            xt[:x.size//2] = x[:x.size//2]*signal.windows.tukey(x.size, alpha=0.1)[:x.size//2]
            xt[x.size//2:] = x[x.size//2:]*signal.windows.tukey(x.size, alpha=0.3)[x.size//2:]
            
            xtu = xu.copy()
            # xtu[:xu.size//2] = xu[:xu.size//2]*signal.windows.tukey(xu.size, alpha=0.1)[:xu.size//2]
            # xtu[xu.size//2:] = xu[xu.size//2:]*signal.windows.tukey(xu.size, alpha=0.3)[xu.size//2:]

            ### interpolate over full time range 
            xi = interpolate.interp1d(t, xt, bounds_error=False, fill_value=0.0)(t_new)
            xiu = interpolate.interp1d(tu, xtu, bounds_error=False, fill_value=(0.0, xtu[-1]))(t_new)
            t = t_new
            ax_t.plot(t_new, xi, c="navy")
            ax_tu.plot(t_new, xiu, c="k")
        else:
            t_new = t 
            xi = x
            xiu = xu
            ax_t.plot(t_new, xi, c="navy")
            ax_tu.plot(t_new, xiu, c="k")
        
        # Compute rFFT and frequencies
        X = np.fft.rfft(xi)
        Xu = np.fft.rfft(xiu)
        freqs = np.fft.rfftfreq(len(xi), 1/fs)
        freqsu = np.fft.rfftfreq(len(xiu), 1/fs)
        magnitude = np.abs(X)*np.sqrt(1/fs/xi.size)
        magnitudeu = np.abs(Xu)*np.sqrt(1/fs/xiu.size)

        ### Plot spectra 
        ax.plot(freqs, magnitude, c="navy", lw=1)
        axfb = ax.twinx()
        axfb.plot(freqsu, magnitudeu, c="k", lw=1)

        ax.set_xlabel(r'Frequency / [$Hz$])')
        ax.set_ylabel(r'Vel. spectrum / [$m/s/\sqrt{Hz}]$', color="navy")
        axfb.set_ylabel(r'Disp. spectrum / [$m/\sqrt{Hz}]$')
        ax.spines['left'].set_color('navy')
        ax.tick_params(axis='y', colors='navy')
        ax.grid(True)
        ax.set_xscale('log')
        ax.set_yscale('log')
        axfb.set_yscale('log')
        ax.set_xlim(freqs[1], freqs.max())
        ax.set_ylim(magnitude[-1]/100, magnitude.max()*10)
        axfb.set_ylim(magnitudeu[-1]/100, magnitudeu.max()*10)
        ax.set_title("Spectrum and waveform at {:.1f} km North, {:.1f} km East".format(self.targets_v[idx].north_shift/1e3, self.targets_v[idx].east_shift/1e3))
        ###
        ax_t.set_xlim(t_new.min(), t_new.max())
        # ax_t.set_xlabel(r'Time since event / [$s$]')
        ax_t.set_ylabel(r'Vertical velocity / [$m/s$]', color="navy")
        ax_t.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
        ax_t.spines['left'].set_color('navy')
        ax_t.tick_params(axis='y', colors='navy')
        # ax_t.get_yticklabels().set_color("")
        ###
        ax_tu.set_xlim(t_new.min(), t_new.max())
        ax_tu.set_xlabel(r'Time since event / [$s$]')
        ax_tu.set_ylabel(r'Vertical disp. / [$m$]')
        ax_tu.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
        ###
        fig.align_labels()
        fig.tight_layout()

        return fig
    

    def plot_wavefront(self, time_save=None):

        if time_save ==None: 
            time_save = self.t_new[::int(np.ceil(self.Nt//9))]
            itime_save = range(0,self.Nt,int(np.ceil(self.Nt//9)))
            
        Nrow = int(np.ceil(len(time_save[1:])/3))
        Ncol = 5
        fig, axes = plt.subplots(ncols=Ncol, nrows=Nrow, 
                                 gridspec_kw= dict(width_ratios=[1,1,1,0.05,0.05]), 
                                 figsize=(8,int(10*(Nrow/Ncol))) )
        
        ### Wavefront (vertical velocity)
        wf = self.VEL
        vmin = np.nanmean(wf)-0.01*(np.nanstd(wf))
        vmax = np.nanmean(wf)+0.01*(np.nanstd(wf))
        
        for ii, (it, tw)  in enumerate(zip(itime_save[1:],time_save[1:])):

            ###
            u = ii%3
            v = ii//3
            im = axes[v][u].pcolormesh(self.EE/1e3, self.NN/1e3, wf[:,:,it], cmap="Greys_r",vmin=vmin, vmax=vmax)
            ###
            if u==2:
                cbar = fig.colorbar(im, cax = axes[v][4], label=r"$V_z$ / [$m/s$] ", fraction=0.3, pad=-60.)
                cbar.formatter.set_useMathText(True)
                axes[v][3].axis("off")
            if u ==0 and v==1:
                axes[v][u].set_ylabel(r"North distance / [$km$]")
            if v==Nrow-1 and u==1:
                axes[v][u].set_xlabel(r"East distance / [$km$]")
            axes[v][u].set_aspect('equal', adjustable="box")
            axes[v][u].text(0.02, 0.98, "{:.0f} s".format(tw),
                            transform=axes[v][u].transAxes,
                            fontsize=8,
                            verticalalignment='top',
                            horizontalalignment='left',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='square,pad=0.3'))

            axes[u][v].tick_params(axis='both', which='major', labelsize=8)

        #fig.suptitle("Vertical velocity")
        axes[0][1].set_title("Vertical velocity", pad=20)
        fig.subplots_adjust(wspace=-0, hspace=0.3, right=0.85, left =0.05, top=0.93)
        return(fig)


# =========================================================================================================
### Defines module-level globals for parallelisation
# =========================================================================================================
list_of_locations = None
fft_vzs = None
fft_uzs = None
att_exp = None
amplification= None
phase_shift_z = None
Nt = None
fs = None
f_VER_1_27 = None
f_dVER_1_27 = None
f_VER_4_28  = None
f_dVER_4_28 = None
z_1_27_calc_m = None
z_4_28_calc_m = None
fac_temperature = None 
fourier_filtering = None
b = None
a = None
loc_save = None
itime_save = None
save_ver = None
gridded = None
dir_save = None 
dphase_att_ampl_dz = None 
zatt = None
zrange = None

# =========================================================================================================
### Calculate theoretical arrival times for P, S and RW 
# =========================================================================================================
def get_arrival_obspy(dist_m, source_depth, velocity_model, model=None, phase="s", r_planet=6371):
    ### See https://github.com/obspy/obspy/issues/2816

    if model is None:
        r_planet = 6371.
        got_moho=False
        zmod = np.concatenate(([0], np.repeat(np.cumsum(velocity_model[:,0]),2)[:-1], [r_planet] ))
        vpmod = np.concatenate((  np.repeat(velocity_model[:,1],2), [velocity_model[-1,1]] ))
        vsmod = np.concatenate((  np.repeat(velocity_model[:,2],2), [velocity_model[-1,2]] ))
        rhomod = np.concatenate(( np.repeat(velocity_model[:,3],2), [velocity_model[-1,3]] ))
        ### Write velocity model to file .nd 
        f = open("./data/model.nd", "w")
        for zi, vpi, vsi, rhoi in zip(zmod, vpmod, vsmod, rhomod):        
            f.write("   {:.1f}  {:.2f}  {:.2f}  {:.1f}\n".format(zi, vpi, vsi, rhoi))
        f.close()

        filename = "./data/model.nd"
        vmodel = VelocityModel.read_nd_file(filename)
        build_taup_model(filename, output_folder=os.getcwd()+"/data/", verbose=False)
        model = TauPyModel(model=os.getcwd()+"/data/model.npz")
 
    # model = TauPyModel(model="ak135")
    dist_deg = dist_m/1e3 / (2.0 * r_planet * np.pi / 360.0)
    # arrival_s = model.get_travel_times(source_depth_in_km=source_depth, distance_in_degree=dist_deg, phase_list=["tts+"])
    if phase=="s":
        arrival = model.get_travel_times(source_depth_in_km=source_depth/1e3, distance_in_degree=dist_deg, phase_list=["tts+"])
    if phase=="p":
        arrival = model.get_travel_times(source_depth_in_km=source_depth/1e3, distance_in_degree=dist_deg, phase_list=["ttp+"])

    print([arr.time for arr in arrival])
    print([arr.name for arr in arrival])
    if len([arr.time for arr in arrival])==0:
        t = -9000
    else:
        t = min([arr.time for arr in arrival])
    
    # arrivals = model.get_ray_paths(source_depth_in_km=source_depth/1e3, distance_in_degree=dist_deg, phase_list=["tts+"])
    # ax = arrivals.plot_rays(plot_type="cartesian")

    return(t, model)


def model_cake_n_layers(velocity_model, qp=10000, qs=2000, radius=6371):
        
    materials = []
    last_depth = 0.
    depths = np.concatenate(([0], np.cumsum(velocity_model[:,0]) ))
    for i_layer in range(velocity_model.shape[0]):
        h, vp, vs, rho = velocity_model[i_layer,:]
        last_depth += h
        if last_depth > radius:
            last_depth = radius
        
        layer = (last_depth*1e3, cake.Material(vp=vp*1e3, vs=vs*1e3, rho=rho*1e3, qp=qp, qs=qs))
        materials.append(layer)
    
    mod = cake.LayeredModel()
    last_depth = 0.
    for i, (depth, material) in enumerate(materials):
        #print(last_depth/1e3, depth/1e3)
        layer = cake.HomogeneousLayer(ztop=last_depth, zbot=depth, m=material)#, name='fullspace')
        last_depth = depth
        mod.append(layer)
        
        # mod.append(cake.Interface(z=depth, name='{:d}'.format(i), 
        #                             mabove=mod.material(depth, direction=-4), 
        #                             mbelow=mod.material(depth, direction=4)))

    return mod


def get_arrival_cake(dist, source_depth, cake_model, re= 6371., phase='s'):
    ### Source depth [m].
    source_depth_m = source_depth #* 1e3

    ### Distances as a numpy array [deg].
    distance = dist* cake.m2d
    
    ### Define the phase to use.d
    if phase=="s":
        Phase = cake.PhaseDef('s')
        Phase_d = cake.PhaseDef('S')
    elif phase=="p":
        Phase = cake.PhaseDef('p')
        Phase_d = cake.PhaseDef('P')

    rays = cake_model.arrivals([distance], phases=Phase, zstart=source_depth_m)
    arr = [p.t for p in rays]

    rays_d = cake_model.arrivals([distance], phases=Phase_d, zstart=source_depth_m)
    arr += [p.t for p in rays_d]
    # for r in rays_d:
    #     print(r)
    # for r in rays:
    #     print(r)
        # p.given_phase()
    if len(arr)==0:
        return(-9999)
    else:
        return(min(arr))
    

def theoretical_arrival_times(dist_m, source_depth, radius=6371, file='./data/Cold_100_for_QSSP.csv'):
    layers = pd.read_csv(file, delim_whitespace=True, header=None, names=['z','vp','vs','rho','Qp','Qs'])  #skiprows=2, 
    layers = layers[:]
    h = np.diff(layers.z)
    layers = layers.iloc[1:]
    layers['h'] = h
    layers = layers.iloc[:]
    #layers.loc[layers.z==layers.z.max(), 'h'] = 0.
    velocity_model = layers.loc[:,['h','vp','vs','rho']].values

    ### frequencies (Hz) or periods (s) 
    ff = 10**np.linspace(-3,0, 100)
    T = 1 / ff[::-1]                  ### disba wants periods, low→high

    ### Rayleigh‑wave phase velocity, fundamental mode (mode 0) ---
    group_disp = GroupDispersion(*velocity_model[:66,:].T)               # unpack into h, vp, vs, ρ
    phase_disp = PhaseDispersion(*velocity_model[:66,:].T)               # unpack into h, vp, vs, ρ
    rayleigh_0 = group_disp(T, mode=0, wave="rayleigh")   # namedtuple
    rayleigh_1 = group_disp(T, mode=1, wave="rayleigh")   # namedtuple
    rayleigh_2 = group_disp(T, mode=2, wave="rayleigh")   # namedtuple
    ###
    t_r_0 = dist_m/rayleigh_0.velocity*1e-3
    t_r_1 = dist_m/rayleigh_1.velocity*1e-3
    t_r_2 = dist_m/rayleigh_2.velocity*1e-3

    cake_model = model_cake_n_layers(velocity_model, qp=10000, qs=2000, radius=radius)
    ###
    t_p = get_arrival_cake(dist_m, source_depth, cake_model, re=radius, phase='p')
    t_s = get_arrival_cake(dist_m, source_depth, cake_model, re=radius, phase='s')

    # t_p, model_obspy = get_arrival_obspy(dist_m, source_depth, velocity_model, model=None, phase="p", r_planet=radius)
    # t_s, model_obspy = get_arrival_obspy(dist_m, source_depth, velocity_model, model=None, phase="s", r_planet=radius)
    # print(t_s, t_p)

    return(t_p, t_s, [(rayleigh_0.period, t_r_0), (rayleigh_1.period, t_r_1), (rayleigh_2.period, t_r_2) ])


# =========================================================================================================
### Filtering function 
# =========================================================================================================
def butterworth_bandpass_response(f, lowcut, highcut, order):
    ### Compute the Butterworth bandpass filter response as a function of frequency.

    ### Avoid division by zero
    f = np.abs(f)
    ### Low-pass and high-pass Butterworth filters
    if f==0.0:
        h_high = 0
    else:
        h_high = 1 / np.sqrt(1 + (lowcut / f)**(2 * order))
    h_low  = 1 / np.sqrt(1 + (f / highcut)**(2 * order))
    return h_high * h_low


def butter_filter(signal, fs, lowcut, highcut, order=10, axis=0):
    if lowcut is None:
        sos = butter(order, Wn=highcut, btype='lowpass', analog=False, fs=fs, output='sos')
    elif highcut is None:
        sos = butter(order, Wn=lowcut, btype='highpass', analog=False, fs=fs, output='sos')
    else:
        sos = butter(order, Wn=[lowcut, highcut], btype='bandpass', analog=False, fs=fs, output='sos')
    return sosfilt(sos, signal, axis=axis)

# =========================================================================================================
### Wavelet spectrogram 
# =========================================================================================================
def next_power_of_2(x):
	x= int(x)
	return 1 if x == 0 else 2**(x - 1).bit_length()

def epoch_to_string(t, fmt):
	x = mdates.num2date(t)
	label = x.strftime(fmt)
	return(label)

def scientific_10(x, pos):
	if abs(x)==0:
		return r"${: 1.0f}$".format(x)
	else :
		exponent = np.floor(np.log10(abs(x)))
		coeff = x/10**exponent
		# print(x, exponent, coeff)
		if abs(coeff) ==1:
			return r"$10^{{ {:.0f} }}$".format(exponent)
		if abs(exponent)==0:
			return r"${: 2.1f} $".format(x)
		elif exponent==1:
			return r"${: 2.0f} $".format(x)
		elif exponent==-1:
			return r"${: 2.1f} $".format(x)
		# elif exponent ==2:
		# 	return r"${: 3.0f} $".format(x)
		else :
			return r"${: 3.1f} \times 10^{{ {:.0f} }}$".format(coeff,exponent)

def plot_scalogram(sig, time, ax, ax_cb=None, title_unit='', font=10, graph="pmesh",
                    fmin=None, fmax=None, cmap="magma", 
                    t_origin = None, **kwargs):	
    
    dt = np.diff(time)[0]

    pad = int(next_power_of_2(sig.size)-sig.size)     # Pads the time series with zeroes (recommended)
    dj = 1/16                                                   # Uses 1/dj sub-octaves per octave
    s0 = 2*dt                                                   # Start at a scale of 2*dt 
    fdisp_min = 5e-4                                            # Minimum frequency to display
    noct = int(np.log((1/s0)/fdisp_min)/np.log(2))              # Estimates the number of octaves from min frequency
    J = noct / dj                                               # Number of octave / voices = total number of scales 
    #print(noct, pad)
    mother = 'MORLET'


    #############################################
    sig = signal.detrend(sig, type="linear")
    sig *= signal.windows.tukey(sig.size, alpha=0.05)
    if fmin is not None and fmax is not None:
        butter_filter(sig, 1/dt, fmin, fmax)
    else:
        fmax = 1/(2*dt)

    
    #############################################
    # wave, period, scale, coi = wavelet(tr.data, dt, pad=pad, dj=dj, s0=s0, J1=J, mother=mother)  ### Web version
    # f = 1/period
    np.int = int
    wave, scale, f, coi, fft, fftfreqs = wavelet.cwt(sig, dt, dj, s0, J,  wavelet.Morlet())   ### Pycwt version
    p = np.abs(wave)**2  # compute wavelet power spectrum
    # p /= scale[:, None]

    ### Select data within fmin, fmax 
    #bol = np.array((f > fmin, f < fmax)).all(axis=0)
    bol = np.array((f > 0, f<1e6)).all(axis=0)
    ### Select frequency axis 
    fr = f[bol]
    
    #############################################
    ### Set the DB scale
    col = 10 * np.log10(p[bol, :])
    if "dbrange" in kwargs:
        if kwargs["dbrange"][0] == "min": 
            dBmin = np.nanmin(col)
        else:
            dBmin = kwargs["dbrange"][0]
        ###
        if kwargs["dbrange"][1] == "max": 
            dBmax = np.nanmax(col)		
        else:
            dBmax = kwargs["dbrange"][1]
        # print(np.nanmin(col),np.nanmax(col))	
    else :
        #dBmin, dBmax = np.nanmin(col)/1.5, np.nanmax(col)
        # wrf = np.where((fr>5e-2) & (fr<2e-1))
        # dBmin = np.mean(col[wrf]) - 1*np.std(col[wrf])
        dBmin = np.mean(col) - 1*np.std(col)
        dBmax = col.max()
    
    #print(dBmin, dBmax)

    ############################################
    if graph=="pmesh":
        ### Option 1: pcolormesh
        ct = ax.pcolormesh(time, fr, col,rasterized=True,
                        vmin= dBmin, vmax=dBmax, cmap=cmap,shading='auto')
        fpmin = fr.min()
    elif graph=="cont":
        ### Option 2: contourf
        lev = np.linspace(dBmin, dBmax,40)
        lev = (lev*10)//10
        ct = ax.contourf(time,fr,col, levels=lev,
                    cmap=cmap, extend="both")#,vmin= dBmin, vmax=dBmax)#, extend="both")
        fpmin = fr.min()
        for c in ct.collections:
            c.set_rasterized(True)


    #################################################################
    ### Main axis decorations 
    ax.set_yscale('log')
    ax.set_ylabel('Frequency / $Hz$', fontsize=font)
    ax.set_ylim([fpmin, fmax])
    ax.set_xlim(time.min(), time.max())
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
    ax.tick_params(axis='both', which='both', labelsize=font-2)

    #################################################################
    ### Region of significance 
    coi[np.where(coi==0)] = 1e-15
    ### Wavelets
    # ax.fill_between(tc, 1/(coi * 0 + 1e8), [max(1/coii,1/period[-1]) for coii in coi ], 
    # 					facecolor="k", edgecolor="none", alpha=0.4)
    #print(coi, 1/coi)
    #ax.plot(tc, 2/coi, 'k')

    ### COLORBAR ####################################################
    if ax_cb is not None:
        cb = plt.colorbar(mappable=ct, cax=ax_cb)
        ax_cb.set_ylabel(r'PSD / $dB$$\cdot$$Hz^{-1}$', fontsize=font)
        ax_cb.tick_params(axis='both', which='both', labelsize=font-2)
        ax_cb.yaxis.set_label_position('left')
        ax_cb.set_title(title_unit ,fontsize=font)
    
    return(ax, ax_cb)


# =========================================================================================================
### Function to convert between different Radiance units and Rayleigh
# =========================================================================================================
def factor_W_to_Rayleigh(L, bandwidth = 0.03, dir="specRadiance_to_Rayleigh"):
    ### bandwidth should be in micrometer if Radiance is in W/m2/micrometer/sr
    ### Note: In B Kenda's thesis, its 4pi.(Ep)^-1 and not (4pi.Ep⁻1) that is correct.
    ### Correct definition of the Rayleigh: 1R = (1/4pi) * 1e10 photons/s/cm2/steradians 
    hp = 6.6260701e-34  ### Planck's constant, J. s
    c = 299792458       ### Speed of light, m/s
    Ep = hp*c/(L*1e-6)    #### Energy of photon 

    if dir=="specRadiance_to_Rayleigh":
        ### Radiance is initially in W/m2/micrometer/sr 
        factor = bandwidth/Ep*4*np.pi*1e-10 
        return(factor)
    elif dir=="Radiance_to_Rayleigh":
        ### Radiance is initially in W/m2/sr 
        factor = 1/Ep*4*np.pi*1e-10 
        return(factor)
    elif dir=="phRadiance_to_Rayleigh":
        ### Radiance is initially in ph/s/m2/sr 
        factor = 4*np.pi*1e-10 
        return(factor)
    elif dir=="Rayleigh_to_Radiance":
        ### Radiance will be in W/m2/sr 
        factor = 1/Ep*4*np.pi*1e-10 
        return(1/factor)
    elif dir=="Rayleigh_to_specRadiance":
        ### Radiance will be in W/m2/micrometer/sr 
        factor = bandwidth/Ep*4*np.pi*1e-10 
        return(1/factor)
    else: 
        raise("Wrong conversion specified.")


# =========================================================================================================
### Function to define global parameters for parallelisation 
# =========================================================================================================
def init_worker_nightlow(_list_of_locations, 
                        #  _fft_vzs, _fft_uzs, 
                         _att_exp, _amplification, _phase_shift_z,
                        _f_VER_1_27,_f_dVER_1_27, _z_1_27_calc_m, _fourier_filtering, _Nt, _fs, 
                        _b, _a, _loc_save, _itime_save, _save_ver, _gridded, _tf_phase_nightglow, _dir_save):
                        # _dphase_att_ampl_dz, 
#                        _zatt, _zrange):
    # global fft_vzs, fft_uzs
    global att_exp, amplification, phase_shift_z
    global f_VER_1_27, f_dVER_1_27, z_1_27_calc_m, fourier_filtering, tf_phase_nightglow, Nt, fs
    global b, a
    global list_of_locations, loc_save, itime_save, gridded, dir_save, save_ver
    # global dphase_att_ampl_dz, 
    # global zatt, zrange

    list_of_locations = _list_of_locations
    # fft_vzs = _fft_vzs
    # fft_uzs = _fft_uzs
    att_exp = _att_exp
    amplification = _amplification
    phase_shift_z = _phase_shift_z
    f_VER_1_27 = _f_VER_1_27
    f_dVER_1_27 = _f_dVER_1_27
    z_1_27_calc_m = _z_1_27_calc_m
    fourier_filtering = _fourier_filtering
    Nt = _Nt
    fs = _fs
    b = _b
    a = _a
    loc_save = _loc_save
    itime_save = _itime_save
    save_ver = _save_ver
    gridded = _gridded
    tf_phase_nightglow = _tf_phase_nightglow
    dir_save = _dir_save
    # dphase_att_ampl_dz = _dphase_att_ampl_dz
    # zatt = _zatt 
    # zrange = _zrange


def init_worker_dayglow(_list_of_locations, _fft_uzs, _att_exp, _amplification, _phase_shift_z, _Nt,
                _z_4_28_calc_m, _fac_temperature, _f_VER_4_28, _f_dVER_4_28, _loc_save, _itime_save, _save_ver, _gridded, _dir_save):
    global fft_uzs, att_exp, amplification, phase_shift_z, Nt
    global z_4_28_calc_m, fac_temperature, f_VER_4_28, f_dVER_4_28
    global list_of_locations, loc_save, itime_save, gridded, dir_save, save_ver

    list_of_locations = _list_of_locations
    fft_uzs = _fft_uzs
    att_exp = _att_exp
    amplification = _amplification
    phase_shift_z = _phase_shift_z
    Nt = _Nt
    z_4_28_calc_m = _z_4_28_calc_m
    f_VER_4_28 = _f_VER_4_28
    f_dVER_4_28 = _f_dVER_4_28
    fac_temperature = _fac_temperature
    loc_save = _loc_save
    itime_save = _itime_save
    save_ver = _save_ver
    gridded = _gridded
    dir_save = _dir_save 


# =========================================================================================================
### Function to propagate a group seismograms up in the atmosphere (simple model, only vertical vel coupling)
def propagate_attenuate(fft, i_east, i_north, att, ampl, psz):
    ### Apply attenuation and amplification at all z 
    ### shape attenuation, phase_shift: (Nz, Nw). Shape velocity, fft: (Ne, Nn, Nz, Nw)
    ### To properly account for the displacement (or velocity) steps, all time traces were zero-padded in time beforehand

    att_vz = fft[i_east, i_north, np.newaxis, :] * att
    ampl_vz_z = att_vz * ampl[:, np.newaxis]

    ### Delay at altitude z 
    fft_vz_z = ampl_vz_z *psz

    ### Back to time domain: inverse FFT to get vz_z for these altitudes
    vz_z = np.real(sfft.ifft(fft_vz_z, axis=1))  # shape: (2, Nt)


    return(vz_z, fft_vz_z)


# =========================================================================================================
### Function to transform velocity at a specific altitude into VER perturbation 
def velocity_to_dVER_nightglow(uz_z, vz_z, fft_vz_z, z_1_27_calc_m, f_VER_1_27, f_dVER_1_27, b, a, tf_phase_nightglow=None, fourier_filtering=False, test=False):

    fver_alt = f_VER_1_27(z_1_27_calc_m)[:,np.newaxis]
    fdver_alt = f_dVER_1_27(z_1_27_calc_m)[:,np.newaxis]

    ### Filter at all altitudes
    ### OPTION 1: USE A TIME-DOMAIN FILTER  
    if not fourier_filtering:
        ### Compute VER and its vertical gradient (numpy gradient) TIME DOMAIN 

        ### VERSION WITH SMOOTH GRADIENT -- WARNING: THIS VERSION YIELDS A NET ZERO INTENSITY WHEN USING ONLY VERTICAL GRADIENTS 
        # dver_vz_z = fver_alt * np.gradient(vz_z, z_1_27_calc_m, axis=0) + fdver_alt*vz_z
        ### VERSION OF PL -- WARNING: THIS VERSION YIELDS A NET ZERO INTENSITY WHEN USING ONLY VERTICAL GRADIENTS 
        # ver_vz = fver_alt * vz_z  # shape: (Nz, Nt)
        # dver_vz_z = np.gradient(ver_vz, z_1_27_calc_m, axis=0)
        ### Adding the advection term 
        # dver_z = lfilter(b, a, dver_vz_z, axis=1) 
        # dver_z += uz_z*fdver_alt

        ### VERSION OF BK 
        dver_vz_z = fver_alt * np.gradient(vz_z, z_1_27_calc_m, axis=0)
        ### VERSION OF BK, transformed by IPP  
        ### Once integrated, it yields the same result, but the ver itself is inexact. 
        # dver_vz_z = -fdver_alt * fft_vz_z
        dver_z = lfilter(b, a, dver_vz_z, axis=1)

    ### OPTION 1: USE A FREQUENCY-DOMAIN FILTER  
    else:
        ### Compute VER and its vertical gradient (numpy gradient) FOURIER DOMAIN 
        
        ### SMOOTH GRADIENT: WARNING: THIS VERSION YIELDS A NET ZERO INTENSITY WHEN USING ONLY VERTICAL GRADIENTS  
        # dver_vz_z = fver_alt * np.gradient(fft_vz_z, z_1_27_calc_m, axis=0) + fdver_alt*fft_vz_z       
        ### VERSION OF PL: WARNING: THIS VERSION YIELDS A NET ZERO INTENSITY WHEN USING ONLY VERTICAL GRADIENTS 
        # ver_vz = fver_alt * fft_vz_z  # shape: (Nz, Nt) 
        # dver_vz_z = np.gradient(ver_vz, z_1_27_calc_m, axis=0)
        ### Adding the advection term : 
        # dver_z = sfft.ifft(tf_phase_nightglow * dver_vz_z, axis=1).real  
        # dver_z +=  uz_z*fdver_alt

        ### VERSION OF BK 
        dver_vz_z = fver_alt * np.gradient(fft_vz_z, z_1_27_calc_m, axis=0)
        ### VERSION OF BK, transformed by IPP  
        ### Once integrated, it yields the same result, but the ver itself is inexact. 
        # dver_vz_z = -fdver_alt * fft_vz_z
        dver_z = sfft.ifft(tf_phase_nightglow * dver_vz_z, axis=1).real

        ### Simple linear detrend 
        ### dver_z = signal.detrend(dver_z, type="linear", axis=1) ### way too slow 
        start = dver_z[:,0][:,None]
        end   = dver_z[:,50][:,None]
        trend = np.linspace(0, 1, dver_z.shape[1])   
        trend = start + (end - start)/(trend[50]-trend[0]) * trend  # shape (Nz, Nt)
        dver_z = dver_z - trend
    
    if test:
        ### TODO: REMOVE 
        ver_dvz_z = fver_alt * np.gradient(vz_z, z_1_27_calc_m, axis=0)
        dver_z2 = lfilter(b, a, ver_dvz_z, axis=1)

        gs_kw = dict(width_ratios=[1,0.5])
        figb, (ax3b,ax3c) = plt.subplots(ncols=2, nrows=1, constrained_layout=True, gridspec_kw=gs_kw)

        fig = plt.figure(figsize=(10, 10))
        # Create GridSpec with 2 rows and 3 columns
        gs = gridspec.GridSpec(4, 3, height_ratios=[2,4, 4, 4], hspace=0.4, wspace=0.3)
        # Top 3 small plots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        # Bottom 2 long plots spanning the width of the top 3
        ax4 = fig.add_subplot(gs[1, :])
        ax5 = fig.add_subplot(gs[2, :])
        ax6 = fig.add_subplot(gs[3, :])
        

        ax1.plot(np.max(np.abs(np.gradient(vz_z, z_1_27_calc_m, axis=0)), axis=1), z_1_27_calc_m/1e3, c="k")
        ax1.set_title("Max div of Vz")
        ax2.plot(fver_alt, z_1_27_calc_m/1e3, c="k")
        ax2b = ax2.twiny()
        ax2b.plot(np.gradient(fver_alt, z_1_27_calc_m, axis=0), z_1_27_calc_m/1e3, ls="--", color="grey")
        ax2.set_title("VER")
        ### Plot dVER/VER and dvz/vz 
        ax3.plot(np.max(np.gradient(fver_alt, z_1_27_calc_m, axis=0)/fver_alt, axis=1), z_1_27_calc_m/1e3, c="k", label="div(VER)/VER")
        ax3.plot(np.max(np.gradient(vz_z, z_1_27_calc_m, axis=0)/vz_z, axis=1), z_1_27_calc_m/1e3, c="k", ls="--", label="div(v)/v")
        print(np.median(  np.abs((np.gradient(vz_z, z_1_27_calc_m, axis=0)/vz_z) / (np.gradient(fver_alt, z_1_27_calc_m, axis=0)/fver_alt)), axis=1) )
        ax3.set_title("Max div of Vz*VER")
        ax3.legend()


        grada = np.gradient(vz_z, z_1_27_calc_m, axis=0)/vz_z 
        gradb = np.gradient(fver_alt, z_1_27_calc_m, axis=0)/fver_alt
        ratio  = abs(grada/gradb)
        for i in range(0,vz_z.shape[1],10):
            ax3b.plot(ratio[:,i],z_1_27_calc_m/1e3, c="tab:blue", alpha=0.1, lw=0.1)
        ax3b.set_xscale("log")
        ax3b.set_xlim(0.1,1e3)
        ax3b.axvline(10, ls="--", c="grey", label="(dVz/vz) >> (dVER/VER)")
        ax3b.legend(loc=1)
        ax3b.plot(np.median(ratio, axis=1), z_1_27_calc_m/1e3, c="k")
        ax3b.set_ylabel("Altitude / [$km$]")
        ax3b.set_xlabel("(dVz/vz)/(dVER/VER) over time")
        ax3c.plot(fver_alt, z_1_27_calc_m/1e3, c="k")
        ax3cb = ax3c.twiny()
        ax3cb.plot(np.gradient(fver_alt, z_1_27_calc_m, axis=0), z_1_27_calc_m/1e3, ls="--", color="grey")
        ax3c.set_xlabel("VER")
        ax3cb.set_xlabel("div(VER)")
        ax3c.set_xlim(-2.5e11,5e11)
        ax3cb.set_xlim(-0.6e8,1.2e8)

        ### 
        i=39
        dz_km = np.diff(z_1_27_calc_m/1e3)[0]
        amp_airglow = integrate.cumulative_trapezoid(dver_z, x=z_1_27_calc_m, axis=0)
        amp_airglow2 = integrate.cumulative_trapezoid(dver_z2, x=z_1_27_calc_m, axis=0)

        # np.save("./results/Resolution_integral_interp2", amp_airglow2[-1,:])
        # np.save("./results/Resolution_integral_interp", amp_airglow[-1,:])

        np.save("./results/Resolution_integral_gauss2", amp_airglow2[-1,:])
        np.save("./results/Resolution_integral_gauss", amp_airglow[-1,:])

        ampgauss = np.load("./results/Resolution_integral_gauss.npy")
        ampgauss2 = np.load("./results/Resolution_integral_gauss2.npy")
        ampinterp = np.load("./results/Resolution_integral_interp.npy")
        ampinterp2 = np.load("./results/Resolution_integral_interp2.npy")

        fig, ax = plt.subplots()
        ax.plot(ampgauss, c="k", ls="-", label="div(VER*vz), gaussian fit")
        ax.plot(ampinterp, c="k", ls="--", label="div(VER*vz), interpolated VER")
        ax.plot(ampgauss2, c="purple", ls="-", label="VER*div(vz), gaussian fit")
        ax.plot(ampinterp2, c="purple", ls="--", label="VER*div(vz), interpolated VER")
        ax.legend()
        ax.set_xlabel("Time iteration")
        ax.set_ylabel("Integrated Intensity")

        Nz = int(4*dver_vz_z.shape[0]/40)
        for i in range(0,dver_vz_z.shape[0],Nz):
            ax4.plot(dver_vz_z[i,:]/np.max(dver_vz_z[i,:])*dz_km*Nz/2 + z_1_27_calc_m[i]/1e3, c="k")#, label="div(VER*v) at altitude 10")
            ax4.plot(ver_dvz_z[i,:]/np.max(dver_vz_z[i,:])*dz_km*Nz/2 + z_1_27_calc_m[i]/1e3, c="r", ls="--")#,  label="VER*div(v) at altitude 10")
            ### 
            ax5.plot(dver_z[i,:]/np.max(dver_z[i,:])*dz_km*Nz/2 + z_1_27_calc_m[i]/1e3, c="k")#, label="Filtered div(VER*v) at altitude 10")
            ax5.plot(dver_z2[i,:]/np.max(dver_z[i,:])*dz_km*Nz/2 + z_1_27_calc_m[i]/1e3, c="r", ls="--")#,  label="Filtered VER*div(v) at altitude 10")
            ### Progressive integral 
            ax6.plot(amp_airglow[i,:]/np.max(amp_airglow[i,:])*dz_km*Nz/2 + z_1_27_calc_m[i]/1e3, c="k")#, label="Filtered div(VER*v) at altitude 10")
            ax6.plot(amp_airglow2[i,:]/np.max(amp_airglow[i,:])*dz_km*Nz/2 + z_1_27_calc_m[i]/1e3, c="r", ls="--")#,  label="Filtered VER*div(v) at altitude 10")    
        
        ax6.plot(amp_airglow[-1,:]/np.max(amp_airglow[-1,:])*dz_km*Nz/2 + z_1_27_calc_m[-1/1e3], c="k")#, label="Filtered div(VER*v) at altitude 10")
        ax6.plot(amp_airglow2[-1,:]/np.max(amp_airglow[-1,:])*dz_km*Nz/2 + z_1_27_calc_m[-1/1e3], c="r", ls="--")#,  label="Filtered VER*div(v) at altitude 10")    
        ###
        ax4.legend()
        ax5.legend()
        ax1.set_ylabel("Altitude / km")

        fig2 = plt.figure(figsize=(10, 5))
        # Create GridSpec with 2 rows and 3 columns
        gs2 = gridspec.GridSpec(1, 3)
        # Top 3 small plots
        ax0 = fig2.add_subplot(gs2[0, 0])
        ax1 = fig2.add_subplot(gs2[0, 1])
        ax2 = fig2.add_subplot(gs2[0, 2])
        it = 2400
        print(dver_vz_z[:,it].min(),dver_vz_z[:,it].max() )
        ###
        ax0.pcolormesh(np.array([0,1]),z_1_27_calc_m/1e3, np.tile(vz_z[:,it],(2,1)).T, cmap="seismic", vmin=-vz_z[:,it].max(), vmax=vz_z[:,it].max())
        ax0.set_title("vz")
        ###
        ax1.pcolormesh(np.array([0,1]),z_1_27_calc_m/1e3, np.tile(dver_vz_z[:,it],(2,1)).T, cmap="seismic", vmin=-1e6, vmax=1e6)
        ax1.set_title("div(VER*vz)")
        ###
        ax2.pcolormesh(np.array([0,1]),z_1_27_calc_m/1e3, np.tile(ver_dvz_z[:,it],(2,1)).T, cmap="seismic", vmin=-1e6, vmax=1e6)
        ax2.set_title("VER*div(vz)")
        quit()

    return(dver_z)


# =========================================================================================================
### Function to transform displacement at a specific altitude into VER perturbation 
def temperature_perturbation_dayglow(uz_z, z_4_28_calc_m, fac_temperature):

    ### Divergence of U * temperature factor 
    dver_z = fac_temperature[:,None] * np.gradient(uz_z, z_4_28_calc_m, axis=0) 
    
    return(dver_z)


# =========================================================================================================
### Function to integrate over line of sight (simple model, vertical LOS)
def integrate_line_of_sight(dver_z, z_calc_m, wavelength):
    ### For now, the LOS is a simple vertical line 
    #amp_dayglow = np.trapz((dVER_ad+1*dVER_tr), x=alts_dayglow, axis=1)/np.trapz(f_VER_dayglow(alts_dayglow), x=alts_dayglow,)

    ### Luminosity of airglow perturbation 
    ### dver_z shape (Nz,Nt)
    amp_airglow = np.trapz(dver_z, x=z_calc_m, axis=0) # /np.trapz(f_VER(alts_dayglow), x=alts_dayglow,)

    ### Convert to Rayleigh
    #amp_airglow *= factor_W_to_Rayleigh(wavelength, dir="Radiance_to_Rayleigh")  ### if VER was in W/m3
    amp_airglow *= factor_W_to_Rayleigh(wavelength, dir="phRadiance_to_Rayleigh")  ### if VER was in ph/s/m3

    return(amp_airglow)


# =========================================================================================================
### Wrapper function for calculating NIGHTglow at one location (outside of class to be parallelisable)
# =========================================================================================================
def nightglow_at_location(i_en, list_of_locations, fft_vzs, fft_uzs, att_exp, amplification, phase_shift_z,f_VER_1_27, f_dVER_1_27,
                        z_1_27_calc_m, fourier_filtering, Nt, fs, b,a, loc_save, itime_save, save_ver, gridded, tf_phase_nightglow, dir_save,
                        # dphase_att_ampl_dz, zatt, zrange):
                        ):
    
    i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]

    # if (i_east, i_north) == loc_save[0]:
    #     print(i_en, i_east, i_north)
    
    vz_z, fft_vz_z = propagate_attenuate(fft_vzs, i_east, i_north, att_exp, amplification, phase_shift_z)
    uz_z, fft_uz_z = propagate_attenuate(fft_uzs, i_east, i_north, att_exp, amplification, phase_shift_z)
    ###
    dver_z = velocity_to_dVER_nightglow(uz_z, vz_z, fft_vz_z, z_1_27_calc_m, f_VER_1_27, f_dVER_1_27, b, a, 
                                        tf_phase_nightglow=tf_phase_nightglow, fourier_filtering= fourier_filtering)
    ### Remove padding
    dver_z = dver_z[:,:Nt]
    vz_z = vz_z[:,:Nt]
    ### Ensure signal starts at 0 
    dver_z -= dver_z[:,0][:,None]

    ### Filter dver below 5e-3 Hz 
    # dver_z = butter_filter(dver_z, fs, 5e-3, None, axis=1, order=10)
    
    amp_nightglow = integrate_line_of_sight(dver_z, z_1_27_calc_m, 1.27)
    ### Attempt: amplitude with a high resolution vertical integral     
    # fft_dver = fft_vzs[i_east, i_north, None,:]*dphase_att_ampl_dz*f_VER_1_27(zatt)[:,None]
    # fft_dver = fft_dver[-zrange.size:,:]
    # ###
    # I = integrate_line_of_sight(fft_dver*tf_phase_nightglow, zrange, 1.27)
    # I = np.fft.ifft(I).real
    # ###
    # start = I[0]
    # end   = I[50]
    # trend = np.linspace(0, 1, I.size)   
    # trend = start + (end - start)/(trend[50]-trend[0]) * trend  # shape (Nz, Nt)
    # amp_nightglow = I - trend

            
    ### Store wavefield info 
    # if gridded:
    #     save_wavefield[i_east, i_north, :,:,0] = vz_z[:,itime_save]    ### Save velocity waveform 
    #     save_wavefield[i_east, i_north, :,:,1] = dver_z[:,itime_save]  ### Save dVER at altitude z 
    #     save_intensity_dver[i_east, i_north,:] = amp_nightglow[itime_save]  ### Save dVER at altitude z 

    ### For specific location, save all times and altitudes.
    ### NOTE: Otherwise very heavy and very slow 
    if (i_east, i_north) in loc_save:
        np.save(dir_save + "nightglow_dver_z_{:d}_{:d}".format(i_east, i_north), dver_z)
        np.save(dir_save + "nightglow_vz_z_{:d}_{:d}".format(i_east, i_north), vz_z)
        np.save(dir_save + "nightglow_I_{:d}_{:d}".format(i_east, i_north), amp_nightglow)

    if save_ver:
        return(vz_z[:,itime_save], dver_z[:,itime_save], amp_nightglow[itime_save])
    else: 
        return((amp_nightglow[itime_save],))



# =========================================================================================================
### Wrapper function for calculating DAYglow at one location (outside of class to be parallelisable)
# =========================================================================================================
def dayglow_at_location(i_en, list_of_locations, fft_uzs, att_exp, amplification, phase_shift_z, Nt,
                        z_4_28_calc_m, fac_temperature, f_VER_4_28, f_dVER_4_28, loc_save, itime_save, save_ver, gridded, dir_save):
    
    i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]

    # if (i_east, i_north) == loc_save[0]:
    #     print(i_en, i_east, i_north)
    
    uz_z, fft_uz_z = propagate_attenuate(fft_uzs, i_east, i_north, att_exp, amplification, phase_shift_z)
    ### Remove padding
    uz_z = uz_z[:,:Nt]
    ### Ensure signal starts at 0 
    uz_z -= uz_z[:,0][:,None]
    ###
    dver_z = temperature_perturbation_dayglow(uz_z, z_4_28_calc_m, fac_temperature)
    ### Add advection of VER term (BK):
    ad_ver = uz_z * f_dVER_4_28(z_4_28_calc_m)[:,None]
    dver_z += ad_ver
    ### 
    amp_dayglow = integrate_line_of_sight(dver_z, z_4_28_calc_m, 4.28)
            
    ### Store wavefield info 
    # if gridded:
    #     save_wavefield[i_east, i_north, :,:,0] = vz_z[:,itime_save]    ### Save velocity waveform 
    #     save_wavefield[i_east, i_north, :,:,1] = dver_z[:,itime_save]  ### Save dVER at altitude z 
    #     save_intensity_dver[i_east, i_north,:] = amp_dayglow[itime_save]  ### Save dVER at altitude z 

    ### For specific location, save all times and altitudes.
    ### NOTE: Otherwise very heavy and very slow 
    if (i_east, i_north) in loc_save:
        np.save(dir_save + "dayglow_dver_z_{:d}_{:d}".format(i_east, i_north), dver_z)
        np.save(dir_save + "dayglow_uz_z_{:d}_{:d}".format(i_east, i_north), uz_z)
        np.save(dir_save + "dayglow_I_{:d}_{:d}".format(i_east, i_north), amp_dayglow)

    if save_ver:
        return(uz_z[:,itime_save], dver_z[:,itime_save], amp_dayglow[itime_save])
    else: 
        return((amp_dayglow[itime_save],))
    


def worker_func_nightglow(i):
    # location = list_of_locations[i]
    return nightglow_at_location(i, list_of_locations, fft_vzs, fft_uzs, att_exp, amplification, 
                               phase_shift_z, f_VER_1_27, f_dVER_1_27, z_1_27_calc_m, 
                               fourier_filtering, Nt, fs, b, a, loc_save, itime_save, save_ver, gridded, tf_phase_nightglow, dir_save)#,
                            #    dphase_att_ampl_dz, zatt, zrange)


def worker_func_dayglow(i):
    # location = list_of_locations[i]
    return dayglow_at_location(i, list_of_locations, fft_uzs, att_exp, amplification, 
                               phase_shift_z, Nt, z_4_28_calc_m, fac_temperature, f_VER_4_28, f_dVER_4_28,
                               loc_save, itime_save, save_ver, gridded, dir_save)


# =========================================================================================================
class AirglowSignal:
# =========================================================================================================

    def __init__(self, SEISMO, Nz = 50, do_plot=False, disable_att = False):
        """
        Initialize the AirglowSignal model.

        Parameters:
        - The Pre-calculated seismograms and everything from Seismograms class
        - Parameters of the precision in altitude (Nz) for the airglow integrations 
        """

        ### SOME CONSTANTS 
        self.hp = 6.6260701e-34  ### Planck's constant, J. s
        self.c = 299792458       ### Speed of light, m/s
        self.unit = "photons"    ### Choose photons/s or Watts for VER definition. 

        ### READ WAVEFORM PARAMETERS 
        # self.__dict__.update(SEISMO.__dict__)
        if hasattr(SEISMO, '__dict__'):
            self.__dict__.update(SEISMO.__dict__)
        elif isinstance(SEISMO, dict):
            self.__dict__.update(SEISMO)

        ### READ ATMOSPHERIC DATA. 
        folder_data            = fold + 'data/'
        file_atmos             = f'{folder_data}profile_VCD_for_scaling_pd.csv'
        file_1_27_airglow      = f'{folder_data}VER_profile_scaled.csv'
        file_4_28_airglow      = f'{folder_data}VER_profile_dayglow.csv'
        file_airglow_kenda     = f'{folder_data}VER_profiles_from_kenda.csv'
        file_attenuation_kenda = f'{folder_data}attenuation_kenda.csv'
        dir_attenuation_GA     = '/staff/marouchka/Documents/ATMOSPHERE/ATTENUATION/Gil_Averbuch_profiles/'
        use_kenda_atm          = False   ### To use or not the atmospheric (T, P, rho, c) of Balthasar Kenda
        use_kenda_att          = True    ### True: uses BK's attenuation, False: uses GA's attenuation 
        
        ### CONSTRUCT INTERPOLATED ATMOSPHERIC MODELS 
        self._load_atmosphere(file_atmos, file_1_27_airglow, file_4_28_airglow, file_airglow_kenda, use_kenda_atm=use_kenda_atm, do_plot=do_plot)

        ### CONSTRUCT INTERPOLATED ABSPORTION / AMPLIFICATION MODELS 
        if use_kenda_att:
            self.f_alpha, self.f_alpha_2d, self.f_amplification = self._load_absorption_amplification(file_attenuation_kenda=file_attenuation_kenda, do_plot=do_plot, disable_att = disable_att)
        else:
            self.f_alpha, self.f_alpha_2d, self.f_amplification = self._load_absorption_amplification(dir_attenuation_GA=dir_attenuation_GA, do_plot=do_plot, disable_att = disable_att)

        ### Definitions for the calculation of 1_27 micrometer nightglow 
        self.tau = 4460  # s ### VERY IMPORTANT: decay time of excited oxygen 
        self.b, self.a = self._def_filter_nightglow()
        self.Nz = Nz    # Number of altitude points for gradients and integrations. 
        # self.z_1_27_calc_m = np.linspace(self.z_1_27_min, self.z_1_27_max, self.Nz)  # in meters, always 
        self.z_1_27_calc_m = np.linspace(80e3, 140e3, self.Nz)  # in meters, always 
        self.z_1_27_calc_km = self.z_1_27_calc_m / 1e3        
        self.dz_1_27_m = np.diff(self.z_1_27_calc_m)[0]
        ### For calculation of cumulated attenuation: 
        self.z_att_1_27_m = np.concatenate((np.arange(self.z_1_27_min,0, -self.dz_1_27_m)[1:][::-1], self.z_1_27_calc_m)) 
        self.I_background_nightglow = integrate_line_of_sight(self.f_VER_1_27(self.z_1_27_calc_m), self.z_1_27_calc_m, 1.27)

        ### Definitions for the calculation of 4_28 micrometer nightglow         
        self.alpha_t = 0.01    ### VERY IMPORTANT: 1% sensitivity to temperature variations 
        # self.z_4_28_calc_m = np.linspace(self.z_4_28_min, self.z_4_28_max, self.Nz)  # in meters, always
        self.z_4_28_calc_m = np.linspace(110e3, 160e3, self.Nz)  # in meters, always  
        # self.z_4_28_calc_m = np.linspace(90e3, 160e3, self.Nz)  # in meters, always  
        self.z_4_28_calc_km = self.z_4_28_calc_m / 1e3        
        self.dz_4_28_m = np.diff(self.z_4_28_calc_m)[0]
        ### For calculation of cumulated attenuation: 
        self.z_att_4_28_m = np.concatenate((np.arange(self.z_4_28_min,0, -self.dz_4_28_m)[1:][::-1], self.z_4_28_calc_m)) 
        self.I_background_dayglow = integrate_line_of_sight(self.f_VER_4_28(self.z_4_28_calc_m), self.z_4_28_calc_m, 4.28)


    def _factor_photons_watt(self, L, dir="ps_to_W"):
        Ep = self.hp*self.c/(L*1e-6)    #### Energy of photon at wavelength L
        
        if dir=="ps_to_W": 
            return(Ep)
        elif dir=="W_to_ps":
            return(1/Ep)
        else: 
            raise("Wrong conversion specified.")
    

    def _load_atmosphere(self, file_atmos, file_1_27_airglow, file_4_28_airglow, file_airglow_kenda,
                         use_kenda_atm=False, do_plot=True):

        atm_profile = pd.read_csv(file_atmos)
        self.f_rho = interpolate.interp1d(atm_profile.altitude, atm_profile.rho, kind='quadratic', bounds_error=False, fill_value=(atm_profile.rho.min(), atm_profile.rho.max()))
        self.f_t = interpolate.interp1d(atm_profile.altitude, atm_profile.t, kind='quadratic')
        self.f_gamma = interpolate.interp1d(atm_profile.altitude, atm_profile.gamma, kind='quadratic')
        self.f_c = interpolate.interp1d(atm_profile.altitude, atm_profile.c, kind='quadratic')

        ### =========================================================================================
        ### Read data for 1_27 micrometer Oxygen volume emission rate
        ### Units of VER: Photons/m3/s
        VER = pd.read_csv(file_1_27_airglow)
        VER.columns=['VER', 'alt']   
        ### CHOOSE VER UNIT   
        if self.unit=="watt":
            ### TEST: Convert VER to W/m3 using the energy of a photon 
            ### Ep = hc/lambda 
            ### Lambda = 1.27 micrometer 
            VER.VER *= self._factor_photons_watt(1.27, dir="ps_to_W")
        ### CONVERT ALTITUDES TO METER 
        VER.alt *= 1e3  
        self.z_1_27_min = VER.alt.min()
        self.z_1_27_max = VER.alt.max()

        ### EITHER FIT THE VER WITH SMOOTH FUNCTION OR INTERPOLATE DATA POINTS 
        do_fit=True#False
        if do_fit:
            ### FIT BY TWO GAUSSIANS
            def double_positive_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
                return (np.abs(a1) * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) +\
                        np.abs(a2) * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)))
            def double_positive_gaussian_deriv(x, a1, mu1, sigma1, a2, mu2, sigma2):
                return (-np.abs(a1) *(x-mu1)/sigma1**2* np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) +\
                        -np.abs(a2) *(x-mu2)/sigma2**2* np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)))
            ### FIT BY ASYMMETRIC GAUSSIAN 
            def asymmetric_gaussian_pulse(x, a, mu, sigma_left, sigma_right):
                sigma = np.where(x < mu, sigma_left, sigma_right)
                return np.abs(a) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            def asymmetric_gaussian_pulse_deriv(x, a, mu, sigma_left, sigma_right):
                sigma = np.where(x < mu, sigma_left, sigma_right)
                return -np.abs(a) * (x-mu)/sigma**2 * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

            initial_guess1 = [VER.VER.max(), 100e3, 10e3, VER.VER.max()/50, 110e3, 10e3]
            popt1_127, pcov1 = curve_fit(double_positive_gaussian, VER.alt, VER.VER, p0=initial_guess1)
            initial_guess2 = [VER.VER.max(), 100e3, 10e3, 11e3]
            popt2_127, pcov2 = curve_fit(asymmetric_gaussian_pulse, VER.alt, VER.VER, p0=initial_guess2)

            ### TEST: USE AN INTERPOLATED PULSE  
            ### NOTE: ENSURE GRADIENT IS IN SI UNITS !!!
            self.f_VER_1_27 = lambda x: asymmetric_gaussian_pulse(x, *popt2_127)
            self.f_dVER_1_27 = lambda x: asymmetric_gaussian_pulse_deriv(x, *popt2_127)  
            # self.f_VER_1_27 = lambda x: double_positive_gaussian(x, *popt1_127)
            # self.f_dVER_1_27 = lambda x: double_positive_gaussian_deriv(x, *popt1_127)  

            #do_plot = False 
            if do_plot:
                fig, (ax1,ax2) = plt.subplots(2,1) 
                z = np.linspace(70e3,150e3,100)   ### Always in meter ! 
                ax1.plot(z, self.f_VER_1_27(z), c="grey")
                ax1.plot(z, double_positive_gaussian(z, *popt1_127) , c="r", label="Fit by 2 Gaussians")
                ax1.plot(z, asymmetric_gaussian_pulse(z, *popt2_127) , c="g", label="Fit by asymetric Gaussian")
                ax1.plot(VER.alt, VER.VER, ls="", marker="d", c="k", label="Data")
                ax1.set_ylabel(r"VER, 1.27 $\mu m$ [$ph/s/m^3$]")
                ax1.set_xlim(z.min(), z.max())
                ax1.legend()
                ###
                ax2.plot(z, self.f_dVER_1_27(z), c="grey")
                ax2.plot(z, double_positive_gaussian_deriv(z, *popt1_127) , c="r", label="Fit by 2 Gaussian")
                ax2.plot(z, asymmetric_gaussian_pulse_deriv(z, *popt2_127) , c="g", label="Fit by asymetric Gaussian")
                ax2.plot(VER.alt, np.gradient(VER.VER, VER.alt, edge_order=2), ls="", marker="d", c="k", label="Data, gradient order 2")
                ax2.set_ylabel(r"Gradient of VER [$ph/s/m^4$]")
                ax2.set_xlabel("Altitude / [$m$]")
                ax2.set_xlim(z.min(), z.max())
                ax2.legend()
                fig.align_labels()
                fig.suptitle(r"Fit to the 1.27 $\mu m$ airglow data")
                # quit()
        else:
            self.f_VER_1_27 = interpolate.interp1d(VER.alt, VER.VER, kind='quadratic', bounds_error=False, fill_value=0.)
            self.f_dVER_1_27 = interpolate.interp1d(VER.alt, np.gradient(VER.VER, VER.alt, edge_order=2), kind='quadratic', bounds_error=False, fill_value=0.)

        
        ### =========================================================================================
        ### Read data for 4_28 micromter CO2 volume emission rate
        ### Units: it is given in W/m3 with a bandwidth of 0.03 micrometer 
        VER = pd.read_csv(file_4_28_airglow)
        VER.columns=['VER', 'alt']  ### ALt is in km
        ### CHOOSE VER UNIT  
        if self.unit=="photons":
            ### TEST: Convert VER to ph/m3/s using the energy of a photon 
            ### Ep = hc/lambda 
            ### Lambda = 4.28 micrometer 
            VER.VER *= self._factor_photons_watt(4.28, dir="W_to_ps")
        ### CONVERT ALTITUDES TO METER 
        VER.alt *= 1e3 
        self.z_4_28_min = VER.alt.min()
        self.z_4_28_max = VER.alt.max()
        # VER.to_csv(file_4_28_airglow.replace('.csv', '_scaled.csv'), index=False)

        ### EITHER FIT THE VER WITH SMOOTH FUNCTION OR INTERPOLATE DATA POINTS 
        do_fit=True#False
        if do_fit:
            ### FIT BY THREE GAUSSIANS
            def triple_positive_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3):
                return (np.abs(a1) * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) +\
                        np.abs(a2) * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)) +\
                        np.abs(a3) * np.exp(-((x - mu3) ** 2) / (2 * sigma3 ** 2)))
            def triple_positive_gaussian_deriv(x, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3):
                return (-np.abs(a1) *(x-mu1)/sigma1**2* np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) +\
                        -np.abs(a2) *(x-mu2)/sigma2**2* np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)) +\
                        -np.abs(a3) *(x-mu3)/sigma3**2* np.exp(-((x - mu3) ** 2) / (2 * sigma3 ** 2)))
            ### FIT BY ASYMMETRIC GAUSSIAN 
            def asymmetric_gaussian_pulse(x, a, mu, sigma_left, sigma_right):
                sigma = np.where(x < mu, sigma_left, sigma_right)
                return np.abs(a) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            def asymmetric_gaussian_pulse_deriv(x, a, mu, sigma_left, sigma_right):
                sigma = np.where(x < mu, sigma_left, sigma_right)
                return -np.abs(a) * (x-mu)/sigma**2 * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

            initial_guess1 = [VER.VER.max()/5.2, 119.4e3, 6.35e3, VER.VER.max(), 138e3, 5e3, VER.VER.max()/8, 145.8e3, 1e3]
            popt1_428, pcov1 = curve_fit(triple_positive_gaussian, VER.alt, VER.VER, p0=initial_guess1)
            ### Popt is not very good for last gaussian so we set it by force 
            popt1_428[-3:] = [VER.VER.max()/8, 145.8e3, 1e3]
            initial_guess2 = [VER.VER.max(), 120e3, 10e3, 11e3]
            popt2_428, pcov2 = curve_fit(asymmetric_gaussian_pulse, VER.alt, VER.VER, p0=initial_guess2)

            ### TEST: USE AN INTERPOLATED PULSE  
            ### NOTE: ENSURE GRADIENT IS IN SI UNITS !!!
            # self.f_VER_4_28 = lambda x: asymmetric_gaussian_pulse(x, *popt2_428)
            # self.f_dVER_4_28 = lambda x: asymmetric_gaussian_pulse_deriv(x, *popt2_428)  
            self.f_VER_4_28 = lambda x: triple_positive_gaussian(x, *popt1_428)
            self.f_dVER_4_28 = lambda x: triple_positive_gaussian_deriv(x, *popt1_428)  
            # self.f_VER_4_28 = interpolate.interp1d(VER.alt, VER.VER, kind='cubic', bounds_error=False, fill_value=(VER.VER.iloc[0], VER.VER.iloc[-1]))
            # self.f_dVER_4_28 = interpolate.interp1d(VER.alt, np.gradient(VER.VER, VER.alt, edge_order=2), kind='quadratic', bounds_error=False, fill_value=0.)

            #do_plot = False#True
            if do_plot:
                fig, (ax1,ax2) = plt.subplots(2,1) 
                z = np.linspace(80e3,160e3,200)   ### Always in meter ! 
                ax1.plot(z, self.f_VER_4_28(z), c="grey")
                ax1.plot(z, triple_positive_gaussian(z, *popt1_428) , c="r", label="Fit by 3 Gaussians")
                ax1.plot(z, asymmetric_gaussian_pulse(z, *popt2_428) , c="g", label="Fit by asymetric Gaussian")
                ax1.plot(VER.alt, VER.VER, ls="", marker="d", c="k", label="Data")
                ax1.set_ylabel(r"VER, 4.28 $\mu m$ [$ph/s/m^3$]")
                ax1.set_xlim(z.min(), z.max())
                ax1.legend()
                ###
                ax2.plot(z, self.f_dVER_4_28(z), c="grey")
                ax2.plot(z, triple_positive_gaussian_deriv(z, *popt1_428) , c="r", label="Fit by 3 Gaussian")
                ax2.plot(z, asymmetric_gaussian_pulse_deriv(z, *popt2_428) , c="g", label="Fit by asymetric Gaussian")
                ax2.plot(VER.alt, np.gradient(VER.VER, VER.alt, edge_order=2), ls="", marker="d", c="k", label="Data, gradient order 2")
                ax2.set_ylabel(r"Gradient of VER [$ph/s/m^4$]")
                ax2.set_xlabel("Altitude / [$m$]")
                ax2.set_xlim(z.min(), z.max())
                ax2.legend()
                fig.align_labels()
                fig.suptitle(r"Fit to the 4.28 $\mu m$ airglow data")
                # quit()
        else:
            self.f_VER_4_28 = interpolate.interp1d(VER.alt, VER.VER, kind='cubic', bounds_error=False, fill_value=(VER.VER.iloc[0], VER.VER.iloc[-1]))
            self.f_dVER_4_28 = interpolate.interp1d(VER.alt, np.gradient(VER.VER, VER.alt, edge_order=2), kind='quadratic', bounds_error=False, fill_value=0.)


        ### =========================================================================================
        ### Use a different type of data 
        if use_kenda_atm:
            ### NOTE: Here the VER curves have already been interpolated  
            ### One is in W/m3, the other in ph/s/m3. 
            ### Altitude starts only at 90 km 
            gamma_kenda=11./9.
            VER_kenda = pd.read_csv(file_airglow_kenda)
            ### CONVERT TO METER 
            VER_kenda.z *= 1e3
            
            self.f_rho = interpolate.interp1d(VER_kenda.z, VER_kenda.rho, kind='quadratic', bounds_error=False, fill_value=(VER_kenda.rho.min(), VER_kenda.rho.max()))
            self.f_t = interpolate.interp1d(VER_kenda.z, VER_kenda['T'], kind='quadratic')
            self.f_gamma = interpolate.interp1d(atm_profile.altitude, atm_profile.gamma*0.+gamma_kenda, kind='linear')  ### NOTE: Constant ? 
            self.f_c = interpolate.interp1d(VER_kenda.z, VER_kenda.c, kind='quadratic')

            self.f_VER_1_27 = interpolate.interp1d(VER_kenda.z, VER_kenda.VER_127, kind='quadratic', bounds_error=False, fill_value=0.)
            self.f_VER_4_28 = interpolate.interp1d(VER_kenda.z, VER_kenda.VER_428*self._factor_photons_watt(4.28, dir="W_to_ps"),
                                                   kind='cubic', bounds_error=False, fill_value=(VER_kenda.VER_428.iloc[0], VER_kenda.VER_428.iloc[-1]))
            
            self.z_1_27_min = VER_kenda.z.min()
            self.z_1_27_max = VER_kenda.z.max()

        if do_plot:
            ### DISPLAY THE ATMOSPHERIC PROPERTIES 
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(9,5)) 
            axes = [ax1, ax2, ax3, ax4, ax5]
            zplot = np.linspace(0e3,150e3,400)
            ### 
            ax1.plot(self.f_rho(zplot), zplot/1e3, c="k")
            ax1.set_xlabel(r"Density / [$kg/m^3$]")
            ###
            ax2.plot(self.f_t(zplot), zplot/1e3, c="k")
            ax2.set_xlabel(r"Temperature / [$K$]")
            ###
            ax3.plot(self.f_c(zplot), zplot/1e3, c="k")
            ax3.set_xlabel(r"Sound speed / [$m/s$]")
            ###
            ax4.plot(self.f_gamma(zplot), zplot/1e3, c="k")
            ax4.set_xlabel(r"$\gamma$")
            ###
            ax5.fill_betweenx(zplot/1e3, zplot*0, self.f_VER_1_27(zplot), alpha=0.5, 
                              edgecolor="forestgreen", facecolor="forestgreen")
            ax5.fill_betweenx(zplot/1e3, zplot*0, self.f_VER_4_28(zplot), alpha=0.5, 
                              edgecolor="orangered", facecolor="orangered")
            ax5.plot([],[], color="forestgreen", label=r"1.27 $\mu m$")
            ax5.plot([],[], color="orangered", label=r"4.28 $\mu m$")
            ax5.set_xlabel(r"VER / [$ph/s/m^3$]")
            ###
            ax5.legend(loc=3, edgecolor="none", framealpha=0)
            for ax in axes:
                ax.set_ylim(0,150)
            axes[0].set_ylabel(r"Altitude / [$km$]")
            fig.suptitle("Sanity check of atmospheric parameters used")
            fig.tight_layout()

        return 

    
    def _load_absorption_amplification(self, file_attenuation_kenda=None, dir_attenuation_GA=None, do_plot=False, disable_att=False):

        if file_attenuation_kenda is not None:
            if not disable_att:
                print("Using Balthasar Kenda's attenuation data")
            ### Then define a function to go from vz_surface to vz_z (amplification + attenuation)
            atten = pd.read_csv(file_attenuation_kenda, header=[0])
            alts = atten.alt.unique()*1e3  ### ALTITUDES ALWAYS IN METER ! 
            freq = atten.frequency.unique()
            FF, AA = np.meshgrid(freq, alts)

            ### Factor to got from Np/m to dB/km
            ### NOTE: We now think that B. Kenda's data is in dB/km. 
            fac_Npm_dBkm = 20*np.log10(np.e)*1000
            alpha_dBkm = atten.alpha.values.reshape((alts.size, freq.size))
            alpha_Npm = alpha_dBkm/fac_Npm_dBkm

            if do_plot:
                fig2, ax2 = plt.subplots()
                cols = plt.get_cmap("viridis")
                for i in range(0,800,80) :
                    ax2.plot(freq, alpha_dBkm[i,:], c=cols(i/799), label = "{:.1f}".format(alts[i]/1e3)) 
                ax2.set_yscale("log")
                ax2.set_xscale("log")
                ax2.legend(title="Altitude / [$km$]", ncol=2, framealpha=0.5, edgecolor="none")
                ax2.set_xlabel("Frequency / [$Hz$]")
                ax2.set_ylabel("Attenuation / [dB/km]")


            ### 1D interpolation for alpha (as a function of frequency / for each individual altitude)
            ### We use Np/m values for this for the exponential calculation
            f_alpha = interpolate.interp1d(freq, alpha_Npm, axis=1, bounds_error=False, fill_value=0.0)
            ### 2D interpolation (as a function of frequency + altitude)
            f_alpha_2d = interpolate.RegularGridInterpolator((alts,freq), alpha_Npm, method='linear',fill_value=0, bounds_error=False)
            ### Read amplification from file 
            amplification = atten.amplification.values.reshape((alts.size, freq.size))[:,0]
            ### 1D interpolatiom (it doesn't depend on frequency)
            f_amplification = interpolate.interp1d(alts, amplification, kind='quadratic')

        elif dir_attenuation_GA is not None:
            if not disable_att:
                print("Using Gil Averbuch's attenuation data")
            files_unsorted = [dir_attenuation_GA + f for f in os.listdir(dir_attenuation_GA)]
            freq_unsorted = [float(f.split("Hz.csv")[0]) for f in os.listdir(dir_attenuation_GA)]

            files = [x for _, x in sorted(zip(freq_unsorted, files_unsorted))]
            freq = np.sort(freq_unsorted)

            alpha = [] 
            for f in files:
                dat = np.genfromtxt(f, skip_header=1, usecols=(0,5), delimiter=",")
                alpha.append(dat)
            alpha_dBkm = np.array(alpha)[:,:,1].T
            ### We know that GA's data is in dB/km
            fac_Npm_dBkm = 20*np.log10(np.e)*1000
            alpha_Npm = alpha_dBkm/fac_Npm_dBkm
            ### Shape: Nfreq, NZ
            alts = np.array(alpha)[0,:,0]*1e3  ### ALTITUDES ALWAYS IN METER 

            if do_plot:
                fig2, ax2 = plt.subplots()
                cols = plt.get_cmap("viridis")
                for i in range(0,alts.size,40) :
                    ax2.plot(freq, alpha_dBkm[i,:], c=cols(i/(alts.size-1)), label = "{:.1f}".format(alts[i]/1e3)) 
                ax2.set_yscale("log")
                ax2.set_xscale("log")
                ax2.legend(title="Altitude / [$km$]", ncol=2, framealpha=0.5, edgecolor="none")
                ax2.set_xlabel("Frequency / [$Hz$]")
                ax2.set_ylabel("Attenuation / [dB/km]")

            ### 1D interpolation for alpha (as a function of frequency / for each individual altitude)
            ### We use Np/m values for this for the exponential calculation
            f_alpha = interpolate.interp1d(freq, alpha_Npm, axis=1, bounds_error=False, fill_value=0.0)
            ### 2D interpolation (as a function of frequency + altitude)
            f_int = RectBivariateSpline(alts, np.log10(freq), np.log10(alpha_Npm), kx=3, ky=3)
            def f_alpha_2d (zf):
                z = zf[0]
                f = np.log10(zf[1])
                int_res = 10**f_int.ev(z, f)
                return( int_res )

            ### Calculate amplification from conservation of kinetic energy
            ### and out atmosphere model: 
            amplification = np.sqrt(self.f_rho(0)*self.f_c(0)/ (self.f_rho(alts)*self.f_c(alts)))
            ### 1D interpolatiom (it doesn't depend on frequency)
            def f_amplification(z):
                return( np.sqrt(self.f_rho(0)*self.f_c(0)/ (self.f_rho(z)*self.f_c(z))) )
            
        if disable_att: 
            print("Attenuation disabled")
            ### Set alpha to zero everywhere 
            f_alpha = interpolate.interp1d(freq, alpha_Npm*0, axis=1, bounds_error=False, fill_value=0.0)
            f_alpha_2d = interpolate.RegularGridInterpolator((alts,freq), alpha_Npm*0, method='linear',fill_value=0, bounds_error=False)

        if do_plot:
            fig, (axa, axb) = plt.subplots(1,2, figsize=(8,4))
            axa.pcolormesh(freq, alts/1e3, np.log10(alpha_dBkm), cmap="viridis", vmin = np.log10(alpha_dBkm.min()), vmax = np.log10(alpha_dBkm.max()))
            axa.set_xscale("log")
            axa.set_ylabel("Altitude / [$km$]")
            axa.set_xlabel("Frequency / [$Hz$]")
            axa.set_title("Raw attenuation (log)")
            axa.set_xlim(freq.min(), freq.max())
            ###
            f_test = 10**np.linspace(np.log10(freq.min()), np.log10(freq.max()), 200)
            z_test = np.linspace(0, alts.max(), 400)
            FF, ZZ = np.meshgrid(f_test, z_test)
            RES = f_alpha_2d((ZZ,FF)) 
            axb.pcolormesh(f_test, z_test/1e3, np.log10(RES*fac_Npm_dBkm), cmap="viridis", vmin = np.log10(alpha_dBkm.min()), vmax = np.log10(alpha_dBkm.max()))
            axb.set_xscale("log")
            axb.set_ylabel("Altitude / [$km$]")
            axb.set_xlabel("Frequency / [$Hz$]")
            axb.set_title("Interpolated attenuation (log)")
            axb.set_xlim(freq.min(), freq.max())
            ###
            fig.tight_layout()
            ###

        if do_plot:
            fig2, ax2b = plt.subplots(1,1,figsize=(5,4))
            zplot = np.linspace(0e3,200e3, 400)
            ###
            ax2b.plot(alts/1e3, amplification, c="k")
            ax2b.plot(zplot/1e3, np.sqrt(self.f_rho(0)*self.f_c(0)/ (self.f_rho(zplot)*self.f_c(zplot))), c="b")
            i90 = np.argmin(np.abs(alts-90e3))
            amp90 =  amplification[i90]
            # ax2b.plot(alts, amplification[:,0]/amp90, c="k")
            ax2b.set_yscale("log")
            # ax2b.legend(title="Frequency / [$Hz$]", ncol=2, framealpha=0.5, edgecolor="none")
            ax2b.set_xlabel("Altitude / [$km$]")
            ax2b.set_ylabel(r"Velocity amplification")
            ax2b.set_xlim(zplot.min()/1e3, zplot.max()/1e3)
            ###
            fig2.suptitle(r"Velocity amplification with altitude: $\sqrt{\rho(0)c(0)} / \sqrt{\rho(z)c(z)}$")
            fig2.tight_layout()
        
        
        return(f_alpha, f_alpha_2d, f_amplification)
    

    def _def_filter_nightglow(self, f_lim=1e-3):
        ### Transfer function for OPTION 2: scipy-defined filtering 
        #tau = 4460  # seconds
        num = [-self.tau]
        den = [self.tau, 1]
        ###
        ### First-order high-pass filter with cutoff f_lim:
        ### H_hp(s) = s / (s + omega_c)
        omega_c = 2 * np.pi * f_lim
        num2 = [1, 0]
        den2 = [1, omega_c]

        ### Multiply the transfer functions
        # num = np.polymul(num, num2)
        # den = np.polymul(den, den2)
        ### Do a second multiplication to get second order 
        # num = np.polymul(num, num2)
        # den = np.polymul(den, den2)

        system_d = cont2discrete((num, den), self.dt, method='bilinear')
        b, a = system_d[0].flatten(), system_d[1].flatten()
        return(b,a)


    def calculate_1_27_airglow(self, list_ieast, list_inorth, loc_save_idx=None, loc_save_EN = None, 
                               time_save = None, save_ver = True, fourier_filtering=False, 
                               n_cpus=10, do_parallel=True, tmax=2500, dir_save="./results/"):
        ### NOTE: To avoid mistakes in gradients, 
        ### all the calculations are done with z in METERS 
        
        ### Ensure we have a time series: 
        if self.t_new is None: 
            t = self.synthetic_traces_v[0].get_xdata()
            self.t_new = np.arange(0., max(t.max(), tmax), self.dt)
            self.Nt = self.t_new.size

        ### Save vertical profiles at 10 first locations by default
        if loc_save_idx is None and loc_save_EN is None:
            # loc_save_idx = list(zip(list_ieast[:10], list_inorth[:10]))
            loc_save_idx = list(zip(list_ieast, list_inorth))
        elif loc_save_EN is not None:
            loc_save_idx = []
            for es, ns in loc_save_EN:
                idx = np.argmin(np.sqrt((self.NN-ns)**2+(self.EE-es)**2)) 
                idx = np.unravel_index(idx, self.EE.shape)
                loc_save_idx.append(idx)
        elif loc_save_idx is not None:
            loc_save_EN = []
            for ies, ins in loc_save_idx:
                loc_save_EN.append((self.EE[ies,ins], self.NN[ies, ins] ))
        print("(East, North) location indices that will be saved: ", loc_save_idx)

        ### Save wavefield every 10 timesteps by default
        if time_save is None: 
            time_save = self.t_new[::int(self.Nt//10)]

        itime_save = [np.where(abs(self.t_new-tw)<self.dt/2)[0][0] for tw in time_save]
        if save_ver:
            save_wavefield = np.zeros((self.Ne, self.Nn, self.Nz, len(time_save),2 ))
        save_intensity_dver = np.zeros((self.Ne, self.Nn,len(time_save)))

        ### We pad seismograms in time: 
        self.dpad = sfft.next_fast_len(self.Nt*2, real=True) - self.Nt
        long_VEL = np.pad(self.VEL, ((0,0),(0,0),(0, self.dpad )), mode='constant')
        long_DIS = np.pad(self.DIS, ((0,0),(0,0),(0, self.dpad )), mode='constant')
        # al = 0.1
        #long_VEL[:,:,self.Nt:] = tukey(2*self.dpad, alpha=al)[None,None,self.dpad:]*self.VEL[:,:,-1][:,:,None]
        # fig, ax = plt.subplots() 
        # ax.plot(long_VEL[0,19,:])
        # ax.plot(self.VEL[0,19,:])
        # print(brou)
        ### Fourier transform of seismograms 
        global fft_vzs, fft_uzs  
        fft_vzs = sfft.fft(long_VEL, axis=2)
        fft_uzs = sfft.fft(long_DIS, axis=2)
        Ntpad = long_VEL.shape[2]
        # fft_vzs = sfft.fft(self.VEL, axis=2)

        ### Define frequencies 
        # freqsi = sfft.fftfreq(d=self.dt, n=self.Nt)
        freqsi = sfft.fftfreq(d=self.dt, n=Ntpad)
        freqsp = abs(freqsi)
    
        ### Pre-calculate the phase shift corresponding to the delay with altitude 
        # self.phase_shift_z = np.zeros((self.Nz, self.Nt), dtype = np.complex64)
        self.phase_shift_z = np.zeros((self.Nz, Ntpad), dtype = np.complex64)
        ### Integrated travel time from zero to airglow altitudes 
        self.travel_time = integrate.cumulative_trapezoid(1/self.f_c(self.z_att_1_27_m), self.z_att_1_27_m)
        self.travel_time = self.travel_time[-self.Nz:]
        for jz in range(self.Nz):
            ### Integrated propagation velocity from zero to altitude z  
            self.phase_shift_z[jz,:] = np.exp(-2 * np.pi * freqsi * 1j * self.travel_time[jz] )

        ### Enforce Hermitian symmetry explicitly (works for odd/even N)
        if Ntpad % 2 == 0:
            pos = np.arange(1, Ntpad//2)        # 1..(N/2-1)
            self.phase_shift_z[:, 0] = self.phase_shift_z[:, 0].real
            self.phase_shift_z[:, Ntpad//2] = self.phase_shift_z[:, Ntpad//2].real
        else:
            pos = np.arange(1, (Ntpad-1)//2 + 1)  # 1..(N-1)//2
            self.phase_shift_z[:, 0] = self.phase_shift_z[:, 0].real
        neg = (-pos) % Ntpad
        self.phase_shift_z[:, neg] = np.conj(self.phase_shift_z[:, pos])



        ### If using fourier filtering, prepare the filter: 
        # if fourier_filtering:
        self.tf_phase_nightglow = -(self.tau/(1+1j*2*np.pi*freqsi[None,:]*self.tau)) 
        ### Definition based on advection term
        # self.tf_phase_nightglow = ((2*self.tau+1/(1j*2*np.pi*freqsi[None,:]))/(1+1j*2*np.pi*freqsi[None,:]*self.tau)) 
        ### Set no gain at low frequencies 
        ### PB: Sets mean to zero but trend is still there
        ### After detrending + resetting VER(0)=0, this effectively doesn't do anything... 
        self.tf_phase_nightglow[:,0] = 0.0 + 0.0j
            
        ### Better option: Applying a high-pass filter to VER(t): H_hp(i omega) = iomega / (i omega + omega_c)
        # self.tf_phase_nightglow *= (1j*freqsi[None,:] / (1j*freqsi[None,:] + 1e-4))**2
        # else:
        #     self.tf_phase_nightglow = None


        ### Grid of amplification 
        self.amplification = self.f_amplification(self.z_1_27_calc_m)

        ### Grid of attenuation: OPTION WITHOUT CUMULATIVE SUM 
        #FFver, ZZver = np.meshgrid(freqsp, self.z_1_27_calc_m)
        #attenuation = self.f_alpha_2d((ZZver, FFver))
        ### Exponential of attenuation (supposing Np/km)
        #self.att_exp = np.exp(-self.z_1_27_calc_km[:,np.newaxis]*attenuation)   ### meter or kilometer ? Cumulative sum or not ? 

        ### NOTE: Cumulative sum works only if we are starting from z=0
        FFver, ZZver2 = np.meshgrid(freqsp, self.z_att_1_27_m)
        attenuation = self.f_alpha_2d((ZZver2, FFver))
        att_exp = np.exp(-integrate.cumulative_trapezoid(attenuation, self.z_att_1_27_m, axis=0))   ### Supposes Np/m 
        self.att_exp = att_exp[-self.Nz:]
        # fig, ax = plt.subplots() 
        # for i in range(self.Nz):
        #     ax.plot(freqsp, self.att_exp[i,:])        
        
        ### Prepare the loop 
        list_of_locations = list(zip(list_ieast, list_inorth))
        list_indices = range(len(list_of_locations))

        ### ATTEMPT: Calculate nightglow intensity faster with high resolution integral. 
        # zrange = np.linspace(90e3,140e3,1000)
        # zatt = np.concatenate((np.linspace(0,zrange[0]-1e3,100),zrange))
        # phase = np.exp(-2*1j*np.pi*freqsi[None,:]*integrate.cumulative_trapezoid(1/self.f_c(zatt), zatt, initial=0)[:,None])
        # dphase_dz = -2*1j*np.pi*freqsi[None,:]*1/self.f_c(zatt)[:,None]*phase 
        # ###
        # ampl = self.f_amplification(zatt)[:,None]
        # dampl_dz = np.gradient(ampl, zatt, axis=0)
        # ###
        # FFver, ZZver2 = np.meshgrid(freqsp, zatt)
        # attenuation = self.f_alpha_2d((ZZver2, FFver))
        # att = np.exp(-integrate.cumulative_trapezoid(attenuation, zatt, axis=0, initial=0))
        # datt_dz = -attenuation*att
        # ###
        # global dphase_att_ampl_dz
        # dphase_att_ampl_dz = dphase_dz*ampl*att + dampl_dz*phase*att + datt_dz*phase*ampl

        # print(datt_dz.shape)
        # print(att.shape)
        # print(phase.shape)
        # print(dphase_dz.shape)
        # print(fft_vzs.shape)
        # print(ampl.shape)
        # print(dampl_dz.shape)
        # quit()


        ### SERIAL VERSION OF THE CALCULATION 
        if not do_parallel: 
            # import time as ptime 
            # t1=ptime.time()
            for i_en in tqdm(list_indices, total=len(list_ieast), disable=False):
                # i_en = 3360
                # save_wavefield, save_intensity_dver = self.nightglow_at_location(i_en, 
                #                                                                 fft_vzs, fourier_filtering, 
                #                                                                 save_wavefield, save_intensity_dver, 
                #                                                                 loc_save, itime_save)
                results = nightglow_at_location(i_en, list_of_locations, fft_vzs, fft_uzs, self.att_exp, self.amplification, 
                                                                          self.phase_shift_z,self.f_VER_1_27,self.f_dVER_1_27,
                                                                        self.z_1_27_calc_m, fourier_filtering, self.Nt, 1/self.dt,self.b, self.a, 
                                                                        loc_save_idx, itime_save, save_ver, self.gridded, self.tf_phase_nightglow, dir_save)#,
                                                                        # dphase_att_ampl_dz, zatt, zrange)
                
                i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]
                if save_ver:
                    save_wavefield[i_east, i_north, :,:,0] = results[0]#vz_z_it        ### Save velocity waveform 
                    save_wavefield[i_east, i_north, :,:,1] = results[1]#dver_z_it      ### Save dVER at altitude z 
                save_intensity_dver[i_east, i_north,:] = results[-1]#amp_nightglow_it  ### Save Intensity at altitude z

            # t2 = ptime.time()
            # print(t2-t1)
        ### PARALLEL WITH MULTIPROCESSING (requires all functions defined outside of class)
        else:
            t0=ptime.time()
            ### NOTE: "fork" is much faster than "spawn" as spawn spends time copying array to different workers 
            ### However, fork doesn't work on windows/mac. To get a more flexible but fast parallelisation, 
            ### we need to move to joblib and memory maps. 
            with get_context("fork").Pool(processes=n_cpus,
                                            initializer=init_worker_nightlow,
                                            initargs=(list_of_locations, 
                                                    #   fft_vzs, fft_uzs,  
                                                      self.att_exp, self.amplification, self.phase_shift_z,
                                                    self.f_VER_1_27, self.f_dVER_1_27, self.z_1_27_calc_m, fourier_filtering,self.Nt, 1/self.dt,
                                                    self.b, self.a, 
                                                    loc_save_idx, itime_save, save_ver, self.gridded, self.tf_phase_nightglow, dir_save)#,
                                                    # dphase_att_ampl_dz, 
                                                    # zatt, zrange)
                                                ) as p:
                
                results = list(tqdm(p.imap(worker_func_nightglow, list_indices), total=len(list_indices), bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}' ))
                t2 = ptime.time()
            print("Time for airglow calculation: {:.1f} s".format(t2-t0))

            ### Store wavefields 
            # if self.gridded: 
            print("Re-aranging gridded wavefield...")
            for i_en, r in enumerate(tqdm(results, total=len(results), bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}' ) ):
                i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]
                    
                if save_ver:
                    vz_z_it = r[0]
                    dver_z_it = r[1]
                    save_wavefield[i_east, i_north, :,:,0] = vz_z_it         ### Save velocity waveform at all requested altitudes and times
                    save_wavefield[i_east, i_north, :,:,1] = dver_z_it       ### Save dVER at all requested altitudes and times
                
                amp_nightglow_it = r[-1]
                save_intensity_dver[i_east, i_north,:] = amp_nightglow_it  ### Save Intensity at requested times  

            
        ### Save the full wavefield, but only at certain times 
        print("Saving gridded wavefield...")
        if save_ver:
            np.save(dir_save + "nightglow_dver_t", save_wavefield)
        np.save(dir_save + "nightglow_I_t", save_intensity_dver)
        print("Grid save completed.")


    def calculate_4_28_airglow(self, list_ieast, list_inorth, loc_save_idx=None, loc_save_EN = None,
                                 time_save = None, save_ver=True,
                                 n_cpus=10, do_parallel=True, tmax=2500, dir_save="./results/"):
        ### NOTE: To avoid mistakes in gradients, 
        ### all the calculations are done with z in METERS 
        
        ### Ensure we have a time series: 
        if self.t_new is None: 
            t = self.synthetic_traces_v[0].get_xdata()
            self.t_new = np.arange(0., max(t.max(), tmax), self.dt)
            self.Nt = self.t_new.size

        ### Save vertical profiles at 10 first locations by default
        if loc_save_idx is None and loc_save_EN is None:
            # loc_save_idx = list(zip(list_ieast[:10], list_inorth[:10]))
            loc_save_idx = list(zip(list_ieast, list_inorth))
            # loc_save_idx = []
        elif loc_save_EN is not None:
            loc_save_idx = []
            for es, ns in loc_save_EN:
                idx = np.argmin(np.sqrt((self.NN-ns)**2+(self.EE-es)**2)) 
                idx = np.unravel_index(idx, self.EE.shape)
                loc_save_idx.append(idx)
        elif loc_save_idx is not None:
            loc_save_EN = []
            for ies, ins in loc_save_idx:
                loc_save_EN.append((self.EE[ies,ins], self.NN[ies, ins] ))
        print("(East, North) location indices that will be saved: ", loc_save_idx)

        ### Save wavefield every 10 timesteps by default
        if time_save is None: 
            time_save = self.t_new[::int(self.Nt//10)]

        itime_save = [np.where(abs(self.t_new-tw)<self.dt/2)[0][0] for tw in time_save]
        if save_ver:
            save_wavefield = np.zeros((self.Ne, self.Nn, self.Nz, len(time_save),2 ))
        save_intensity_dver = np.zeros((self.Ne, self.Nn,len(time_save)))

        ### We pad seismograms in time: 
        self.dpad = sfft.next_fast_len(self.Nt*2, real=True) - self.Nt
        al = 0.1
        long_DIS = np.pad(self.DIS, ((0,0),(0,0),(0, self.dpad )), mode='constant')
        #long_DIS[:,:,self.Nt:] = tukey(2*self.dpad, alpha=al)[None,None,self.dpad:]*self.DIS[:,:,-1][:,:,None]
        # fig, ax = plt.subplots() 
        # ax.plot(long_DIS[0,19,:])
        # ax.plot(self.DIS[0,19,:])
        # print(brou)
        ### Fourier transform of seismograms 
        fft_uzs = sfft.fft(long_DIS, axis=2)
        Ntpad = long_DIS.shape[2]
        # fft_uzs = sfft.fft(self.DIS, axis=2)
        # fft_vzs = sfft.fft(self.VEL, axis=2)

        ### Define frequencies 
        # freqsi = sfft.fftfreq(d=self.dt, n=self.Nt)
        freqsi = sfft.fftfreq(d=self.dt, n=Ntpad)
        freqsp = abs(freqsi)

        ### Pre-calculate the phase shift corresponding to the delay with altitude 
        # self.phase_shift_z = np.zeros((self.Nz, self.Nt), dtype = np.complex64)
        self.phase_shift_z = np.zeros((self.Nz, Ntpad), dtype = np.complex64)
        ### Integrated travel time from zero to airglow altitudes 
        self.travel_time = self.dz_4_28_m * np.cumsum(1/self.f_c(self.z_att_4_28_m))
        self.travel_time = self.travel_time[-self.Nz:]
        for jz in range(self.Nz):
            ### Integrated propagation velocity from zero to altitude z  
            self.phase_shift_z[jz,:] = np.exp(-2 * np.pi * freqsi * 1j * self.travel_time[jz] )
        ### Enforce Hermitian symmetry explicitly (works for odd/even N)
        if Ntpad % 2 == 0:
            pos = np.arange(1, Ntpad//2)        # 1..(N/2-1)
            self.phase_shift_z[:, 0] = self.phase_shift_z[:, 0].real
            self.phase_shift_z[:, Ntpad//2] = self.phase_shift_z[:, Ntpad//2].real
        else:
            print("here")
            pos = np.arange(1, (Ntpad-1)//2 + 1)  # 1..(N-1)//2
            self.phase_shift_z[:, 0] = self.phase_shift_z[:, 0].real
        neg = (-pos) % Ntpad
        self.phase_shift_z[:, neg] = np.conj(self.phase_shift_z[:, pos])

        ### Grid of amplification 
        self.amplification = self.f_amplification(self.z_4_28_calc_m)

        ### NOTE: Cumulative sum works only if we are starting from z=0
        FFver, ZZver2 = np.meshgrid(freqsp, self.z_att_4_28_m)
        attenuation = self.f_alpha_2d((ZZver2, FFver))
        # att_exp = np.exp(-self.dz_4_28_m*np.cumsum(attenuation, axis=0))   ### Supposes Np/m 
        att_exp = np.exp(-integrate.cumulative_trapezoid(attenuation, self.z_att_4_28_m, axis=0))
        self.att_exp = att_exp[-self.Nz:]

        ### Temperature factor: 
        # dVER_dayglow = self.alpha * self.f_VER_4_28(alt)*(self.f_gamma(alt)-1)*self.f_t(alt)*np.gradient(uz)
        self.fac_temperature = self.alpha_t * self.f_VER_4_28(self.z_4_28_calc_m)*\
                                    (self.f_gamma(self.z_4_28_calc_m)-1)*\
                                    self.f_t(self.z_4_28_calc_m)
        ### Test with P.Y. F. expression
        # self.fac_temperature = self.alpha_t * self.f_VER_4_28(self.z_4_28_calc_m)*\
        #                             (self.f_gamma(self.z_4_28_calc_m)-1)*\
        #                             self.f_t(self.z_4_28_calc_m) * 1/self.f_c(self.z_4_28_calc_m)
        
        ### Prepare the loop 
        list_of_locations = list(zip(list_ieast, list_inorth))
        list_indices = range(len(list_of_locations))

        ### SERIAL VERSION OF THE CALCULATION 
        if not do_parallel: 
            # import time as ptime 
            t0=ptime.time()
            for i_en in tqdm(list_indices, total=len(list_ieast), disable=False):
                # i_en = 3360
                results = dayglow_at_location(i_en, list_of_locations, fft_uzs, self.att_exp, self.amplification, 
                                                                          self.phase_shift_z, self.Nt, self.z_4_28_calc_m, self.fac_temperature, self.f_VER_4_28, self.f_dVER_4_28, 
                                                                          loc_save_idx, itime_save, save_ver, self.gridded, dir_save)
                
                i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]
                if save_ver:
                    uz_z_it, dver_z_it = results[0], results[1]
                    save_wavefield[i_east, i_north, :,:,0] = uz_z_it    ### Save dispalcement waveform 
                    save_wavefield[i_east, i_north, :,:,1] = dver_z_it  ### Save dVER at altitude z 
                
                amp_dayglow_it = results[-1]
                save_intensity_dver[i_east, i_north,:] = amp_dayglow_it  ### Save Intensity at altitude z

            t2 = ptime.time()
            print("Time for dayglow calculation: {:.1f} s".format(t2-t0))

        ### PARALLEL WITH MULTIPROCESSING (requires all functions defined outside of class)
        else:
            t0=ptime.time()
            ### NOTE: "fork" is much faster than "spawn" as spawn spends time copying array to different workers 
            ### However, fork doesn't work on windows/mac. To get a more flexible but fast parallelisation, 
            ### we need to move to joblib and memory maps. 
            with get_context("fork").Pool(processes=n_cpus,
                                            initializer=init_worker_dayglow,
                                            initargs=(list_of_locations, fft_uzs, self.att_exp, self.amplification, self.phase_shift_z, self.Nt,
                                                    self.z_4_28_calc_m, self.fac_temperature, self.f_VER_4_28, self.f_dVER_4_28, loc_save_idx, 
                                                    itime_save, save_ver, self.gridded, dir_save)
                                                ) as p:
                
                results = list(tqdm(p.imap(worker_func_dayglow, list_indices), total=len(list_indices), bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}' ))
                t2 = ptime.time()
            print("Time for dayglow calculation: {:.1f} s".format(t2-t0))

            ### Store wavefields 
            # if self.gridded: 
            print("Re-aranging gridded dayglow wavefield...")
            for i_en, r in enumerate(tqdm(results, total=len(results), bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}' ) ):
                i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]
                    
                if save_ver:
                    uz_z_it = r[0]
                    dver_z_it = r[1]
                    save_wavefield[i_east, i_north, :,:,0] = uz_z_it         ### Save displacement waveform at all requested altitudes and times
                    save_wavefield[i_east, i_north, :,:,1] = dver_z_it       ### Save dVER at all requested altitudes and times
                
                amp_airglow_it = r[-1]
                save_intensity_dver[i_east, i_north,:] = amp_airglow_it  ### Save Intensity at requested times  

            
        ### Save the full wavefield, but only at certain times 
        print("Saving gridded dayglow wavefield...")
        if save_ver:
            np.save(dir_save + "dayglow_dver_t", save_wavefield)
        np.save(dir_save + "dayglow_I_t", save_intensity_dver)
        print("Grid save completed.")
    
    
    def plot_nightglow_traces(self, loc_plot = None, idx_plot = None, z1=92, z2=112, time_end=2000, dir_save="results/"):

        ### Convert east and north coordinate to indices if needed 
        if loc_plot is None and idx_plot is None:
            raise("Requires a couple of coordinates (loc_plot) or a couple of indices (idx_plot).")
        elif loc_plot is not None:
            idx = np.argmin(np.sqrt((self.NN-loc_plot[1])**2+(self.EE-loc_plot[0])**2)) 
            i_east, i_north = np.unravel_index(idx, self.EE.shape)
        else:
            i_east, i_north = idx_plot[0], idx_plot[1]
    
        ### Find the right altitude
        alts_airglow = self.z_1_27_calc_m
        iz1 = np.argmin(np.abs(alts_airglow-z1*1e3))
        iz2 = np.argmin(np.abs(alts_airglow-z2*1e3))
        ### Get appropriate signals 
        dver_z_loc = np.load(dir_save + "nightglow_dver_z_{:d}_{:d}.npy".format(i_east, i_north))
        dver_z1 = dver_z_loc[iz1,:]
        dver_z2 = dver_z_loc[iz2,:]
        vz_z_loc = np.load(dir_save + "nightglow_vz_z_{:d}_{:d}.npy".format(i_east, i_north))
        vz_z1 = vz_z_loc[iz1,:]
        vz_z2 = vz_z_loc[iz2,:]
        ### Load integrated intensity and calculate background 
        I_dver_nightglow = np.load(dir_save + "nightglow_I_{:d}_{:d}.npy".format(i_east, i_north))

        ####################################################################################
        fig, (axt,axm, axb, axbb) = plt.subplots(4,1,figsize=(8,8) )
        axt.plot(self.t_new, self.VEL[i_east, i_north,:], c="k", lw=1) 
        axt.set_title("Ground signal at {:.0f} km E, {:.0f} km N".format(self.EE[i_east,i_north]/1e3,self.NN[i_east,i_north]/1e3))
        axt.set_ylabel(r"Velocity / [$m/s$]")
        # axt.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
        axt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ###
        axm.plot(self.t_new, vz_z1, c="navy", lw=1, label="z={:.1f} km".format(alts_airglow[iz1]/1e3))
        axm.plot(self.t_new, vz_z2, c="purple", lw=1, label="z={:.1f} km".format(alts_airglow[iz2]/1e3))
        axm.set_title("Surface signal")
        axm.set_title("Amplified, propagated, attenuated")
        axm.set_ylabel(r"Velocity / [$m/s$]")
        axm.legend(loc=2, frameon=False)
        #axm.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
        axm.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ####################################################################################
        axb.plot(self.t_new, dver_z1/self.f_VER_1_27(alts_airglow[iz1])*100, c="navy", ls="-", label="z={:.1f} km".format(alts_airglow[iz1]/1e3), lw=1)
        axb.plot(self.t_new, dver_z2/self.f_VER_1_27(alts_airglow[iz2])*100, c="purple", ls="-", label="z={:.1f} km".format(alts_airglow[iz2]/1e3), lw=1)
        axb.set_title("Percentage of perturbation of VER")

        # axb.plot(self.t_new, dver_z1, c="navy", ls="-", label="z={:.1f} km".format(alts_airglow[iz1]), lw=1)
        # axb.plot(self.t_new, dver_z2, c="purple", ls="-", label="z={:.1f} km".format(alts_airglow[iz2]), lw=1)
        # axb.set_title("dVER")
        axb.legend(loc=2, frameon=False)
        axb.set_ylabel("dVER/VER / [%]")
        #axb.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
        axb.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ####################################################################################
        ### VERTICAL INTEGRATION 
        axbb.plot(self.t_new, I_dver_nightglow/self.I_background_nightglow*100, c="k", ls="-", lw=1)
        axbb2 = axbb.twinx()
        axbb2.plot(self.t_new, I_dver_nightglow, ls="")#, c="b", ls="--", lw=1)
        axbb.set_ylabel("Intensity pert.\n" + r"[% background]")
        axbb2.set_ylabel(r"Intensity pert. / [$R$]")
        axbb.set_xlabel(r"Time / [$s$]")
        axbb.set_title("Integration, vertical line of sight")
        # axbb.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
        # axbb2.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
        axbb.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        axbb2.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        for ax in [axt, axm, axb, axbb]:
            # ax.set_xlim(self.t_new.min(), self.t_new.max())
            ax.set_xlim(self.t_new.min(), time_end)
        ###
        fig.suptitle(r"Calculation of $1.27 \mu m$ Nightglow")
        fig.align_labels()
        fig.tight_layout()
        fig.savefig("./Figures/Seismic_to_Nightglow_ie_{:d}_in_{:d}.png".format(i_east, i_north), dpi=300)
        

    def plot_dayglow_traces(self, loc_plot = None, idx_plot = None, z1=92, z2=112, time_end=2000, dir_save="results/"):

        ### Convert east and north coordinate to indices if needed 
        if loc_plot is None and idx_plot is None:
            raise("Requires a couple of coordinates (loc_plot) or a couple of indices (idx_plot).")
        elif loc_plot is not None:
            idx = np.argmin(np.sqrt((self.NN-loc_plot[1])**2+(self.EE-loc_plot[0])**2)) 
            i_east, i_north = np.unravel_index(idx, self.EE.shape)
        else:
            i_east, i_north = idx_plot[0], idx_plot[1]
    
        ### Find the right altitude
        alts_airglow = self.z_4_28_calc_m
        iz1 = np.argmin(np.abs(alts_airglow-z1*1e3))
        iz2 = np.argmin(np.abs(alts_airglow-z2*1e3))
        ### Get appropriate signals 
        dver_z_loc = np.load(dir_save + "dayglow_dver_z_{:d}_{:d}.npy".format(i_east, i_north))
        dver_z1 = dver_z_loc[iz1,:]
        dver_z2 = dver_z_loc[iz2,:]
        uz_z_loc = np.load(dir_save + "dayglow_uz_z_{:d}_{:d}.npy".format(i_east, i_north))
        uz_z1 = uz_z_loc[iz1,:]
        uz_z2 = uz_z_loc[iz2,:]
        ### Load integrated intensity and calculate background 
        I_dver_dayglow = np.load(dir_save + "dayglow_I_{:d}_{:d}.npy".format(i_east, i_north))

        ####################################################################################
        fig, (axt,axm, axb, axbb) = plt.subplots(4,1,figsize=(8,8) )
        axt.plot(self.t_new, self.DIS[i_east, i_north,:], c="k", lw=1)
        axt.set_title("Ground signal at {:.0f} km E, {:.0f} km N".format(self.EE[i_east,i_north]/1e3,self.NN[i_east,i_north]/1e3))
        axt.set_ylabel(r"Disp. / [$m$]")
        # axt.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
        axt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ###
        axm.plot(self.t_new, uz_z1, c="navy", lw=1, label="z={:.1f} km".format(alts_airglow[iz1]/1e3))
        axm.plot(self.t_new, uz_z2, c="purple", lw=1, label="z={:.1f} km".format(alts_airglow[iz2]/1e3))
        axm.set_title("Surface signal")
        axm.set_title("Amplified, propagated, attenuated")
        axm.set_ylabel(r"Disp. / [$m$]")
        axm.legend(loc=2, frameon=False)
        # axm.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
        axm.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ####################################################################################
        axb.plot(self.t_new, dver_z1/self.f_VER_1_27(alts_airglow[iz1])*100, c="navy", ls="-", label="z={:.1f} km".format(alts_airglow[iz1]/1e3), lw=1)
        axb.plot(self.t_new, dver_z2/self.f_VER_1_27(alts_airglow[iz2])*100, c="purple", ls="-", label="z={:.1f} km".format(alts_airglow[iz2]/1e3), lw=1)
        axb.set_title("Percentage of perturbation of VER")

        # axb.plot(self.t_new, dver_z1, c="navy", ls="-", label="z={:.1f} km".format(alts_airglow[iz1]), lw=1)
        # axb.plot(self.t_new, dver_z2, c="purple", ls="-", label="z={:.1f} km".format(alts_airglow[iz2]), lw=1)
        # axb.set_title("dVER")
        axb.legend(loc=2, frameon=False)
        axb.set_ylabel("dVER/VER / [%]")
        # axb.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
        axb.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ####################################################################################
        ### VERTICAL INTEGRATION 
        axbb.plot(self.t_new, I_dver_dayglow/self.I_background_dayglow*100, c="k", ls="-", lw=1)
        axbb2 = axbb.twinx()
        axbb2.plot(self.t_new, I_dver_dayglow, ls="")#, c="b", ls="--", lw=1)
        axbb.set_ylabel("Intensity pert.\n" + r"[% background]")
        axbb2.set_ylabel(r"Intensity pert. / [$R$]")
        axbb.set_xlabel(r"Time / [$s$]")
        axbb.set_title("Integration, vertical line of sight")
        # axbb.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
        # axbb2.get_yaxis().set_major_formatter(ticker.FuncFormatter(scientific_10))
        axbb.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        axbb2.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        for ax in [axt, axm, axb, axbb]:
            # ax.set_xlim(self.t_new.min(), self.t_new.max())
            ax.set_xlim(self.t_new.min(), time_end)
        ###
        fig.suptitle(r"Calculation of $4.28 \mu m$ Dayglow")
        fig.align_labels()
        fig.tight_layout()
        fig.savefig("./Figures/Seismic_to_Dayglow_ie_{:d}_in_{:d}.png".format(i_east, i_north), dpi=300)
        

    def plot_dVER_traces(self, loc_plot = None, idx_plot = None, time_end=2000, dir_save="results/"):

        ### Convert east and north coordinate to indices if needed 
        if loc_plot is None and idx_plot is None:
            raise("Requires a couple of coordinates (loc_plot) or a couple of indices (idx_plot).")
        elif loc_plot is not None:
            idx = np.argmin(np.sqrt((self.NN-loc_plot[1])**2+(self.EE-loc_plot[0])**2)) 
            i_east, i_north = np.unravel_index(idx, self.EE.shape)
        else:
            i_east, i_north = idx_plot[0], idx_plot[1]
    
        ### Recover altitude 
        alts_airglow = self.z_1_27_calc_m

        ### Calculate all intensities 
        dver_z_loc = np.load(dir_save + "nightlow_dver_z_{:d}_{:d}.npy".format(i_east, i_north))
        maxv = np.max(np.abs(dver_z_loc))
        dz = np.diff(alts_airglow)[0]  ### in meters, always 
        I = integrate.cumulative_trapezoid(dver_z_loc, alts_airglow , axis=0, initial=0)
        maxI = np.max(np.abs(I))

        ####################################################################################
        ### PLOT VER(z) at all altitudes 
        fig2, (axm,axi) = plt.subplots(1,2,figsize=(10,6) )
        norm = dz*5/1e3 ### Makes the waveform a little taller than one altitude separation
        for iz, z in enumerate(alts_airglow):
            axm.plot(self.t_new, dver_z_loc[iz,:]/maxv*norm + z/1e3, c="k", lw=1, label="z={:.1f} km".format(alts_airglow[iz]/1e3))
            axi.plot(self.t_new, I[iz,:]/maxI*norm + z/1e3, c="k", lw=1, label="z={:.1f} km".format(alts_airglow[iz]/1e3))
        
        axi.plot(self.t_new, I[iz,:]/maxI*norm + z/1e3, c="r", lw=2, label="z={:.1f} km".format(alts_airglow[iz]/1e3))
        
        axm.set_title("Perturbation of VER")
        axi.set_title("Integrated intensity")
        # axm.legend(loc=2, frameon=False)
        axm.set_ylabel("VER / [scaled to max waveform]")
        axi.set_ylabel("Progressively summing Intensity")
        axm.set_xlabel(r"Time / [$s$]")
        axi.set_xlabel(r"Time / [$s$]")
        ### VERTICAL INTEGRATION 
        # axm.set_xlim(self.t_new.min(), self.t_new.max())
        axm.set_xlim(self.t_new.min(), time_end)
        axi.set_xlim(self.t_new.min(), time_end)
        ###
        fig2.align_labels()
        fig2.tight_layout()
        # fig2.savefig("./Figures/VER_z.png")
    
    
    def plot_arrival_times(self, ax, t_p, t_s, t_rs, atmo_time=0, c="w"):
        ### atmo_time: Time for propagation towards airglow layer 

        ### PLOT P
        ax.axvline(t_p+atmo_time , color=c, ls="--")
        ax.text(t_p+atmo_time , 0.02, "P", color='black', ha='right', va='bottom',
                transform=ax.get_xaxis_transform(),
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='square,pad=0.2'))
        ### PLOT S 
        ax.axvline(t_s+atmo_time , color=c, ls="--")
        ax.text(t_s+atmo_time , 0.02, "S", color='black', ha='right', va='bottom',
                transform=ax.get_xaxis_transform(),
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='square,pad=0.2'))
        ### PLOT RW FUNDAMENTAL 
        ax.plot(t_rs[0][1]+atmo_time ,1/t_rs[0][0], color=c, ls="--")
        ax.text(t_rs[0][1][-1]+atmo_time , max(1/t_rs[0][0][-1], 1e-3), "RW, fund.", color='black', ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='square,pad=0.2'))
        ### PLOT ORDER 1
        ax.plot(t_rs[1][1]+atmo_time ,1/t_rs[1][0], color=c, ls="-.")
        ax.text(t_rs[1][1][-1]+atmo_time , max(1/t_rs[1][0][-1], 1e-3), "RW, 1", color='black', ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='square,pad=0.2'))
        ### PLOT ORDER 2
        ax.plot(t_rs[2][1]+atmo_time ,1/t_rs[2][0], color=c, ls=":")
        ax.text(t_rs[2][1][-1]+atmo_time , max(1/t_rs[2][0][-1], 1e-3), "RW, 2", color='black', ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='square,pad=0.2'))
        return()
        

    def plot_FTAN(self, loc_plot = None, idx_plot = None, fmin=1e-4, it=-1, dir_save ="results/", typeglow="nightglow"):
        ### Convert east and north coordinate to indices if needed 
        if loc_plot is None and idx_plot is None:
            raise("Requires a couple of coordinates (loc_plot) or a couple of indices (idx_plot).")
        elif loc_plot is not None:
            idx = np.argmin(np.sqrt((self.NN-loc_plot[1])**2+(self.EE-loc_plot[0])**2)) 
            i_east, i_north = np.unravel_index(idx, self.EE.shape)
        else:
            i_east, i_north = idx_plot[0], idx_plot[1]
    
        ### Radial distance for arrival time calculations: 
        rr = np.sqrt(self.EE[i_east,i_north]**2 + self.NN[i_east,i_north]**2)

        ### Find the right altitude
        # alts_airglow = self.z_1_27_calc_m   ### Always in meter 
        # iz1 = np.where( np.abs(alts_airglow-z1*1e3) == np.min(np.abs(alts_airglow-z1*1e3)))[0][0]
        # iz2 = np.where( np.abs(alts_airglow-z2*1e3) == np.min(np.abs(alts_airglow-z2*1e3)))[0][0]

        ### Get appropriate signals 
        if typeglow == "nightglow":
            # iz1 = np.argmin(self.z_1_27_calc_m)
            # iz2 = np.argmax(self.z_1_27_calc_m)
            # print(iz1, iz2)
            # vz_z_loc = np.load(dir_save + "{:0}_vz_z_{:d}_{:d}.npy".format(typeglow, i_east, i_north))
            # vz_z1 = vz_z_loc[iz1,:]
            # vz_z2 = vz_z_loc[iz2,:]

            ### Calculate theoretical arrival times
            t_p, t_s, t_rs = theoretical_arrival_times(rr,self.depth)
            air_travel_time = self.dz_1_27_m * np.cumsum(1/self.f_c(self.z_att_1_27_m))
            air_travel_time = air_travel_time[-self.Nz:]
            it1 = np.argmax(air_travel_time)
            it2 = np.argmin(air_travel_time)
        elif typeglow=="dayglow":
            # iz1 = np.argmin(self.z_4_28_calc_m)
            # iz2 = np.argmax(self.z_4_28_calc_m)
            # print(iz1, iz2)
            # uz_z_loc = np.load(dir_save + "{:0}_uz_z_{:d}_{:d}.npy".format(typeglow, i_east, i_north))
            # uz_z1 = uz_z_loc[iz1,:]
            # uz_z2 = uz_z_loc[iz2,:]
            
            ### Calculate theoretical arrival times
            t_p, t_s, t_rs = theoretical_arrival_times(rr,self.depth)
            it2 = np.argmax(self.f_VER_4_28(self.z_4_28_calc_m))
            air_travel_time = self.dz_4_28_m * np.cumsum(1/self.f_c(self.z_att_4_28_m))
            air_travel_time = air_travel_time[-self.Nz:]
            it1 = np.argmax(air_travel_time)
        # dver_z_loc = np.load(dir_save + "{:0}_dver_z_{:d}_{:d}.npy".format(typeglow, i_east, i_north))
        # dver_z1 = dver_z_loc[iz1,:]
        # dver_z2 = dver_z_loc[iz2,:]
        ### Load integrated intensity and calculate background 
        I_dver_nightglow = np.load(dir_save + "{:0}_I_{:d}_{:d}.npy".format(typeglow, i_east, i_north))
        
        


        #######################################################
        fig = plt.figure(figsize=(9,6), layout="constrained")
        gs = fig.add_gridspec(2, 2,width_ratios=[60,2])
        ###
        ax = fig.add_subplot(gs[0, 0])
        axc = fig.add_subplot(gs[0, 1])
        vax = fig.add_subplot(gs[1, 0])
        vaxc = fig.add_subplot(gs[1, 1])

        ax.set_title("Ground signal at {:.0f} km E, {:.0f} km N".format(self.EE[i_east,i_north]/1e3,self.NN[i_east,i_north]/1e3))
        if typeglow=="nightglow":
            ax, axc = plot_scalogram(self.VEL[i_east, i_north,:it], self.t_new[:it], ax, ax_cb=axc, title_unit=r'Velocity [$m/s$]', font=10, fmin=fmin)
        elif typeglow=="dayglow":
            ax, axc = plot_scalogram(self.DIS[i_east, i_north,:it], self.t_new[:it], ax, ax_cb=axc, title_unit=r'Displacement [$m$]', font=10, fmin=fmin)
        self.plot_arrival_times(ax, t_p, t_s, t_rs)
        ###
        vax.set_title(f"{typeglow} intensity perturbation, vertical line of sight")
        vax, vaxc = plot_scalogram(I_dver_nightglow[:it] + self.I_background_nightglow*0, self.t_new[:it], vax, ax_cb=vaxc, title_unit=r"Intensity / [$R$]", font=10, fmin=fmin)
        self.plot_arrival_times(vax, t_p, t_s, t_rs, atmo_time=air_travel_time[it1])
        self.plot_arrival_times(vax, t_p, t_s, t_rs, atmo_time=air_travel_time[it2], c="grey")
        if typeglow=="nightglow":
            vax.plot([],[], c="grey",label="to bottom of layer" )
            vax.plot([],[], c="w",label="to top of layer" )
        elif typeglow=="dayglow":
            vax.plot([],[], c="grey",label="to max of layer" )
            vax.plot([],[], c="w",label="to top of layer" )
        leg = vax.legend(loc=1, framealpha=0, edgecolor="none", labelcolor="w")
        # leg.
        #fig.suptitle("") 
        fig.align_labels()

        return()


    def plot_nightglow_images(self, time_save=None, dir_save="results/", typeglow="nightglow"):       

        ### Wavefront (integrated intensity)
        #wf = np.load(dir_save + f"{typeglow}_dver_t.npy")
        wf = np.load(dir_save + f"{typeglow}_I_t.npy")
        ### NOTE: Did not sum the background here
        vmin = np.mean(wf)-0.5*np.std(wf) #-1
        vmax = np.mean(wf)+0.5*np.std(wf) # 1

        if time_save ==None: 
            time_save = self.t_new[::int(np.ceil(self.Nt//9))]
            itime_save = range(0,self.Nt,int(np.ceil(self.Nt//9)))
            
        Nrow = int(np.ceil(len(time_save[1:])/3))
        Ncol = 5
        fig, axes = plt.subplots(ncols=Ncol, nrows=Nrow, 
                                 gridspec_kw= dict(width_ratios=[1,1,1,0.05,0.05]), 
                                 figsize=(8,int(10*(Nrow/Ncol))) )
        

        for ii, (it, tw)  in enumerate(zip(itime_save[1:],time_save[1:])):

            ###
            u = ii%3
            v = ii//3
            ### Changed from ii 
            im = axes[v][u].pcolormesh(self.EE/1e3, self.NN/1e3, wf[:,:,int(it*wf.shape[-1]/self.Nt)], cmap="Greys_r",vmin=vmin, vmax=vmax)
        #                        norm=colors.SymLogNorm(linthresh=np.mean(I), linscale=1,
        #                                              vmin=np.min(I), vmax=np.max(I), base=10))
            ###
            if u==2:
                cbar=fig.colorbar(im, cax = axes[v][4], label=r"I / Rayleighs", fraction=0.3, pad=-60.)
                # axes[v][4].ticklabel_format(style='sci', axis='y', scilimits=(-2,2), useMathText=True)
                # axes[v][4].yaxis.get_offset_text().set_position((5, 1.02))  # (x, y) in axis coordinates
                cbar.formatter.set_powerlimits((-2,2))
                cbar.formatter.set_useMathText(True)
                axes[v][3].axis("off")
            if u ==0 and v==1:
                axes[v][u].set_ylabel(r"North distance / [$km$]")
            if v==Nrow-1 and u==1:
                axes[v][u].set_xlabel(r"East distance / [$km$]")
            axes[v][u].set_aspect('equal', adjustable="box")
            axes[v][u].text(0.02, 0.98, "{:.0f} s".format(tw),
                            transform=axes[v][u].transAxes,
                            fontsize=8,
                            verticalalignment='top',
                            horizontalalignment='left',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='square,pad=0.3'))

            axes[u][v].tick_params(axis='both', which='major', labelsize=8)

        #fig.suptitle("Vertical velocity")
        axes[0][1].set_title("Vertically integrated VER: " + typeglow, pad=20)
        fig.subplots_adjust(wspace=-0, hspace=0.3, right=0.85, left =0.05, top=0.93)
        fig.savefig("./Figures/Image_" + typeglow + ".png", dpi=300)


    def make_movie(self, dir_save = "./results/"):

        I_night = np.load(dir_save + "nightglow_I_t.npy")
        I_day   = np.load(dir_save + "dayglow_I_t.npy")
        vmin_night = np.mean(I_night)-1*np.std(I_night) #-1
        vmax_night = np.mean(I_night)+1*np.std(I_night) # 1
        vmin_day = np.mean(I_day)-1*np.std(I_day) #-1
        vmax_day = np.mean(I_day)+1*np.std(I_day) # 1
        vmin_vel = np.mean(self.VEL)-1*np.std(self.VEL) #-1
        vmax_vel = np.mean(self.VEL)+1*np.std(self.VEL) # 1

        def plot_frame_3panel(it):

            # Load three data arrays
            VEL = self.VEL[:, :, it]
            I_night_t = I_night[:, :, it]
            I_day_t   = I_day[:, :, it]

            # Pack all in a list
            datasets = [VEL, I_night_t, I_day_t]
            titles = ["Seismic Velocity", "Nightglow Intensity", "Dayglow Intensity"]
            cmaps = ["Greys_r", "Greys_r", "Greys_r"]  # Adjust colormaps as needed
            units = ["m/s", "Rayleighs", "Rayleighs"]
            vmins = [vmin_vel, vmin_night, vmin_day]
            vmaxs = [vmax_vel, vmax_night, vmax_day]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            axes[1].set_yticklabels([])
            axes[2].set_yticklabels([])

            for i, (data, title, cmap, unit, vmin, vmax) in enumerate(zip(datasets, titles, cmaps, units, vmins, vmaxs)):
                ax = axes[i]

                im = ax.pcolormesh(self.EE / 1e3, self.NN / 1e3, data, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=10)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("East / [km]")
                if i == 0:
                    ax.set_ylabel("North / [km]")

                cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
                cbar.set_label(unit, fontsize=10)
                cbar.ax.tick_params(labelsize=10)
                cbar.formatter.set_powerlimits((-2, 2))
                cbar.formatter.set_useMathText(True)

            fig.suptitle(f"t = {self.t_new[it]:.1f} s", fontsize=12)

            fig.savefig(f"{dir_save}/movieframes/frame_{it:04d}.png", dpi=150)
            plt.close(fig)
        
        # plot_frame_3panel(100)
        for it in tqdm(range(self.Nt), total=self.Nt, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}' ) :
            plot_frame_3panel(it)

        # ffmpeg -framerate 20 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 23 airglow_movie.mp4


    
    def plot_vertical_slice(self, time_save=None, wtype="VER"):       

        ### Wavefront (integrated intensity)
        if wtype=="VER":
            wf = np.load("./results/dver_t.npy")[:,:,:,:,1]  ### Select dver 
            vmin = -1e9#np.mean(wf)-0.5*np.std(wf)
            vmax = 1e9#np.mean(wf)+0.5*np.std(wf)
        elif wtype=="VEL":
            wf = np.load("./results/dver_t.npy")[:,:,:,:,0]  ### Select dver 
            vmin = -1#np.mean(wf)-0.5*np.std(wf)
            vmax = 1#np.mean(wf)+0.5*np.std(wf)
        

        ### We want to plot a slice center on North=0 
        inn = np.where(self.NN==0.0)
        wf = wf[inn]
        EE, ZZ = np.meshgrid(self.EE[0,:], self.z_1_27_calc_km)

        if time_save ==None: 
            time_save = self.t_new[::int(np.ceil(self.Nt//9))]
            itime_save = range(0,self.Nt,int(np.ceil(self.Nt//9)))
            
        Nrow = int(np.ceil(len(time_save[1:])/3))
        Ncol = 5
        fig = plt.figure(figsize=(10, int(8 * (Nrow / Ncol))))#, layout="constrained")
        gs = gridspec.GridSpec(nrows=Nrow, ncols=Ncol, width_ratios=[1]*(Ncol-2) + [0.01,0.05], figure=fig)

        # Create the main axes grid
        axes = np.empty((Nrow, Ncol), dtype=object)
        for i in range(Nrow):
            for j in range(Ncol-2):
                axes[i, j] = fig.add_subplot(gs[i, j])

        # Create the spanning axis on the last column (column 4), spanning rows 0 and 1
        ax_cb = fig.add_subplot(gs[:2, Ncol-1])  # spans both rows, column Ncol (i.e. column 4 if Ncol=4)

                

        for ii, (it, tw)  in enumerate(zip(itime_save[1:],time_save[1:])):

            ###
            u = ii%3
            v = ii//3
            im = axes[v][u].pcolormesh(EE/1e3, ZZ, wf[:,:,ii].T, cmap="Greys_r",#vmin=vmin, vmax=vmax)
                                norm=SymLogNorm(linthresh=np.std(wf)/10, linscale=1,
                                                      vmin=vmin, vmax=vmax, base=10))
            ###
            # if u==2 and v==0:
            #     cbar=fig.colorbar(im, cax = axes[v][4], label=r"$\Delta$VER / [$W/m^3$]", fraction=0.3, pad=-60.)
            #     #cbar.formatter.set_useMathText(True)
            #     axes[v][3].axis("off")
            # axes[1][3].axis("off")
            # axes[2][3].axis("off")
            # axes[1][4].axis("off")
            # axes[2][4].axis("off")
            if wtype=="VER":
                cbar=fig.colorbar(im, cax = ax_cb, label=r"$\Delta$VER / [$W/m^3$]", fraction=0.3, pad=-60.)
            elif wtype =="VEL":
                cbar=fig.colorbar(im, cax = ax_cb, label=r"$v_z$ / [$m/s$]", fraction=0.3, pad=-60.)
            #     
            if u ==0 and v==1:
                axes[v][u].set_ylabel(r"Altitude / [$km$]")
            if u != 0 :
                axes[v][u].set_yticklabels([])
            else:
                axes[v][u].set_yticks([90,100,110,120])
            if v==Nrow-1 and u==1:
                axes[v][u].set_xlabel(r"East distance / [$km$]")
            #axes[v][u].set_aspect('equal', adjustable="box")
            axes[v][u].text(0.02, 0.98, "{:.0f} s".format(tw),
                            transform=axes[v][u].transAxes,
                            fontsize=8,
                            verticalalignment='top',
                            horizontalalignment='left',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='square,pad=0.3'))

            axes[u][v].tick_params(axis='both', which='major', labelsize=8)

        #fig.suptitle("Vertical velocity")
        if wtype=="VER":
            axes[0][1].set_title("VER Perturbation with altitude", pad=20)
        elif wtype=="VEL":
            axes[0][1].set_title("Velocity Perturbation with altitude", pad=20)
        fig.subplots_adjust(wspace=0.2, hspace=0.5, right=0.85, left =0.05, top=0.85, bottom=0.15)


    def plot_phase_velocity_extraction(self, method=3, file_model='./data/Cold_100_for_QSSP.csv', n_modes=6, plot_wf=False):

        ### Load the Intensity matrix 
        I_matrix = np.load("./results/I_t.npy")

        ### Cut on a line 
        north_index = I_matrix.shape[1]//2
        I_east = I_matrix[north_index,:,:]
        dist_east = self.EE[north_index,:]/1e3
        dE = np.diff(self.EE[north_index,:])[0]
        Nw = I_east.shape[0]

        ### PLOT SIGNALS WITH DISTANCE #####################################################
        if plot_wf:
            fig, ax = plt.subplots()
            for i in range(Nw):
                wv = I_east[i,:]
                wv/=np.max(np.abs(wv))
                #ax.plot(AIRGLOW.t_new,wv*dE/1e3-AIRGLOW.EE[north_index,i]/1e3, c="k", lw=1)
                ax.fill_between(self.t_new,-self.EE[north_index,i]/1e3,wv*dE/1e3-self.EE[north_index,i]/1e3,
                                color='k',alpha=1, lw=0.5)
            ax.set_xlabel("Time / [$s$]")
            ax.set_ylabel("Distance / [$km$]")
            fig.tight_layout()
        ####################################################################################


        #####################################################################################
        ### Calculate theoretical phase velocity ############################################
        from disba import PhaseDispersion, GroupDispersion
        layers = pd.read_csv(file_model, delim_whitespace=True, 
                                header=None, names=['z','vp','vs','rho','Qp','Qs'])  #skiprows=2, 
        layers = layers[:]
        h = np.diff(layers.z)
        layers = layers.iloc[1:]
        layers['h'] = h
        layers = layers.iloc[:]
        velocity_model = layers.loc[:,['h','vp','vs','rho']].values

        ### frequencies (Hz) or periods (s) 
        ff = 10**np.linspace(-3,0, 100)[::-1]
        T = 1 / ff                                ### disba wants periods, low→high

        ### Rayleigh‑wave phase velocity, fundamental mode (mode 0) ---
        #group_disp = GroupDispersion(*velocity_model[:66,:].T)               # unpack into h, vp, vs, ρ
        phase_disp = PhaseDispersion(*velocity_model[:66,:].T)                # unpack into h, vp, vs, ρ
        rayleigh_i = [phase_disp(T, mode=i, wave="rayleigh") for i in range(n_modes)]   # namedtuple
        v_r_i = [rayleigh_i[i].velocity*1e3 for i in range(n_modes)]
        #######################################################################################

        #######################################################################################
        ### Calculate spectrum 
        FFTsig = sfft.fft(I_east, axis=1)
        freqsi = np.fft.fftfreq(self.Nt, d=self.dt)  # frequency axis
        mask = freqsi > 0
        freqs = freqsi[mask]

        ### OPTION 1 ########################################################################
        ### CROSS-SPECTRUM METHOD 
        if method==1:
            ### Calculate phase delay: 
            ic = (Nw)//2
            lags = np.arange(-self.Nt + 1, self.Nt)*self.dt
            ### Calculate spectrum 
            ###
            ### Cross-spectrum every step
            CC = FFTsig[1:,:] * np.conj(FFTsig[:-1,:])
            ### Calculate Cross spectrum from reference pixel to others 
            idx = np.arange(0,Nw,1)
            iref = 10                           ### Reference point
            ical = np.delete(idx, iref)         ### All the others
            CC2 = FFTsig[iref:iref+1,:] * np.conj(FFTsig[ical,:])
            dist_c2 = dist_east[ical]-dist_east[iref]  ### Distance to reference point 

            ### Phase difference at each frequency
            phase_spectrum = np.array([np.angle(CC[i,:]) for i in range(Nw-1)])  # in radians
            phase_spectrum2 = np.array([np.angle(CC2[i,:]) for i in range(Nw-1)])  # in radians
            ### Unwrap phase 
            unwrapped_phase = np.unwrap(phase_spectrum, axis=1)
            unwrapped_phase2 = np.unwrap(phase_spectrum2, axis=1)

            # Keep only positive frequencies
            phase_spectrum = phase_spectrum[:,mask]
            phase_spectrum2 = phase_spectrum2[:,mask]
            unwrapped_phase = unwrapped_phase[:,mask]
            unwrapped_phase2 = unwrapped_phase2[:,mask]

            ### Calculate phase velocity
            phase_vel = 2*np.pi*freqs[None,:]*dE/np.abs(unwrapped_phase)
            phase_vel2 = 2*np.pi*freqs[None,:]*dist_c2[:,None]*1e3/np.abs(unwrapped_phase2)

            ### PLOT METHOD 1 #####################################################################
            fig, (ax1,ax2,ax3) = plt.subplots(3,1,height_ratios=[0.3,1,1], figsize=(8,12)) 
            #ax.plot(freqs, phase_spectrum) 
            for i in range(Nw-1):
                ax1.plot(freqs, unwrapped_phase[i,:], c="k", lw=1,alpha=0.2, ) 
            ax1.set_ylabel("Unwrapped phase / [$rad.$]")
            ###
            #Close=[4,6,3]
            cols = plt.get_cmap("viridis")
            for i in range(Nw-1):#Close:
                ax2.plot(freqs, phase_vel[i-1,:], c=cols(i/(Nw-1)), lw=1, alpha=0.2, 
                        label="CC {:.0f}-{:.0f} km".format(dist_east[i], dist_east[i+1]))
                ax3.plot(freqs, phase_vel2[i-1,:], c=cols(i/(Nw-1)), lw=1, alpha=0.2, 
                        label="CC {:.0f} km".format(dist_c2[i]))
            ###
            cmap=plt.get_cmap("magma")
            for i in range(n_modes):
                ax2.plot(1/rayleigh_i[i].period, v_r_i[i], c=cmap(i/n_modes), ls="--", label=r"Theoretical $v_{\varphi}$"+ "{:d}".format(i))
                ax3.plot(1/rayleigh_i[i].period, v_r_i[i], c=cmap(i/n_modes), ls="--", label=r"Theoretical $v_{\varphi}$"+ "{:d}".format(i))
            ax2.legend(framealpha=1, edgecolor="none")
            ax3.legend(framealpha=1, edgecolor="none")
            ###
            ax1.set_xscale("log")
            ax2.set_xscale("log")
            ax3.set_xscale("log")
            ax2.set_ylabel("Phase Velocity / [$m/s$]")
            ax2.set_xlabel("Frequency / [$Hz$]")
            ax3.set_ylabel("Phase Velocity / [$m/s$]")
            ax3.set_xlabel("Frequency / [$Hz$]")
            ax1.set_xlim(ff.min(), ff.max())
            ax2.set_xlim(ff.min(), ff.max())
            ax2.set_ylim(2e3,8e3)
            ax3.set_xlim(ff.min(), ff.max())
            ax3.set_ylim(2e3,8e3)
            fig.tight_layout()
            fig.align_labels()

        ### OPTION 2 ########################################################################
        ### TAU-P METHOD 
        elif method==2:
            ### First, construct an array of slowness p:
            Np = 200
            parr = 1/np.linspace(12,0.1, Np)
            ### Construct an linear interpolation function for Intensity in time : 
            f_Ieast = interpolate.interp1d(self.t_new, I_east, kind='linear', axis=1, bounds_error=False, fill_value=0)
            ### Next, calculate slant integral over line.
            TP = np.zeros((self.Nt,Np)) 
            for ip, p in enumerate(parr):
                t_eval = self.t_new[None,:] + dist_east[:,None]*p
                ### Sum over waves 
                slant = np.array([f_Ieast(t_eval[i,:])[i,:] for i in range(Nw)])
                #print(slant.shape)
                TP[:,ip] = np.trapz(slant, dist_east*1e3, axis=0)
            ### Take fourier transform:
            WP = sfft.fft(TP, axis=0)   
            ### PLOT METHOD 2 #############################################################################
            ### Convert P to velocity and plot 
            figs, axs = plt.subplots(figsize=(8,6))
            A = np.abs(WP[mask,:].T)
            A = A/np.max(A, axis=0)[None,:]
            axs.pcolormesh(freqs,1/parr, A, cmap="magma")
            ###
            cmap=plt.get_cmap("magma")
            lss = ["-", "--", "-.", ":","--", "-.", ":" ]
            for i in range(n_modes):
                axs.plot(1/rayleigh_i[i].period, v_r_i[i]/1e3, c="w", lw = 1, ls=lss[i])
                axs.plot([],[], c="k", ls=lss[i], lw=1.5, label=r"Theoretical $v_{\varphi}$"+ "{:d}".format(i))
            axs.legend(framealpha=1, edgecolor="none")
            ###
            axs.set_xlabel("Frequency / [$Hz$]")
            axs.set_ylabel("Velocity / [$km/s$]")
            axs.set_xscale("log")
            axs.set_xlim(freqs.min(), 1)#freqs.max())
            figs.tight_layout()      

        ### OPTION 3 ########################################################################
        ### TAU-P IN FREQUENCY DOMAIN 
        elif method==3:
            ### First, construct an array of velocity c:
            Nc = 200
            carr = np.linspace(0.1,12, Nc)
            ### Take fourier transform of signal with distance
            ###
            ### Next, calculate slant integral over line.
            WC = np.zeros((self.Nt,Nc)) 
            for icc, c in enumerate(carr):
                phase_shift = np.exp(1j*2*np.pi*freqsi[None,:]*dist_east[:,None]/c)
                ### Sum over waves 
                shifted = FFTsig  * phase_shift
                ### Normalize by complex amplitude
                shifted = shifted/np.abs(shifted)
                #print(slant.shape)
                #WC[:,ic] = np.trapz(shifted, dist_east*1e3, axis=0)
                WC[:,icc] = np.abs(np.sum(shifted, axis=0)*dE)  
            
            ### PLOT METHOD 3 #############################################################################
            figw, axw = plt.subplots(figsize=(8,6))
            A = np.abs(WC[mask,:].T)
            A = A/np.max(A, axis=0)[None,:]
            axw.pcolormesh(freqs,carr, A, cmap="magma")
            ###
            cmap=plt.get_cmap("magma")
            lss = ["-", "--", "-.", ":","--", "-.", ":" ]
            for i in range(n_modes):
                axw.plot(1/rayleigh_i[i].period, v_r_i[i]/1e3, c="w", lw = 1, ls=lss[i])
                axw.plot([],[], c="k", ls=lss[i], lw=1.5, label=r"Theoretical $v_{\varphi}$"+ "{:d}".format(i))
            axw.legend(framealpha=1, edgecolor="none")
            ###
            axw.set_xlabel("Frequency / [$Hz$]")
            axw.set_ylabel("Velocity / [$km/s$]")
            axw.set_xscale("log")
            axw.set_xlim(freqs.min(), 1)#freqs.max())
            figw.tight_layout()



# =========================================================================================================
### AIRGLOW INTENSITY SCALER 
# =========================================================================================================
def compute_airglow_scaler_new(mw=None, strike=45, dip=45, rake=45, do_plot=True, effect=None, tit ="", 
                               store_ids_dists = [('GF_venus_Cold100_atten_qssp_nearfield',0e3,50e3),('GF_venus_Cold100_atten_qssp',50e3,8000e3)]):
    '''
    We calculate airglow signals for a series of receiver locations and source depths. 
    '''

    ### First, define the grid of locations. 
    gridded       = True
    min_grid_dist = 0e3
    max_grid_dist = 5000e3 # 4000e3
    delta_dist    = 50e3 
    Np = 1+ int(2*max_grid_dist/delta_dist)
    north_shifts  = np.linspace(-max_grid_dist, max_grid_dist, Np, endpoint=True)[::10]
    east_shifts   = np.linspace(-max_grid_dist, max_grid_dist, Np, endpoint=True)[::10]
    Ntr = east_shifts.size*north_shifts.size 
    
    ### Source depths 
    delta_depth = 5e3
    depths = np.arange(5e3, 50e3+delta_depth, delta_depth)

    ### Option for Pyrocko 
    opt_synthetics = dict(
        ### Options for source 
        mw = mw,               ### if none: We only get the Green's function
        depth = depths[5],     ### Only one depth
        strike = strike,       ### Default mechanism 
        dip =  dip, 
        rake = rake,
        stf_type = None,       ### Dirac source 
        #stf_type = 'triangle', 
        # stf_type = 'sinus', 
        #effective_duration = 25.,
        ###  
        ### Options for store
        base_folder='/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/',
        ### Old option, single store 
        #store_id = 'GF_venus_Cold100_qssp',
        #store_id = 'GF_venus_Cold100_qssp_grid',
        ### Give store names, and min and max valid distance 
        store_ids_dists = store_ids_dists,
        ###
        ### Options for grid 
        north_shifts = north_shifts, 
        east_shifts = east_shifts,
        gridded=gridded
    )

    ### INITIALIZE SEISMOGRAM CLASS 
    ### we build seismograms over grid
    ### 
    SEISMO = Seismograms(**opt_synthetics)
    print("Total grid size: ", SEISMO.NN.size)

    ### OPTIONAL: Plot to check 
    ns, es       = 1000e3, 0e3
    ### Plot one of the waveforms for check
    # fig1 = SEISMO.plot_traces(ns, es, do_interpolate=True)

    ### Store results inside a regular grid. 
    dt_airglow = 0.5
    SEISMO.arrange_interpolate_synthetics(tmax=3000, dt=dt_airglow)

    ### Normalize all the velocity traces to 1: 
    mvel = np.max(np.abs(SEISMO.VEL), axis=2)
    SEISMO.VEL/=mvel[:,:,None] 
    SEISMO.DIS/=mvel[:,:,None]

    ### OPTIONAL: Plot velocity wavefront at the surface for sanity check 
    # fig2 = SEISMO.plot_wavefront()

    ### Now, we will calculate the AIRGLOW at every loc of the grid
    ### We first load the airglow class
    AIRGLOW = AirglowSignal(SEISMO, Nz=500)

    ### Now we compute the AIRGLOW at all locations and timesteps. 
    ### NOTE : This can be quite heavy ! 
    ### List of all north and east indices:
    dir_save="./results_detectability/"
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    list_inorth, list_ieast = AIRGLOW.iNN, AIRGLOW.iEE

    def airglow_scaler_calculation(AIRGLOW):
        ### Calculation of the Nightglow
        AIRGLOW.calculate_1_27_airglow(list_ieast, list_inorth, loc_save_idx=[],
                                    do_parallel=True, 
                                    fourier_filtering=False,   ### Use time filtering 
                                    dir_save = dir_save,
                                    save_ver = False,          ### Faster if we save only I(lat, lon, t)
                                    time_save = AIRGLOW.t_new) ### Save all timesteps 
        ### Calculation of the Dayglow
        AIRGLOW.calculate_4_28_airglow(list_ieast, list_inorth, loc_save_idx=[],
                                    do_parallel=True, 
                                    dir_save=dir_save, 
                                    save_ver = False,          ### Faster if we save only I(lat, lon, t)
                                    time_save = AIRGLOW.t_new) ### Save all timesteps 

        I_nightglow = np.load(dir_save + "nightglow_I_t.npy")
        I_dayglow = np.load(dir_save + "dayglow_I_t.npy")
        
        ### ==========================================================================================================
        ### FIGURE: We make some frequency bins 
        # freq_bins = np.logspace(np.log10(1e-3), np.log10(5e-1), 5)
        fmean = [10**-3, 10**-2, 10**-1, 10**0]
        freq_bins = [None, 10**-2.5, 10**-1.5, 10**-0.5, None]  ### Centered around 1e-2, 1e-1, 1. 
        f_targets = []
        for _, (binleft, binright) in enumerate(zip(freq_bins[:-1], freq_bins[1:])):
            f_targets += [[binleft, binright]]
        print(" Filter bins: ", f_targets)

        scaling_airglow = pd.DataFrame()
        ### We loop over locations and store the max amplitude in a dataframe: 
        for f1, f2 in tqdm(f_targets, disable=True):

            ### To scale with the velocity amplitude in each freq, band 
            waveform_vel_filt = butter_filter(AIRGLOW.VEL, 1/dt_airglow, f1,f2, order=5, axis=2)
            
            waveform_nightglow_filt = butter_filter(I_nightglow, 1/dt_airglow, f1,f2, order=5, axis=2)
            perturb_nightglow_filt = waveform_nightglow_filt/AIRGLOW.I_background_nightglow/\
                                    np.max(abs(waveform_vel_filt), axis=2)[:,:,None] * 100

            waveform_dayglow_filt = butter_filter(I_dayglow, 1/dt_airglow, f1,f2, order=5, axis=2)
            perturb_dayglow_filt = waveform_dayglow_filt/AIRGLOW.I_background_dayglow/\
                                    np.max(abs(waveform_vel_filt), axis=2)[:,:,None] * 100

            for (ies, ins) in zip(AIRGLOW.iEE.ravel(), AIRGLOW.iNN.ravel()):
                es, ns = AIRGLOW.EE[ies, ins], AIRGLOW.NN[ies,ins]
                loc_dict = dict(ns=ns, es=es, 
                                f1=f1 if f1 is not None else 0, 
                                f2=f2 if f2 is not None else 1., 
                                nightglow=abs(perturb_nightglow_filt[ies, ins,:]).max(),
                                dayglow=abs(perturb_dayglow_filt[ies, ins,:]).max())
                # dayglow=abs(waveform_dayglow).max()
                scaling_airglow = pd.concat([scaling_airglow, pd.DataFrame([loc_dict])])

        ### Calculate statistics 
        scaling_nightglow_plot = scaling_airglow.groupby(['f1', 'f2',])['nightglow'].median().reset_index()
        scaling_nightglow_plot['nightglow_q25'] = scaling_airglow.groupby(['f1', 'f2',])['nightglow'].quantile(q=0.25).reset_index()['nightglow']
        scaling_nightglow_plot['nightglow_q75'] = scaling_airglow.groupby(['f1', 'f2',])['nightglow'].quantile(q=0.75).reset_index()['nightglow']
        ###
        scaling_dayglow_plot = scaling_airglow.groupby(['f1', 'f2',])['dayglow'].median().reset_index()
        scaling_dayglow_plot['dayglow_q25'] = scaling_airglow.groupby(['f1', 'f2',])['dayglow'].quantile(q=0.25).reset_index()['dayglow']
        scaling_dayglow_plot['dayglow_q75'] = scaling_airglow.groupby(['f1', 'f2',])['dayglow'].quantile(q=0.75).reset_index()['dayglow']

        scaling_nightglow_plot.to_csv(dir_save + "nightglow_scaler"+tit + ".csv", header=True, index=False)
        scaling_dayglow_plot.to_csv(dir_save + "dayglow_scaler"+tit + ".csv", header=True, index=False)

        return(fmean, scaling_nightglow_plot, scaling_dayglow_plot)
    

    def plot_scaler(fmean, scaling_nightglow_plot, scaling_dayglow_plot):
        fig, ax = plt.subplots()

        ax.plot(fmean, scaling_nightglow_plot.nightglow, 
                color='forestgreen', marker="s", label=r"1.27$\mu m$ Nightglow")
        ax.fill_between(fmean, 
                        scaling_nightglow_plot.nightglow_q25, scaling_nightglow_plot.nightglow_q75,
                        color='forestgreen', alpha=0.3)
        ax.plot(fmean, scaling_dayglow_plot.dayglow, 
                color='orangered', marker="s", label=r"4.28$\mu m$ Dayglow")
        ax.fill_between(fmean, 
                        scaling_dayglow_plot.dayglow_q25, scaling_dayglow_plot.dayglow_q75,
                        color='orangered', alpha=0.3)
        
        ax.legend(frameon=False)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel("Frequency / [$Hz$]")
        ax.set_ylabel(r"Airglow Intensity perturbation [$\%/(1\,m/s)$]")
        ax.set_title("Intensity perturbation for 1 $m/s$ peak velocity at the ground surface")
        
        fig.savefig(dir_save + "Airglow_scaler"+ tit + ".png", dpi=300)
    

    def plot_scaler_ampl(fmean, ng, dg, ng_min, dg_min, ng_max, dg_max):
        fig, ax = plt.subplots(figsize=(5.5,4))

        ax.plot(fmean, ng.nightglow, 
                color='forestgreen', marker="s", label=r"1.27$\mu m$ Nightglow")
        ax.fill_between(fmean, ng.nightglow_q25, ng.nightglow_q75,
                        color='forestgreen', alpha=0.3)
        ax.plot(fmean, dg.dayglow, 
                color='orangered', marker="s", label=r"4.28$\mu m$ Dayglow")
        ax.fill_between(fmean, dg.dayglow_q25, dg.dayglow_q75,
                        color='orangered', alpha=0.3)
        
        ax.plot(fmean, ng_min.nightglow, color='grey', marker="v", ls="--", label=r"Minimum Amplification")
        ax.fill_between(fmean, ng_min.nightglow_q25, ng_min.nightglow_q75,color='grey', alpha=0.2)
        ax.plot(fmean, dg_min.dayglow, color='grey', marker="v", ls ="--")
        ax.fill_between(fmean, dg_min.dayglow_q25, dg_min.dayglow_q75,color='grey', alpha=0.2)
        ###
        ax.plot(fmean, ng_max.nightglow, color='grey', marker="^", ls=":", label=r"Maximum Amplification")
        ax.fill_between(fmean, ng_max.nightglow_q25, ng_max.nightglow_q75,color='grey', alpha=0.2)
        ax.plot(fmean, dg_max.dayglow, color='grey', marker="^", ls =":")
        ax.fill_between(fmean, dg_max.dayglow_q25, dg_max.dayglow_q75,color='grey', alpha=0.2)

        ax.legend(frameon=False, loc=3)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel("Frequency / [$Hz$]")
        ax.set_ylabel(r"Airglow Intensity perturbation [$\%/(1\,m/s)$]")
        ax.set_title("Intensity perturbation for 1 $m/s$"+"\npeak velocity at the ground surface")
        fig.tight_layout()
        
        fig.savefig(dir_save + "Airglow_scaler_amplification.png", dpi=300)
        fig.savefig(dir_save + "Airglow_scaler_amplification.pdf")

        
    if effect=="ampl":
        ### We open various atmospheric profiles for density
        dens_files = np.load("/staff/marouchka/Documents/VCD_Results/QB_detectability_data/density_fromtopo.npy")
        soundspeed_files = np.load("/staff/marouchka/Documents/VCD_Results/QB_detectability_data/soundspeed_fromtopo.npy")
        ltst_files = np.load("/staff/marouchka/Documents/VCD_Results/QB_detectability_data/ltst_fromtopo.npy")
        alt_files = np.load("/staff/marouchka/Documents/VCD_Results/QB_detectability_data/altitude_m.npy")
        lat_files = np.load("/staff/marouchka/Documents/VCD_Results/QB_detectability_data/latitude.npy")
        lon_files = np.load("/staff/marouchka/Documents/VCD_Results/QB_detectability_data/longitude.npy")
        shape = dens_files.shape[:2]
        dens_files = dens_files.reshape((-1, dens_files.shape[2]))
        ltst_files = ltst_files.reshape((-1, ltst_files.shape[2]))
        soundspeed_files = soundspeed_files.reshape((-1, soundspeed_files.shape[2]))
        ampl = np.sqrt( (dens_files[:,0][:,None]*soundspeed_files[:,0][:,None])/ (dens_files*soundspeed_files))

        ### ===========================================================
        fig,ax = plt.subplots() 
        for i in range(dens_files.shape[0]):
            #ax.plot(dens_files[i,:],alt_files/1e3, c="k", alpha=0.2)
            ax.plot(ampl[i,:],alt_files/1e3, c="grey", alpha=0.05)
        ax.plot(AIRGLOW.f_amplification(alt_files), alt_files/1e3, c="k", label="Original")
        ax.set_title("")
        ax.set_xlim(1,1e8)
        ax.set_ylim(1,200)
        fig.tight_layout()
        fig.savefig(dir_save + "amplification.png", dpi=300)
        fig.savefig(dir_save + "amplification.pdf")
        ### ===========================================================
        

        imax = np.argmax(ampl[:,-1])
        imin = np.argmin(ampl[:,-1])
        idx_min = np.unravel_index(imin, shape)
        idx_max = np.unravel_index(imax, shape)
        print("Amplification minimum for Lon={:.1f}, Lat={:.1f}, at LTST={:.1f}h".format(lon_files[idx_min[0]], lon_files[idx_min[1]], ltst_files[imin,0] ) )
        print("Amplification maximum for Lon={:.1f}, Lat={:.1f}, at LTST={:.1f}h".format(lon_files[idx_max[0]], lon_files[idx_max[1]], ltst_files[imax,0] ) )
        # Amplification minimum for Lon=30.0, Lat=5.0, at LTST=14.4h
        # Amplification maximum for Lon=-135.0, Lat=-15.0, at LTST=1.4h

        prof_ampl_min = ampl[imin,:]
        prof_ampl_max = ampl[imax,:]
        ax.plot(prof_ampl_max,alt_files/1e3, c="r", label="Max amplification from VCD")
        ax.plot(prof_ampl_min,alt_files/1e3, c="b", label="Min amplification from VCD")
        ###
        ax.legend(frameon=False)
        ax.set_xscale("log")
        ax.set_xlabel("Amplification")
        ax.set_ylabel("Altitude / [$km$]")
        fig.tight_layout()
        
        # np.save("./data/altitude_amplification", alt_files)
        # np.save("./data/amplification_min", prof_ampl_min)
        # np.save("./data/amplification_max", prof_ampl_max)
        # quit()
        ### STORE ORIGINAL AMPLIFICATION 
        fampl = AIRGLOW.f_amplification

        ### Do MIN 
        AIRGLOW.f_amplification = interpolate.interp1d(alt_files, prof_ampl_min, kind='quadratic')
        tit = "ampli_min"
        fmean, scaling_nightglow_plot_ampli_min, scaling_dayglow_plot_ampli_min = airglow_scaler_calculation(AIRGLOW)
        ### Do MAX 
        AIRGLOW.f_amplification = interpolate.interp1d(alt_files, prof_ampl_max, kind='quadratic')
        tit = "ampli_max"
        fmean, scaling_nightglow_plot_ampli_max, scaling_dayglow_plot_ampli_max = airglow_scaler_calculation(AIRGLOW)
        ### Do ORIG 
        AIRGLOW.f_amplification = fampl
        tit = "ampli"
        fmean, scaling_nightglow_plot_ampli, scaling_dayglow_plot_ampli = airglow_scaler_calculation(AIRGLOW)
        
        plot_scaler_ampl(fmean, scaling_nightglow_plot_ampli, scaling_dayglow_plot_ampli, 
                         scaling_nightglow_plot_ampli_min, scaling_dayglow_plot_ampli_min, 
                         scaling_nightglow_plot_ampli_max, scaling_dayglow_plot_ampli_max)
    else:
        fmean, scaling_nightglow_plot, scaling_dayglow_plot = airglow_scaler_calculation(AIRGLOW)
        if do_plot:
            plot_scaler(fmean, scaling_nightglow_plot, scaling_dayglow_plot)

  
    def airglow_scaler_test1(do_fig_spectrum=False):

        ### ==========================================================================================================
        ### OTHER OPTION: Filter velocity before calculating Airglow 
        ### Then calculate the max intensity 
        fmean = [10**-3, 10**-2, 10**-1, 10**0]
        freq_bins = [None, 10**-2.5, 10**-1.5, 10**-0.5, None]  ### Centered around 1e-2, 1e-1, 1. 
        f_targets = []
        for _, (binleft, binright) in enumerate(zip(freq_bins[:-1], freq_bins[1:])):
            f_targets += [[binleft, binright]]
        print(" Filter bins: ", f_targets)
        store_Imax_ng = []
        store_Imax_dg = []
        for f1, f2 in tqdm(f_targets, disable=True):

            ### To scale with the velocity amplitude in each freq, band 
            waveform_vel_filt = butter_filter(AIRGLOW.VEL, 1/dt_airglow, f1,f2, order=5, axis=2).reshape(Ntr,AIRGLOW.Nt)
            
            waveform_nightglow_filt = butter_filter(I_nightglow, 1/dt_airglow, f1,f2, order=5, axis=2).reshape(Ntr,AIRGLOW.Nt)
            sensmax_nightglow_filt = np.max(abs(waveform_nightglow_filt), axis=1)/AIRGLOW.I_background_nightglow/\
                                    np.max(abs(waveform_vel_filt), axis=1) * 100

            waveform_dayglow_filt = butter_filter(I_dayglow, 1/dt_airglow, f1,f2, order=5, axis=2).reshape(Ntr,AIRGLOW.Nt)
            sensmax_dayglow_filt = np.max(abs(waveform_dayglow_filt), axis=1)/AIRGLOW.I_background_dayglow/\
                                    np.max(abs(waveform_vel_filt), axis=1)* 100

            ### TO RECALCULATE USING FILTERED WAVE ======================================
            # VEL_filt = butter_filter(AIRGLOW.VEL, 1/dt_airglow, f1,f2, order=5, axis=2)
            # DIS_filt = butter_filter(AIRGLOW.DIS, 1/dt_airglow, f1,f2, order=5, axis=2)
            
            # ### Normalize all the velocity traces to 1: 
            # mvel = np.max(np.abs(VEL_filt), axis=2)
            # SEISMO.VEL = VEL_filt/mvel[:,:,None] 
            # SEISMO.DIS = DIS_filt/mvel[:,:,None]

            # Vfilt = VEL_filt.reshape(Ntr, AIRGLOW.Nt)
            # Vfilt = Vfilt / mvel.reshape(Ntr)[:,None]

            # AIRGLOW_filt = AirglowSignal(SEISMO, Nz=500)
            # dir_save_filt="./results_detectabilityf/"
            # if not os.path.exists(dir_save_filt):
            #     os.makedirs(dir_save_filt)
            # list_inorth, list_ieast = AIRGLOW_filt.iNN, AIRGLOW_filt.iEE
            # ### Calculation of the Nightglow
            # AIRGLOW_filt.calculate_1_27_airglow(list_ieast, list_inorth, loc_save_idx=[],
            #                                do_parallel=True, 
            #                                fourier_filtering=False,   ### Use time filtering 
            #                                dir_save = dir_save_filt,
            #                                save_ver = False,          ### Faster if we save only I(lat, lon, t)
            #                                time_save = AIRGLOW_filt.t_new) ### Save all timesteps 
            # ### Calculation of the Dayglow
            # AIRGLOW_filt.calculate_4_28_airglow(list_ieast, list_inorth, loc_save_idx=[],
            #                                do_parallel=True, 
            #                                dir_save=dir_save_filt, 
            #                                save_ver = False,          ### Faster if we save only I(lat, lon, t)
            #                                time_save = AIRGLOW_filt.t_new) ### Save all timesteps 
            # ### Now we recover the max intensity 
            # I_nightglow_filt = np.load(dir_save_filt + "nightglow_I_t.npy").reshape(Ntr,AIRGLOW_filt.Nt)
            # I_dayglow_filt = np.load(dir_save_filt + "dayglow_I_t.npy").reshape(Ntr,AIRGLOW_filt.Nt)

            # sensmax_nightglow_filt = np.max(abs(I_nightglow_filt), axis=1)/AIRGLOW_filt.I_background_nightglow/\
            #                          np.max(abs(Vfilt), axis=1) * 100
            # sensmax_dayglow_filt = np.max(abs(I_dayglow_filt), axis=1)/AIRGLOW_filt.I_background_dayglow/\
            #                          np.max(abs(Vfilt), axis=1) * 100

            store_Imax_ng.append(sensmax_nightglow_filt)
            store_Imax_dg.append(sensmax_dayglow_filt)
        store_Imax_ng = np.array(store_Imax_ng)
        store_Imax_dg = np.array(store_Imax_dg)

        if do_fig_spectrum:
            ### ==========================================================================================================
            ### FIGURE 1: Look at the ratio of airglow spectra to velocity spectra 
            fr = np.fft.rfftfreq(n=AIRGLOW.t_new.size, d=dt_airglow)[1:]
            sp = abs(np.fft.rfft(AIRGLOW.VEL, axis=2))[:,:,1:].reshape(Ntr, fr.size)         ### Ground Velocity spectrum
            spin = abs(np.fft.rfft(I_nightglow, axis=2))[:,:,1:].reshape(Ntr, fr.size)       ### Nightglow spectrum
            spid = abs(np.fft.rfft(I_dayglow, axis=2))[:,:,1:].reshape(Ntr, fr.size)         ### Dayglow spectrum 

            fig, ((ax1, ax2, ax3),(ax5,ax4,ax6)) = plt.subplots(2,3, figsize=(13,8)) 
            sens_ng = spin/AIRGLOW.I_background_nightglow/sp*100
            sens_dg = spid/AIRGLOW.I_background_dayglow/sp*100
            ###
            ax1b = ax1.twinx()
            for es in range(Ntr):
                ax1.plot(fr, spin[es,:], c="forestgreen", lw=0.5, alpha=0.01)
                ax1.plot(fr, spid[es,:], c="orangered", lw=0.5, alpha=0.01)
                ax1b.plot(fr, sp[es,:], c="k", lw=0.1, alpha=0.01)
            ax1.plot(fr, np.median(spin, axis=0), c="forestgreen", lw=1.5,label="Nightglow")
            ax1.plot(fr, np.median(spid, axis=0), c="orangered", lw=1.5,label="Dayglow")
            ax1b.plot(fr, np.median(sp, axis=0), c="k", lw=1.5,label="Ground Velocity")
            ax1.plot([],[], c="k", lw=1.5,label="Ground Velocity")
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1b.set_xscale("log")
            ax1b.set_yscale("log")
            ax1.grid(ls =":")
            ax1.set_xlabel(r"Frequency / [$Hz$]")
            ax1.set_ylabel(r"Intensity / [$R$]")
            ax1b.set_ylabel(r"Velocity / [$m\cdot s^{-1}$]")
            ax1.legend(framealpha=1, edgecolor="none", loc=3) 
            ax1.set_xlim(8e-4, 1.2)
            ###
            ax2.fill_between(fr, np.min(sens_dg, axis=0), np.max(sens_dg, axis=0), color="orangered", alpha=0.3)
            ax2.fill_between(fr, np.min(sens_ng, axis=0), np.max(sens_ng, axis=0), color="forestgreen", alpha=0.3)
            ax2.plot(fr, np.median(sens_ng, axis=0), color="forestgreen", lw=1.5, label="Nightglow")
            ax2.plot(fr, np.median(sens_dg, axis=0), color="orangered", lw=1.5, label="Dayglow")
            for es in range(Ntr):
                ax2.plot(fr, sens_ng[es,:], c="k", lw=0.1, alpha=0.01)
                ax2.plot(fr, sens_dg[es,:], c="k", lw=0.1, alpha=0.01)
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel(r"Frequency / [$Hz$]")
            ax2.set_ylabel(r"Sensitivity / [$\%/(m\cdot s^{-1})$]")
            ax2.grid(ls =":")
            ax2.legend(framealpha=1, edgecolor="none", loc=3) 
            ax2.set_xlim(8e-4, 1.2)
            ax2.set_ylim(1e-3, 1e7)
            ###
            f_targetse = np.array(f_targets).T
            f_targetse[0,0] = 1e-4
            f_targetse[1,-1] = 1e4
            f_targetse[0,:] = fmean - f_targetse[0,:] 
            f_targetse[1,:] = f_targetse[1,:] - fmean
            # ax3.fill_between(fmean, np.min(store_Imax_dg, axis=1), np.max(store_Imax_dg, axis=1), color="orangered", alpha=0.3)
            # ax3.fill_between(fmean, np.min(store_Imax_ng, axis=1), np.max(store_Imax_ng, axis=1), color="forestgreen", alpha=0.3)
            ax3.fill_between(fmean, np.quantile(store_Imax_dg, q=0.25, axis=1), np.quantile(store_Imax_dg, q=0.75, axis=1), color="orangered", alpha=0.3)
            ax3.fill_between(fmean, np.quantile(store_Imax_ng, q=0.25, axis=1), np.quantile(store_Imax_ng, q=0.75, axis=1), color="forestgreen", alpha=0.3)
            ax3.errorbar(fmean, np.median(store_Imax_ng, axis=1), xerr=f_targetse, color="forestgreen", lw=1.5, label="Nightglow", marker="s")
            ax3.errorbar(fmean, np.median(store_Imax_dg, axis=1), xerr=f_targetse, color="orangered", lw=1.5, label="Dayglow", marker="s")
            ax3.set_xscale("log")
            ax3.set_yscale("log")
            ax3.set_xlabel(r"Frequency of input signal / [$Hz$]")
            ax3.set_ylabel(r"Max excitation / [$\%/(m\cdot s^{-1})$]")
            ax3.grid(ls =":")
            ax3.legend(framealpha=1, edgecolor="none", loc=3)    
            ax3.set_xlim(8e-4, 1.2)
            ax3.set_ylim(1e-3, 1e7)
            ###
            ### Choose 8 random velocity traces 
            choice = np.random.choice(Ntr,8)
            Vshape = AIRGLOW.VEL.reshape(Ntr, AIRGLOW.Nt)
            for ic, c in enumerate(choice):
                ax5.plot(AIRGLOW.t_new, Vshape[c,:]/abs(Vshape[c,:]).max() + 2*ic, lw=1, c="k")
            ax5.set_xlim(AIRGLOW.t_new.min(), AIRGLOW.t_new.max())
            ax5.set_xlabel(r"Time / [$s$]")
            ax5.set_ylabel(r"Input velocity traces")
            ax5.set_yticks([])
            ###
            ### Choose random velocity traces and filter them 
            choice = np.random.choice(Ntr)
            V = Vshape[choice,:]
            for fi, (f1, f2) in enumerate(f_targets):
                ### To scale with the velocity amplitude in each freq, band 
                Vf = butter_filter(V, 1/dt_airglow, f1,f2, order=5)
                ax6.plot(AIRGLOW.t_new, Vf/abs(Vf).max() + 2*fi, lw=1, c="k")
            ax6.set_xlim(AIRGLOW.t_new.min(), AIRGLOW.t_new.max())
            ax6.set_xlabel(r"Time / [$s$]")
            ax6.set_ylabel(r"Filtered velocity traces (Example)")
            ax6.set_yticks([2*fi for fi in range(len(f_targets))])
            ax6.set_yticklabels(["-- {:.1g} Hz".format(10**-2.5), "{:.1g}--{:.1g} Hz".format(10**-2.5, 10**-1.5), 
                                "{:.1g}--{:.1g} Hz".format(10**-1.5,10**-0.5), "{:.1g} Hz--".format(10**-0.5)])
                                # freq_bins = [None, 10**-2.5, 10**-1.5, 10**-0.5, None] 

            ###
            # ax4.set_axis_off()
            In = I_nightglow.reshape(Ntr, AIRGLOW.Nt)[choice,:]
            Id = I_dayglow.reshape(Ntr, AIRGLOW.Nt)[choice,:]
            for fi, (f1, f2) in enumerate(f_targets):
                ### To scale with the velocity amplitude in each freq, band 
                Ifn = butter_filter(In, 1/dt_airglow, f1,f2, order=5)
                ax4.plot(AIRGLOW.t_new, Ifn/abs(Ifn).max() + 2*fi, lw=1, c="forestgreen")
                Ifd = butter_filter(Id, 1/dt_airglow, f1,f2, order=5)
                ax4.plot(AIRGLOW.t_new, Ifd/abs(Ifd).max() + 2*fi, lw=1, c="orangered")
            ax4.set_xlim(AIRGLOW.t_new.min(), AIRGLOW.t_new.max())
            ax4.set_xlabel(r"Time / [$s$]")
            ax4.set_ylabel(r"Filtered airglow traces (Example)")
            ax4.set_yticks([2*fi for fi in range(len(f_targets))])
            ax4.set_yticklabels(["-- {:.1g} Hz".format(10**-2.5), "{:.1g}--{:.1g} Hz".format(10**-2.5, 10**-1.5), 
                                "{:.1g}--{:.1g} Hz".format(10**-1.5,10**-0.5), "{:.1g} Hz--".format(10**-0.5)])
                                # freq_bins = [None, 10**-2.5, 10**-1.5, 10**-0.5, None] 
            ###
            fig.tight_layout()
            # fig.savefig(dir_save + "Sensitivity_based_on_spectra"+ tit + ".png", dpi=300)
            # fig.savefig(dir_save + "Sensitivity_based_on_spectra"+ tit + ".pdf")
            ### ==========================================================================================================
            
    # airglow_scaler_test1(do_fig_spectrum=True)




def compute_airglow_scaler_sine(mw=None, strike=45, dip=45, rake=45, do_plot=True, effect=None, tit ="", 
                               store_ids_dists = [('GF_venus_Cold100_atten_qssp_nearfield',0e3,500e3),('GF_venus_Cold100_atten_qssp',500e3,8000e3)]):
    '''
    We calculate airglow sensitivity curves for a sine perturbation. 
    '''

    freq = np.array([0.01, 0.02, 0.1, 0.2]) ### Hz
    # freq = np.array([10**-3, 10**-2, 10**-1, 10**0])
    # freq = np.array([0.4,0.2,0.1]) ### Hz
    # freq = np.array([0.01]) ### Hz
    c = 200                 ### m/s
    dt = 0.1                ### s
    tf = 1500               ### s
    # hstart = 90e3           ### m   ### To test Kenda 2018 
    # ampl_at_start = 5e-3    ### Ampl of Mw 6.5, 30degree distance, 90 km alt (kenda 2018) 
    hstart = 100e3          ### m   ### To test Sutin 2018
    ampl_at_start = 4e-2    ### Ampl of Mw 6.5, 10degree distance, 100 km alt (sutin 2018) 

    # time = np.arange(-tf/10, tf, dt)
    time = np.arange(-tf/10, tf, dt)
    Nt = time.size
    VEL_ng = np.zeros((1,freq.size, Nt))
    DIS_ng = np.zeros((1,freq.size, Nt))
    VEL_dg = np.zeros((1,freq.size, Nt))
    DIS_dg = np.zeros((1,freq.size, Nt))
    

    def tapered_sinusoid(t, z, f0, c=200):
        ### Tapered sinusoid propagating at speed c 
        sig = np.sin(2*np.pi*f0*(t - z/c))
        Nsine = int((1/f0)/dt)
        tp = tukey(Nsine, alpha=0.0)

        tap = np.zeros(t.shape)
        Nt0 = np.where(t<0)[0].size
        if np.isscalar(z):
            Nprop = int(z/c/dt)
            tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
            sig*=tap
        else:
            for i in range(z.size):
                Nprop = int(z[i,0]/c/dt)
                tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
                sig[i,:]*=tap
        return(sig)

    def tapered_sinusoid_dt(t, z, f0, c=200):
        ### Time derivative of tapered sinusoid
        sig = 2*np.pi*f0*np.cos(2*np.pi*f0*(t - z/c))
        Nsine = int((1/f0)/dt)
        tp = tukey(Nsine, alpha=0.0)
        
        tap = np.zeros(t.shape)
        Nt0 = np.where(t<0)[0].size
        if np.isscalar(z):
            Nprop = int(z/c/dt)
            tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
            sig*=tap
            sig -= np.mean(sig)
        else:
            for i in range(z.size):
                Nprop = int(z[i,0]/c/dt)
                tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
                sig[i,:]*=tap
            sig -= np.mean(sig, axis=1)[:,None]
        return(sig)

    ### To verify that the signal and its derivative is calculated corectly
    # fig, ax = plt.subplots() 
    # for fi, f in enumerate(freq):
    #     ax.plot(time, VEL[0,fi,:])
    #     # ax.plot(time, tapered_sinusoid(time, hstart, f), ls="--")
    #     ax.plot(time, tapered_sinusoid_dz(time, hstart, f), ls="--")
    #     ax.plot(time, np.gradient( tapered_sinusoid(time[None,:], hstart+np.linspace(-1e2,1e2,11)[:,None], f), hstart+np.linspace(-1e2,1e2,11), axis=0 )[5,:] , ls=":")
    # plt.show() 
    # quit()

    ### Same for gaussian signal 
    # fig, ax = plt.subplots() 
    # for fi, f in enumerate(freq):
    #     # ax.plot(time, VEL[0,fi,:])
    #     # ax.plot(time, tapered_gaussian(time, hstart, f), ls="--")
    #     # ax.plot(time, tapered_gaussian_dz(time, hstart, f), ls="--")
    #     # ax.plot(time, np.gradient( tapered_gaussian(time[None,:], hstart+np.linspace(-1e2,1e2,11)[:,None], f), hstart+np.linspace(-1e2,1e2,11), axis=0 )[5,:] , ls=":")
    #     ax.plot(time, tapered_gaussian_dt(time, hstart, f), ls="--")
    #     ax.plot(time, np.gradient( tapered_gaussian(time, hstart, f), time) , ls=":")
    # plt.show() 
    # quit()


    ## Construct ground sinusoid for the Nightglow, based on our framework: 
    for fi, f0 in enumerate(freq):
        VEL_ng[0,fi,:] = tapered_sinusoid(time, 0e3, f0, c=c)
        DIS_ng[0,fi,:] = integrate.cumulative_trapezoid(tapered_sinusoid(time, 0e3, f0, c=c), time, initial=0)

    ### Construct a Dayglow sinusoid for our framework
    for fi, f0 in enumerate(freq):
        VEL_dg[0,fi,:] = tapered_sinusoid_dt(time, 0e3, f0, c=c)
        DIS_dg[0,fi,:] = tapered_sinusoid(time, 0e3, f0, c=c)
    ### Normalize velocity and displacement so that max(VEL)=1
    # tvmax = np.max(np.abs(VEL), axis=2)
    # VEL=VEL/tvmax[:,:,None] 
    # DIS=DIS/tvmax[:,:,None] 

    north_shifts = freq 
    east_shifts = np.array([0. for i in range(north_shifts.size)])
    Ntr = 1*north_shifts.size 

    SEISMO_ng ={"dt": dt,
             "t_new": time,
             "Nt": Nt,
             "VEL": VEL_ng, 
             "DIS": DIS_ng,
             "Nn" : north_shifts.size,  
             "Ne" : 1 ,
             "EE": east_shifts, 
             "NN" : north_shifts,
             "iEE" : [0 for i in freq],
             "iNN" : range(north_shifts.size),
             "gridded": False, 
    }

    SEISMO_dg ={"dt": dt,
             "t_new": time,
             "Nt": Nt,
             "VEL": VEL_dg, 
             "DIS": DIS_dg,
             "Nn" : north_shifts.size,  
             "Ne" : 1 ,
             "EE": east_shifts, 
             "NN" : north_shifts,
             "iEE" : [0 for i in freq],
             "iNN" : range(north_shifts.size),
             "gridded": False, 
    }

    # fig, ax = plt.subplots()
    # for fi in range(freq.size):
    #     ax.plot(time, VEL_ng[0,fi,:])
    #     ax.plot(time, DIS_dg[0,fi,:], ls="--")
    # plt.show()

    ################################################################################################################
    ### FRAMEWORK CALCUATION (using above classes): 
    ################################################################################################################    
    dt_airglow = dt
    AIRGLOW_ng = AirglowSignal(SEISMO_ng, Nz = 500, do_plot=False, disable_att=True)
    AIRGLOW_dg = AirglowSignal(SEISMO_dg, Nz = 500, do_plot=False, disable_att=True)
    ### Wavelength of ver: 
    lambda_min = np.min(AIRGLOW_ng.f_c(AIRGLOW_ng.z_1_27_calc_m))/max(freq)
    dz_min = AIRGLOW_ng.dz_1_27_m
    print("Min wavelength of VER = {:.1f} m".format(lambda_min))
    print("Min vertical resolution = {:.1f} m".format(dz_min))
    if lambda_min<=2*dz_min:
        print("WARNING: vertical resolution of integration might be insufficient for desired frequency")

    ### We want the sines to have an amplitude of 1 m/s at 0 km 
    # AIRGLOW.VEL = AIRGLOW.VEL #* ampl_at_start/AIRGLOW.f_amplification(hstart)
    # AIRGLOW.DIS = AIRGLOW.DIS #* ampl_at_start/AIRGLOW.f_amplification(hstart)
    # print(np.max(AIRGLOW.VEL, axis=2))
    # print(np.max(AIRGLOW.DIS, axis=2))

    ### Now we compute the AIRGLOW at all locations and timesteps. 
    ### NOTE : This can be quite heavy ! 
    ### List of all north and east indices:
    dir_save="./results_detectability_sine/"
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    list_inorth, list_ieast = AIRGLOW_ng.iNN, AIRGLOW_ng.iEE
    ### Calculation of the Nightglow
    AIRGLOW_ng.calculate_1_27_airglow(list_ieast, list_inorth, loc_save_idx=[],
                                   do_parallel=True, 
                                   fourier_filtering=False,   ### Use time filtering 
                                   dir_save = dir_save,
                                   save_ver= False,              ### Faster calculation
                                   time_save = AIRGLOW_ng.t_new) ### Save all timesteps 
    ### Calculation of the Dayglow
    AIRGLOW_dg.calculate_4_28_airglow(list_ieast, list_inorth, loc_save_idx=[],
                                   do_parallel=True, 
                                   dir_save=dir_save, 
                                   save_ver= False,              ### Faster calculation 
                                   time_save = AIRGLOW_dg.t_new) ### Save all timesteps 

    I_nightglow = np.load(dir_save + "nightglow_I_t.npy")
    I_dayglow = np.load(dir_save + "dayglow_I_t.npy")

        ### ==========================================================================================================
    ### OTHER OPTION: Filter velocity before calculating Airglow 
    ### Then calculate the max intensity 
    fmean = [10**-3, 10**-2, 10**-1, 10**0]
    freq_bins = [None, 10**-2.5, 10**-1.5, 10**-0.5, None]  ### Centered around 1e-2, 1e-1, 1. 
    f_targets = []
    for _, (binleft, binright) in enumerate(zip(freq_bins[:-1], freq_bins[1:])):
        f_targets += [[binleft, binright]]
    print(" Filter bins: ", f_targets)
    store_Imax_ng = []
    store_Imax_dg = []
    for f1, f2 in tqdm(f_targets, disable=True):
        ### DAYGLOW
        VEL_filt = butter_filter(AIRGLOW_dg.VEL, 1/dt_airglow, f1,f2, order=5, axis=2)
        DIS_filt = butter_filter(AIRGLOW_dg.DIS, 1/dt_airglow, f1,f2, order=5, axis=2)
        ### Normalize all the velocity traces to 1: 
        mvel = np.max(np.abs(VEL_filt), axis=2)
        SEISMO_dg["VEL"] = VEL_filt/mvel[:,:,None] 
        SEISMO_dg["DIS"] = DIS_filt/mvel[:,:,None]
        Vfilt_dg = VEL_filt.reshape(Ntr, AIRGLOW_dg.Nt)
        Vfilt_dg = Vfilt_dg / mvel.reshape(Ntr)[:,None]

        ### NIGHTGLOW
        VEL_filt = butter_filter(AIRGLOW_ng.VEL, 1/dt_airglow, f1,f2, order=5, axis=2)
        DIS_filt = butter_filter(AIRGLOW_ng.DIS, 1/dt_airglow, f1,f2, order=5, axis=2)
        ### Normalize all the velocity traces to 1: 
        mvel = np.max(np.abs(VEL_filt), axis=2)
        SEISMO_ng["VEL"] = VEL_filt/mvel[:,:,None] 
        SEISMO_ng["DIS"] = DIS_filt/mvel[:,:,None]
        Vfilt_ng = VEL_filt.reshape(Ntr, AIRGLOW_ng.Nt)
        Vfilt_ng = Vfilt_ng / mvel.reshape(Ntr)[:,None]

        AIRGLOW_filt_dg = AirglowSignal(SEISMO_dg, Nz=500)
        AIRGLOW_filt_ng = AirglowSignal(SEISMO_ng, Nz=500)
        dir_save_filt="./results_detectabilityf/"
        if not os.path.exists(dir_save_filt):
            os.makedirs(dir_save_filt)
        list_inorth, list_ieast = AIRGLOW_filt_dg.iNN, AIRGLOW_filt_dg.iEE
        ### Calculation of the Nightglow
        AIRGLOW_filt_ng.calculate_1_27_airglow(list_ieast, list_inorth, loc_save_idx=[],
                                       do_parallel=True, 
                                       fourier_filtering=False,   ### Use time filtering 
                                       dir_save = dir_save_filt,
                                       save_ver = False,          ### Faster if we save only I(lat, lon, t)
                                       time_save = AIRGLOW_filt_ng.t_new) ### Save all timesteps 
        ### Calculation of the Dayglow
        AIRGLOW_filt_dg.calculate_4_28_airglow(list_ieast, list_inorth, loc_save_idx=[],
                                       do_parallel=True, 
                                       dir_save=dir_save_filt, 
                                       save_ver = False,          ### Faster if we save only I(lat, lon, t)
                                       time_save = AIRGLOW_filt_dg.t_new) ### Save all timesteps 
        ### Now we recover the max intensity 
        I_nightglow_filt = np.load(dir_save_filt + "nightglow_I_t.npy").reshape(Ntr,AIRGLOW_filt_ng.Nt)
        I_dayglow_filt = np.load(dir_save_filt + "dayglow_I_t.npy").reshape(Ntr,AIRGLOW_filt_dg.Nt)

        sensmax_nightglow_filt = np.max(abs(I_nightglow_filt), axis=1)/AIRGLOW_filt_ng.I_background_nightglow/\
                                 np.max(abs(Vfilt_ng), axis=1) * 100
        sensmax_dayglow_filt = np.max(abs(I_dayglow_filt), axis=1)/AIRGLOW_filt_dg.I_background_dayglow/\
                                 np.max(abs(Vfilt_dg), axis=1) * 100

        store_Imax_ng.append(sensmax_nightglow_filt)
        store_Imax_dg.append(sensmax_dayglow_filt)
    store_Imax_ng = np.array(store_Imax_ng)
    store_Imax_dg = np.array(store_Imax_dg)

    ### ==========================================================================================================
    ### FIGURE 1: Look at the ratio of airglow spectra to velocity spectra 
    fr = np.fft.rfftfreq(n=AIRGLOW_ng.t_new.size, d=dt_airglow)[1:]
    sp_ng = abs(np.fft.rfft(AIRGLOW_ng.VEL, axis=2))[:,:,1:].reshape(Ntr, fr.size)         ### Ground Velocity spectrum (nightglow)
    sp_dg = abs(np.fft.rfft(AIRGLOW_dg.VEL, axis=2))[:,:,1:].reshape(Ntr, fr.size)         ### Ground Velocity spectrum (dayglow)
    spin = abs(np.fft.rfft(I_nightglow, axis=2))[:,:,1:].reshape(Ntr, fr.size)       ### Nightglow spectrum
    spid = abs(np.fft.rfft(I_dayglow, axis=2))[:,:,1:].reshape(Ntr, fr.size)         ### Dayglow spectrum 

    fig, ((ax1, ax2, ax3),(ax4,ax5,ax6)) = plt.subplots(2,3, figsize=(13,8)) 
    sens_ng = spin/AIRGLOW_ng.I_background_nightglow/sp_ng*100
    sens_dg = spid/AIRGLOW_dg.I_background_dayglow/sp_dg*100
    ###
    ax1b = ax1.twinx()
    for es in range(Ntr):
        ax1.plot(fr, spin[es,:], c="forestgreen", lw=0.5, alpha=0.01)
        ax1.plot(fr, spid[es,:], c="orangered", lw=0.5, alpha=0.01)
        ax1b.plot(fr, sp_ng[es,:], c="grey", lw=0.5, alpha=0.01)
        ax1b.plot(fr, sp_dg[es,:], c="k", lw=0.5, alpha=0.01)
    ax1.plot([],[], c="grey", lw=0.5, alpha=0.01, label="Velocity for Nightglow")
    ax1.plot([],[], c="k", lw=0.5, alpha=0.01, label="Velocity for Dayglow")
    ax1.plot(fr, np.median(spin, axis=0), c="forestgreen", lw=1.5,label="Nightglow")
    ax1.plot(fr, np.median(spid, axis=0), c="orangered", lw=1.5,label="Dayglow")
    ax1b.plot(fr, np.median(sp_ng, axis=0), c="grey", lw=1.5,label="Ground Velocity")
    ax1b.plot(fr, np.median(sp_dg, axis=0), c="k", lw=1.5,label="Ground Velocity")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1b.set_xscale("log")
    ax1b.set_yscale("log")
    ax1.grid(ls =":")
    ax1.set_xlabel(r"Frequency / [$Hz$]")
    ax1.set_ylabel(r"Intensity / [$R$]")
    ax1b.set_ylabel(r"Velocity / [$m\cdot s^{-1}$]")
    ax1.legend(framealpha=1, edgecolor="none", loc=3) 
    ax1.set_xlim(8e-4, 1.2)
    ###
    ax2.fill_between(fr, np.min(sens_dg, axis=0), np.max(sens_dg, axis=0), color="orangered", alpha=0.3)
    ax2.fill_between(fr, np.min(sens_ng, axis=0), np.max(sens_ng, axis=0), color="forestgreen", alpha=0.3)
    ax2.plot(fr, np.median(sens_ng, axis=0), color="forestgreen", lw=1.5, label="Nightglow")
    ax2.plot(fr, np.median(sens_dg, axis=0), color="orangered", lw=1.5, label="Dayglow")
    for es in range(Ntr):
        ax2.plot(fr, sens_ng[es,:], c="k", lw=0.1, alpha=0.01)
        ax2.plot(fr, sens_dg[es,:], c="k", lw=0.1, alpha=0.01)
    ax2.plot(freq, np.max(abs(I_nightglow[0,:,:]), axis=1)/AIRGLOW_filt_ng.I_background_nightglow*100, color="forestgreen", lw=1.5, marker="s", label="Sutin")
    ax2.plot(freq, np.max(abs(I_dayglow[0,:,:]), axis=1)/AIRGLOW_filt_ng.I_background_dayglow*100, color="orangered", lw=1.5, marker="s")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Frequency / [$Hz$]")
    ax2.set_ylabel(r"Sensitivity / [$\%/(m\cdot s^{-1})$]")
    ax2.grid(ls =":")
    ax2.legend(framealpha=1, edgecolor="none", loc=3) 
    ax2.set_xlim(8e-4, 1.2)
    ax2.set_ylim(1e-3, 1e7)
    ###
    f_targetse = np.array(f_targets).T
    f_targetse[0,0] = 1e-4
    f_targetse[1,-1] = 1e4
    f_targetse[0,:] = fmean - f_targetse[0,:] 
    f_targetse[1,:] = f_targetse[1,:] - fmean
    # ax3.fill_between(fmean, np.min(store_Imax_dg, axis=1), np.max(store_Imax_dg, axis=1), color="orangered", alpha=0.3)
    # ax3.fill_between(fmean, np.min(store_Imax_ng, axis=1), np.max(store_Imax_ng, axis=1), color="forestgreen", alpha=0.3)
    ls = ["-", "--", "-.", ":"]
    for fi, f in enumerate(freq):
        ax3.plot(fmean, store_Imax_ng[:,fi], color="forestgreen", lw=1.5, label="{:.3g} Hz".format(f), marker="s", ls=ls[fi])
        ax3.plot(fmean, store_Imax_dg[:,fi], color="orangered", lw=1.5, marker="s", ls=ls[fi])
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel(r"Frequency of input signal / [$Hz$]")
    ax3.set_ylabel(r"Max excitation / [$\%/(m\cdot s^{-1})$]")
    ax3.grid(ls =":")
    ax3.legend(framealpha=1, edgecolor="none", loc=3)    
    ax3.set_xlim(8e-4, 1.2)
    ax3.set_ylim(1e-3, 1e7)
    ###
    ### Choose 8 random velocity traces 
    for ic, f in enumerate(freq):
        ax5.plot(AIRGLOW_ng.t_new, AIRGLOW_ng.VEL[0,ic,:]/abs(AIRGLOW_ng.VEL[0,ic,:]).max() + 2*ic, lw=1, c="k")
        ax5.plot(AIRGLOW_dg.t_new, AIRGLOW_dg.VEL[0,ic,:]/abs(AIRGLOW_dg.VEL[0,ic,:]).max() + 2*ic, lw=1, c="k", ls="--")
    ax5.set_xlim(AIRGLOW_ng.t_new.min(), AIRGLOW_ng.t_new.max())
    ax5.set_xlabel(r"Time / [$s$]")
    ax5.set_ylabel(r"Input velocity traces (Example)")
    ax5.set_yticks([])
    ###
    ### Show 4 sinusoid 
    for fi, (f1, f2) in enumerate(f_targets):
        ### To scale with the velocity amplitude in each freq, band 
        for ic, f in enumerate(freq):
            Vf = butter_filter(AIRGLOW_ng.VEL[0,ic,:], 1/dt_airglow, f1,f2, order=5)
            ax6.plot(AIRGLOW_ng.t_new, Vf/abs(Vf).max() + 2*fi, lw=1, c="k")
            Vf = butter_filter(AIRGLOW_dg.VEL[0,ic,:], 1/dt_airglow, f1,f2, order=5)
            ax6.plot(AIRGLOW_dg.t_new, Vf/abs(Vf).max() + 2*fi, lw=1, c="k", ls="--")
    ax6.set_xlim(AIRGLOW_ng.t_new.min(), AIRGLOW_ng.t_new.max())
    ax6.set_xlabel(r"Time / [$s$]")
    ax6.set_ylabel(r"Filtered Input velocity traces (Example)")
    ax6.set_yticks([2*fi for fi in range(len(f_targets))])
    ax6.set_yticklabels(["-- {:.1g} Hz".format(10**-2.5), "{:.1g}--{:.1g} Hz".format(10**-2.5, 10**-1.5), 
                         "{:.1g}--{:.1g} Hz".format(10**-1.5,10**-0.5), "{:.1g} Hz--".format(10**-0.5)])
                         # freq_bins = [None, 10**-2.5, 10**-1.5, 10**-0.5, None] 

    ###
    ax4.set_axis_off()
    ###
    fig.tight_layout()
    fig.savefig(dir_save + "Sensitivity_based_on_spectra_sine_"+ tit + ".png", dpi=300)
    fig.savefig(dir_save + "Sensitivity_based_on_spectra_sine_"+ tit + ".pdf")
    ### ==========================================================================================================
    

    ### Now, we make some frequency bins 
    # freq_bins = np.logspace(np.log10(1e-3), np.log10(5e-1), 5)
    fmean = [10**-3, 10**-2, 10**-1, 10**0]
    freq_bins = [None, 10**-2.5, 10**-1.5, 10**-0.5, None]  ### Centered around 1e-2, 1e-1, 1. 
    fr = np.fft.rfftfreq(n=time.size, d=dt_airglow)[1:]
    f_targets = []
    for ibin, (binleft, binright) in enumerate(zip(freq_bins[:-1], freq_bins[1:])):
        f_targets += [[binleft, binright]]
    print(" Filter bins: ", f_targets)

    scaling_airglow = pd.DataFrame()
    ### We loop over locations and store the max amplitude in a dataframe: 
    for fi, (f1, f2) in tqdm(enumerate(f_targets), disable=True):

        ### To scale with the velocity amplitude in each freq, band 
        waveform_vel_ng_filt = butter_filter(AIRGLOW_ng.VEL, 1/dt_airglow, f1,f2, order=4, axis=2)
        waveform_vel_dg_filt = butter_filter(AIRGLOW_dg.VEL, 1/dt_airglow, f1,f2, order=4, axis=2)
        
        # waveform_nightglow_filt =np.repeat(butter_filter(I_nightglow[:,fi:fi+1,:], 1/dt_airglow, f1,f2, order=5, axis=2), freq.size, axis=1)
        waveform_nightglow_filt =np.repeat(I_nightglow[:,fi:fi+1,:], freq.size, axis=1)
        perturb_nightglow_filt = waveform_nightglow_filt/AIRGLOW_ng.I_background_nightglow/\
                                np.max(abs(waveform_vel_ng_filt), axis=2)[:,:,None] * 100

        # waveform_dayglow_filt = np.repeat(butter_filter(I_dayglow[:,fi:fi+1,:], 1/dt_airglow, f1,f2, order=5, axis=2), freq.size, axis=1)
        waveform_dayglow_filt = np.repeat(I_dayglow[:,fi:fi+1,:], freq.size, axis=1)
        perturb_dayglow_filt = waveform_dayglow_filt/AIRGLOW_dg.I_background_dayglow/\
                                np.max(abs(waveform_vel_dg_filt), axis=2)[:,:,None] * 100

        print(f1, f2, freq[fi])
        
        # fig, ax = plt.subplots() 
        # fr = np.fft.rfftfreq(n=time.size, d=dt_airglow)
        # ax.plot(fr, abs(np.fft.rfft(perturb_nightglow_filt[0,fi,:]))*np.sqrt(dt/AIRGLOW_ng.Nt), c="k", label="Sensitivity [%]")
        # axb = ax.twinx()
        # axb.plot(fr, abs(np.fft.rfft(VEL_ng[0,fi,:]))*np.sqrt(dt/AIRGLOW_ng.Nt), c="r", label="Vel at 0 km")
        # axb.plot(fr, abs(np.fft.rfft(I_nightglow[0,fi,:]/abs(I_nightglow[0,fi,:]).max()))*np.sqrt(dt/AIRGLOW_ng.Nt), 
        #          c="grey", lw=1, label="Intensity")
        
        # ax.legend(loc=0)
        # axb.legend(loc=4)
        # ax.set_xscale("log")
        # # ax.set_yscale("log")
        # ax.axvline(freq[fi])
        # ax.axvline(fmean_ng[fi], ls="--")
        # if f1 == None: 
        #     f1b = 0 
        # else: 
        #     f1b = f1
        # if f2 == None: 
        #     f2b = 1e8
        # else: 
        #     f2b = f2
        # ax.axvspan(f1b,f2b,color="grey", alpha=0.4)
        # ax.set_title("F = {:.3g} Hz".format(freq[fi]))

        for (ies, ins) in zip(AIRGLOW_ng.iEE, AIRGLOW_ng.iNN):
            es, ns = AIRGLOW_ng.EE[ies], AIRGLOW_ng.NN[ins]
            loc_dict = dict(ns=ns, es=es, 
                            f1=f1 if f1 is not None else 0, 
                            f2=f2 if f2 is not None else 1., 
                            nightglow=abs(perturb_nightglow_filt[ies, ins,:]).max(),
                            dayglow=abs(perturb_dayglow_filt[ies, ins,:]).max())
            # dayglow=abs(waveform_dayglow).max()
            scaling_airglow = pd.concat([scaling_airglow, pd.DataFrame([loc_dict])])

    # print(scaling_airglow)
    ### Calculate statistics 
    scaling_nightglow_plot = scaling_airglow.groupby(['f1', 'f2',])['nightglow'].median().reset_index()
    scaling_nightglow_plot['nightglow_q25'] = scaling_airglow.groupby(['f1', 'f2',])['nightglow'].quantile(q=0.25).reset_index()['nightglow']
    scaling_nightglow_plot['nightglow_q75'] = scaling_airglow.groupby(['f1', 'f2',])['nightglow'].quantile(q=0.75).reset_index()['nightglow']
    ###
    scaling_dayglow_plot = scaling_airglow.groupby(['f1', 'f2',])['dayglow'].median().reset_index()
    scaling_dayglow_plot['dayglow_q25'] = scaling_airglow.groupby(['f1', 'f2',])['dayglow'].quantile(q=0.25).reset_index()['dayglow']
    scaling_dayglow_plot['dayglow_q75'] = scaling_airglow.groupby(['f1', 'f2',])['dayglow'].quantile(q=0.75).reset_index()['dayglow']

    scaling_nightglow_plot.to_csv(dir_save + "nightglow_scaler"+tit + ".csv", header=True, index=False)
    scaling_dayglow_plot.to_csv(dir_save + "dayglow_scaler"+tit + ".csv", header=True, index=False)

    if do_plot:
        fig, ax = plt.subplots()

        ### Instead of using filtering, we find the dominant freq of the initial signal 
        # fmean_dg = fmean 
        # fmean_ng = fmean 
        # fmean_dg = [] 
        # fmean_ng = [] 
        # fr = np.fft.rfftfreq(n=time.size, d=dt_airglow)[1:]
        # for fi in range(freq.size):
        #     sp = abs(np.fft.rfft(AIRGLOW_dg.VEL[0,fi,:]))
        #     fmean_dg.append(fr[np.argmax(sp[1:])])

        #     sp = abs(np.fft.rfft(AIRGLOW_ng.VEL[0,fi,:]))
        #     fmean_ng.append(fr[np.argmax(sp[1:])])


        ax.plot(fmean, scaling_nightglow_plot.nightglow, 
                color='forestgreen', marker="s", label=r"1.27$\mu m$ Nightglow")
        ax.fill_between(fmean, 
                        scaling_nightglow_plot.nightglow_q25, scaling_nightglow_plot.nightglow_q75,
                        color='forestgreen', alpha=0.3)
        ax.plot(fmean, scaling_dayglow_plot.dayglow, 
                color='orangered', marker="s", label=r"4.28$\mu m$ Dayglow")
        ax.fill_between(fmean, 
                        scaling_dayglow_plot.dayglow_q25, scaling_dayglow_plot.dayglow_q75,
                        color='orangered', alpha=0.3)
        
        ax.legend(frameon=False)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel("Frequency / [$Hz$]")
        ax.set_ylabel(r"Airglow Intensity perturbation [$\%/(1\,m/s)$]")
        ax.set_title("Intensity perturbation for 1 $m/s$ peak velocity at the ground surface")
        # fig.savefig(dir_save + "Airglow_scaler"+ tit + ".png", dpi=300)




def compute_airglow_scaler_Hots(mw=None, strike=45, dip=45, rake=45, do_plot=True, effect=None, tit ="", 
                               store_ids_dists = [('GF_venus_Hot10_atten_qssp_nearfield',0e3,50e3),('GF_venus_Hot10_atten_qssp',50e3,8000e3)]):
    '''
    We calculate airglow signals for a series of receiver locations and source depths. 
    '''

    ### First, define the grid of locations. 
    gridded       = True
    min_grid_dist = 0e3
    max_grid_dist = 5000e3 # 4000e3
    delta_dist    = 50e3 
    Np = 1+ int(2*max_grid_dist/delta_dist)
    north_shifts  = np.linspace(-max_grid_dist, max_grid_dist, Np, endpoint=True)[::10]
    east_shifts   = np.linspace(-max_grid_dist, max_grid_dist, Np, endpoint=True)[::10]
    Ntr = east_shifts.size*north_shifts.size 
    
    ### Source depths 
    delta_depth = 5e3
    depths = np.arange(5e3, 50e3+delta_depth, delta_depth)

    ### Option for Pyrocko 
    opt_synthetics = dict(
        ### Options for source 
        mw = mw,               ### if none: We only get the Green's function
        depth = depths[5],     ### Only one depth
        strike = strike,       ### Default mechanism 
        dip =  dip, 
        rake = rake,
        stf_type = None,       ### Dirac source 
        #stf_type = 'triangle', 
        # stf_type = 'sinus', 
        #effective_duration = 25.,
        ###  
        ### Options for store
        base_folder='/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/',
        ### Old option, single store 
        #store_id = 'GF_venus_Cold100_qssp',
        #store_id = 'GF_venus_Cold100_qssp_grid',
        ### Give store names, and min and max valid distance 
        store_ids_dists = store_ids_dists,
        ###
        ### Options for grid 
        north_shifts = north_shifts, 
        east_shifts = east_shifts,
        gridded=gridded
    )

    dir_save="./results_detectability/"
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    

    def airglow_scaler_calculation(opt_synthetics):
        ### INITIALIZE SEISMOGRAM CLASS 
        ### we build seismograms over grid
        ### 
        SEISMO = Seismograms(**opt_synthetics)
        print("Total grid size: ", SEISMO.NN.size)

        ### OPTIONAL: Plot to check 
        ns, es       = 1000e3, 0e3
        ### Plot one of the waveforms for check
        # fig1 = SEISMO.plot_traces(ns, es, do_interpolate=True)

        ### Store results inside a regular grid. 
        dt_airglow = 0.5
        SEISMO.arrange_interpolate_synthetics(tmax=3000, dt=dt_airglow)

        ### Normalize all the velocity traces to 1: 
        mvel = np.max(np.abs(SEISMO.VEL), axis=2)
        SEISMO.VEL/=mvel[:,:,None] 
        SEISMO.DIS/=mvel[:,:,None]

        ### OPTIONAL: Plot velocity wavefront at the surface for sanity check 
        # fig2 = SEISMO.plot_wavefront()

        ### Now, we will calculate the AIRGLOW at every loc of the grid
        ### We first load the airglow class
        AIRGLOW = AirglowSignal(SEISMO, Nz=500)

        ### Now we compute the AIRGLOW at all locations and timesteps. 
        ### NOTE : This can be quite heavy ! 
        ### List of all north and east indices:
        list_inorth, list_ieast = AIRGLOW.iNN, AIRGLOW.iEE

        ### Calculation of the Nightglow
        AIRGLOW.calculate_1_27_airglow(list_ieast, list_inorth, loc_save_idx=[],
                                    do_parallel=True, 
                                    fourier_filtering=False,   ### Use time filtering 
                                    dir_save = dir_save,
                                    save_ver = False,          ### Faster if we save only I(lat, lon, t)
                                    time_save = AIRGLOW.t_new) ### Save all timesteps 
        ### Calculation of the Dayglow
        AIRGLOW.calculate_4_28_airglow(list_ieast, list_inorth, loc_save_idx=[],
                                    do_parallel=True, 
                                    dir_save=dir_save, 
                                    save_ver = False,          ### Faster if we save only I(lat, lon, t)
                                    time_save = AIRGLOW.t_new) ### Save all timesteps 

        I_nightglow = np.load(dir_save + "nightglow_I_t.npy")
        I_dayglow = np.load(dir_save + "dayglow_I_t.npy")
        
        ### ==========================================================================================================
        ### FIGURE: We make some frequency bins 
        # freq_bins = np.logspace(np.log10(1e-3), np.log10(5e-1), 5)
        fmean = [10**-3, 10**-2, 10**-1, 10**0]
        freq_bins = [None, 10**-2.5, 10**-1.5, 10**-0.5, None]  ### Centered around 1e-2, 1e-1, 1. 
        f_targets = []
        for _, (binleft, binright) in enumerate(zip(freq_bins[:-1], freq_bins[1:])):
            f_targets += [[binleft, binright]]
        print(" Filter bins: ", f_targets)

        scaling_airglow = pd.DataFrame()
        ### We loop over locations and store the max amplitude in a dataframe: 
        for f1, f2 in tqdm(f_targets, disable=True):

            ### To scale with the velocity amplitude in each freq, band 
            waveform_vel_filt = butter_filter(AIRGLOW.VEL, 1/dt_airglow, f1,f2, order=5, axis=2)
            
            waveform_nightglow_filt = butter_filter(I_nightglow, 1/dt_airglow, f1,f2, order=5, axis=2)
            perturb_nightglow_filt = waveform_nightglow_filt/AIRGLOW.I_background_nightglow/\
                                    np.max(abs(waveform_vel_filt), axis=2)[:,:,None] * 100

            waveform_dayglow_filt = butter_filter(I_dayglow, 1/dt_airglow, f1,f2, order=5, axis=2)
            perturb_dayglow_filt = waveform_dayglow_filt/AIRGLOW.I_background_dayglow/\
                                    np.max(abs(waveform_vel_filt), axis=2)[:,:,None] * 100

            for (ies, ins) in zip(AIRGLOW.iEE.ravel(), AIRGLOW.iNN.ravel()):
                es, ns = AIRGLOW.EE[ies, ins], AIRGLOW.NN[ies,ins]
                loc_dict = dict(ns=ns, es=es, 
                                f1=f1 if f1 is not None else 0, 
                                f2=f2 if f2 is not None else 1., 
                                nightglow=abs(perturb_nightglow_filt[ies, ins,:]).max(),
                                dayglow=abs(perturb_dayglow_filt[ies, ins,:]).max())
                # dayglow=abs(waveform_dayglow).max()
                scaling_airglow = pd.concat([scaling_airglow, pd.DataFrame([loc_dict])])

        ### Calculate statistics 
        scaling_nightglow_plot = scaling_airglow.groupby(['f1', 'f2',])['nightglow'].median().reset_index()
        scaling_nightglow_plot['nightglow_q25'] = scaling_airglow.groupby(['f1', 'f2',])['nightglow'].quantile(q=0.25).reset_index()['nightglow']
        scaling_nightglow_plot['nightglow_q75'] = scaling_airglow.groupby(['f1', 'f2',])['nightglow'].quantile(q=0.75).reset_index()['nightglow']
        ###
        scaling_dayglow_plot = scaling_airglow.groupby(['f1', 'f2',])['dayglow'].median().reset_index()
        scaling_dayglow_plot['dayglow_q25'] = scaling_airglow.groupby(['f1', 'f2',])['dayglow'].quantile(q=0.25).reset_index()['dayglow']
        scaling_dayglow_plot['dayglow_q75'] = scaling_airglow.groupby(['f1', 'f2',])['dayglow'].quantile(q=0.75).reset_index()['dayglow']

        scaling_nightglow_plot.to_csv(dir_save + "nightglow_scaler"+tit + ".csv", header=True, index=False)
        scaling_dayglow_plot.to_csv(dir_save + "dayglow_scaler"+tit + ".csv", header=True, index=False)

        return(fmean, scaling_nightglow_plot, scaling_dayglow_plot)
    
    
    def plot_scaler_hot(fmean, ng_hot25, dg_hot25, ng_hot10, dg_hot10, ng_hot40, dg_hot40, ng_cold100, dg_cold100):
        fig, ax = plt.subplots(figsize=(5.5,4))

        ax.plot(fmean, ng_hot25.nightglow, color='forestgreen', marker="s", label=r"'Hot25', 1.27$\mu m$ Nightglow")
        ax.fill_between(fmean, ng_hot25.nightglow_q25, ng_hot25.nightglow_q75,color='forestgreen', alpha=0.3)
        ax.plot(fmean, dg_hot25.dayglow, color='orangered', marker="s", label=r"'Hot25', 4.28$\mu m$ Dayglow")
        ax.fill_between(fmean, dg_hot25.dayglow_q25, dg_hot25.dayglow_q75, color='orangered', alpha=0.3)
        ###
        ax.plot(fmean, ng_hot10.nightglow, color='forestgreen', marker="v", ls="--", label=r"'Hot10'")
        # ax.fill_between(fmean, ng_hot10.nightglow_q25, ng_hot10.nightglow_q75,color='forestgreen', alpha=0.2)
        ax.plot(fmean, dg_hot10.dayglow, color='orangered', marker="v", ls ="--")
        # ax.fill_between(fmean, dg_hot10.dayglow_q25, dg_hot10.dayglow_q75,color='orangered', alpha=0.2)
        ###
        ax.plot(fmean, ng_hot40.nightglow, color='forestgreen', marker="^", ls=":", label=r"'Hot40'")
        # ax.fill_between(fmean, ng_hot40.nightglow_q25, ng_hot40.nightglow_q75,color='forestgreen', alpha=0.2)
        ax.plot(fmean, dg_hot40.dayglow, color='orangered', marker="^", ls =":")
        # ax.fill_between(fmean, dg_hot40.dayglow_q25, dg_hot40.dayglow_q75,color='orangered', alpha=0.2)
        ###
        ax.plot(fmean, ng_cold100.nightglow, color='k', marker="s", label=r"'Cold100'")
        # ax.fill_between(fmean, ng_cold100.nightglow_q25, ng_hot25.nightglow_q75,color='k', alpha=0.3)
        ax.plot(fmean, dg_cold100.dayglow, color='k', marker="s")
        # ax.fill_between(fmean, dg_cold100.dayglow_q25, dg_hot25.dayglow_q75, color='k', alpha=0.3)
        

        leg = ax.legend(frameon=False, title="Subsurface model", loc=3)
        leg._legend_box.align = "left"
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(1e-3,1e6)
        ax.set_xlabel("Frequency / [$Hz$]")
        ax.set_ylabel(r"Airglow Intensity perturbation [$\%/(1\,m/s)$]")
        ax.set_title("Intensity perturbation for 1 $m/s$"+"\npeak velocity at the ground surface")
        fig.tight_layout()
        
        fig.savefig(dir_save + "Airglow_scaler_Hots.png", dpi=300)
        fig.savefig(dir_save + "Airglow_scaler_Hots.pdf")



    ### Do MIN 
    opt_synthetics["store_ids_dists"] = [('GF_venus_Hot10_atten_qssp_nearfield',0e3,50e3),('GF_venus_Hot10_atten_qssp',50e3,8000e3)]
    tit = "_Hot10"
    fmean, scaling_nightglow_plot_hot10, scaling_dayglow_plot_hot10 = airglow_scaler_calculation(opt_synthetics)
    ### Do MAX 
    opt_synthetics["store_ids_dists"] = [('GF_venus_Hot40_atten_qssp_nearfield',0e3,50e3),('GF_venus_Hot40_atten_qssp',50e3,8000e3)]
    tit = "_Hot40"
    fmean, scaling_nightglow_plot_hot40, scaling_dayglow_plot_hot40 = airglow_scaler_calculation(opt_synthetics)
    ### Do ORIG 
    opt_synthetics["store_ids_dists"] = [('GF_venus_Hot25_atten_qssp_nearfield',0e3,50e3),('GF_venus_Hot25_atten_qssp',50e3,8000e3)]
    tit = "_Hot25"
    fmean, scaling_nightglow_plot_hot25, scaling_dayglow_plot_hot25 = airglow_scaler_calculation(opt_synthetics)
    ### Do ORIG 
    opt_synthetics["store_ids_dists"] = [('GF_venus_Cold100_atten_qssp_nearfield',0e3,50e3),('GF_venus_Cold100_atten_qssp',50e3,8000e3)]
    tit = "_Cold100"
    fmean, scaling_nightglow_plot_cold100, scaling_dayglow_plot_cold100 = airglow_scaler_calculation(opt_synthetics)
    
    plot_scaler_hot(fmean, scaling_nightglow_plot_hot25, scaling_dayglow_plot_hot25, 
                        scaling_nightglow_plot_hot10, scaling_dayglow_plot_hot10, 
                         scaling_nightglow_plot_hot40, scaling_dayglow_plot_hot40,
                         scaling_nightglow_plot_cold100, scaling_dayglow_plot_cold100)



# =========================================================================================================
### VERIFICATIONS AND TESTS 
# =========================================================================================================
def check_simple_perturbation_nightglow(test="kenda"):
    ### Testing two simple approaches: 
    ###  - As in Kenda's thesis (2018), we send a sinusoidal 
    ###    perturbation with amplitude defined at 90 km.
    ###  - As in Sutin et al. (2018), we send a gaussian 
    ###    perturbation with amplitude defined at 100 km. 

    ### Time parameters 
    dt = 1                ### s
    tf = 1000               ### s
    time = np.arange(-tf/2, tf, dt)
    # time = np.arange(-tf, tf, dt)
    Nt = time.size
    # print(Nt)
    # quit()

    ### Input signal parameter
    freq = np.array([0.2,0.1,0.02, 0.01]) ### Hz
    # freq = np.array([0.4,0.2,0.1]) ### Hz
    # freq = np.array([0.01]) ### Hz
    if test=="kenda":
        c = 200                 ### Fixed vertical wave velocity m/s
        hstart = 90e3           ### m   ### To test Kenda 2018 
        ampl_at_start = 5e-3    ### Ampl of Mw 6.5, 30degree distance, 90 km alt (kenda 2018) 
    elif test=="sutin":
        c = 200                 ### Fixed vertical wave velocity m/s
        hstart = 100e3          ### m   ### To test Sutin 2018
        ampl_at_start = 4e-2    ### Ampl of Mw 6.5, 10degree distance, 100 km alt (sutin 2018) 

    ### Input functions
    def tapered_gaussian(t, z, f0, c=200):
        #fc = np.sqrt(2)*f0
        ### Tapered Ricker propagating at speed c 
        sig = (1 - 2*((t-z/c)*f0*np.pi)**2) * np.exp(-np.pi**2*(t-z/c)**2*f0**2)   ### Ricker
        Nsine = int((8/f0/np.sqrt(2))/dt)
        if not Nsine%2==0: 
            Nsine+=1
        tp = tukey(Nsine, alpha=0.0)

        tap = np.zeros(t.shape)
        Nt0 = np.where(t<0)[0].size
        if np.isscalar(z):
            Nprop = int(z/c/dt)
            tap = np.pad(tp,(int(Nt0+Nprop-Nsine//2),int(Nt-Nt0-Nsine//2-Nprop)))
            sig*=tap
        else:
            for i in range(z.size):
                Nprop = int(z[i,0]/c/dt)
                tap = np.pad(tp,(int(Nt0+Nprop-Nsine//2),int(Nt-Nt0-Nsine//2-Nprop)))
                sig[i,:]*=tap
        return(sig)
    
    def tapered_gaussian_dz(t, z, f0, c=200):
        tau = t - z/c
        a = np.pi**2 * f0**2

        sig = (a/c) * tau * (6 - 4*a*tau**2) * np.exp(-a*tau**2)
        ### Tapered sinusoid propagating at speed c 
        # sig = (fc**2 / c) * (t - z/c) * (3 - fc**2 * (t - z/c)**2) \
        #          * np.exp(-0.5 * (t - z/c)**2 * fc**2)
        Nsine = int((8/f0/np.sqrt(2))/dt)
        if not Nsine%2==0: 
            Nsine+=1
        tp = tukey(Nsine, alpha=0.0)

        tap = np.zeros(t.shape)
        Nt0 = np.where(t<0)[0].size
        if np.isscalar(z):
            Nprop = int(z/c/dt)
            tap = np.pad(tp,(int(Nt0+Nprop-Nsine//2),int(Nt-Nt0-Nsine//2-Nprop)))
            sig*=tap
        else:
            for i in range(z.size):
                Nprop = int(z[i,0]/c/dt)
                tap = np.pad(tp,(int(Nt0+Nprop-Nsine//2),int(Nt-Nt0-Nsine//2-Nprop)))
                sig[i,:]*=tap
        return(sig)
    
    def tapered_gaussian_dt(t, z, f0, c=200):
        tau = t - z/c
        a = np.pi**2 * f0**2

        sig = a * tau * (4*a*tau**2 - 6) * np.exp(-a*tau**2)
        ### Tapered gaussian propagating at speed c 
        # sig = -fc**2 * (t - z/c) * (3 - fc**2 * (t - z/c)**2) \
        #         * np.exp(-0.5 * (t - z/c)**2 * fc**2)

        Nsine = int((8/f0/np.sqrt(2))/dt)
        if not Nsine%2==0: 
            Nsine+=1
        tp = tukey(Nsine, alpha=0.0)

        tap = np.zeros(t.shape)
        Nt0 = np.where(t<0)[0].size
        if np.isscalar(z):
            Nprop = int(z/c/dt)
            tap = np.pad(tp,(int(Nt0+Nprop-Nsine//2),int(Nt-Nt0-Nsine//2-Nprop)))
            sig*=tap
        else:
            for i in range(z.size):
                Nprop = int(z[i,0]/c/dt)
                tap = np.pad(tp,(int(Nt0+Nprop-Nsine//2),int(Nt-Nt0-Nsine//2-Nprop)))
                sig[i,:]*=tap
        return(sig)


    def tapered_sinusoid(t, z, f0, c=200):
        ### Tapered sinusoid propagating at speed c 
        sig = np.sin(2*np.pi*f0*(t - z/c))
        Nsine = int((1/f0)/dt)
        tp = tukey(Nsine, alpha=0.0)

        tap = np.zeros(t.shape)
        Nt0 = np.where(t<0)[0].size
        if np.isscalar(z):
            Nprop = int(z/c/dt)
            tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
            sig*=tap
        else:
            for i in range(z.size):
                Nprop = int(z[i,0]/c/dt)
                tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
                sig[i,:]*=tap
        return(sig)
    
    def tapered_sinusoid_dz(t, z, f0, c=200):
        ### Vertical gradient of tapered sinusoid 
        sig = -2*np.pi*f0/c*np.cos(2*np.pi*f0*(t - z/c))
        Nsine = int((1/f0)/dt)
        tp = tukey(Nsine, alpha=0.0)
        
        tap = np.zeros(t.shape)
        Nt0 = np.where(t<0)[0].size
        if np.isscalar(z):
            Nprop = int(z/c/dt)
            tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
            sig*=tap
            sig -= np.mean(sig)
        else:
            for i in range(z.size):
                Nprop = int(z[i,0]/c/dt)
                tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
                sig[i,:]*=tap
            sig -= np.mean(sig, axis=1)[:,None]
        return(sig)

    def tapered_sinusoid_dt(t, z, f0, c=200):
        ### Time derivative of tapered sinusoid
        sig = 2*np.pi*f0*np.cos(2*np.pi*f0*(t - z/c))
        Nsine = int((1/f0)/dt)
        tp = tukey(Nsine, alpha=0.0)
        
        tap = np.zeros(t.shape)
        Nt0 = np.where(t<0)[0].size
        if np.isscalar(z):
            Nprop = int(z/c/dt)
            tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
            sig*=tap
            sig -= np.mean(sig)
        else:
            for i in range(z.size):
                Nprop = int(z[i,0]/c/dt)
                tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
                sig[i,:]*=tap
            sig -= np.mean(sig, axis=1)[:,None]
        return(sig)

    ### To verify that the signal and its derivative is calculated corectly
    # fig, ax = plt.subplots() 
    # for fi, f in enumerate(freq):
    #     ax.plot(time, VEL[0,fi,:])
    #     # ax.plot(time, tapered_sinusoid(time, hstart, f), ls="--")
    #     ax.plot(time, tapered_sinusoid_dz(time, hstart, f), ls="--")
    #     ax.plot(time, np.gradient( tapered_sinusoid(time[None,:], hstart+np.linspace(-1e2,1e2,11)[:,None], f), hstart+np.linspace(-1e2,1e2,11), axis=0 )[5,:] , ls=":")
    # plt.show() 
    # quit()

    ### Same for gaussian/Ricker signal 
    fig, ax = plt.subplots() 
    for fi, f in enumerate(freq):
        # ax.plot(time, VEL[0,fi,:])
        ax.plot(time, tapered_gaussian(time, hstart, f), ls="--")
        # ax.plot(time, tapered_gaussian_dz(time, hstart, f), ls="--")
        # ax.plot(time, np.gradient( tapered_gaussian(time[None,:], hstart+np.linspace(-1e2,1e2,11)[:,None], f), hstart+np.linspace(-1e2,1e2,11), axis=0 )[5,:] , ls=":")
        # ax.plot(time, tapered_gaussian_dt(time, hstart, f), ls="--")
        # ax.plot(time, np.gradient( tapered_gaussian(time, hstart, f), time) , ls=":")
    # fig, ax = plt.subplots()
    # for fi, f in enumerate(freq):
    #     s = tapered_gaussian(time, hstart, f)
    #     fr = np.fft.rfftfreq(n = s.size, d=dt)
    #     ax.plot(fr, abs(np.fft.rfft(s)), ls="--")
    plt.show() 
    quit()


    ### Construct ground sinusoid for our own framework: 
    ### We use 4 different frequencies 
    VEL = np.zeros((1,freq.size, Nt))
    DIS = np.zeros((1,freq.size, Nt))
    if test=="kenda":
        for fi, f0 in enumerate(freq):
            VEL[0,fi,:] = tapered_sinusoid(time, 0e3, f0, c=c)
            DIS[0,fi,:] = integrate.cumulative_trapezoid(tapered_sinusoid(time, 0e3, f0, c=c), time, initial=0)
    elif test=="sutin":
        # for fi, f0 in enumerate(freq):
        #     VEL[0,fi,:] = tapered_sinusoid(time, 0e3, f0, c=c)
        #     DIS[0,fi,:] = integrate.cumulative_trapezoid(tapered_sinusoid(time, 0e3, f0, c=c), time, initial=0)
        ### Alternative for Sutin 2018: Construct ground gaussian for our own framework: 
        for fi, f0 in enumerate(freq):
            VEL[0,fi,:] = tapered_gaussian(time, 0e3, f0, c=c)
            DIS[0,fi,:] = integrate.cumulative_trapezoid(tapered_gaussian(time, 0e3, f0, c=c), time, initial=0)
        
    north_shifts = freq 
    east_shifts = np.array([0. for i in range(north_shifts.size)])

    ### Prepare the SEISMO dictionay that is absorbed by the AirglowSignal class
    SEISMO ={"dt": dt,
             "t_new": time,
             "Nt": Nt,
             "VEL": VEL, 
             "DIS": DIS,
             "Nn" : north_shifts.size,  
             "Ne" : 1 ,
             "EE": east_shifts, 
             "NN" : north_shifts,
             "iEE" : [0 for i in freq],
             "iNN" : range(north_shifts.size),
             "gridded": False, 
    }

    ################################################################################################################
    ### FRAMEWORK CALCUATION (using above classes): 
    ################################################################################################################    
    AR = AirglowSignal(SEISMO, Nz = 500, do_plot=False, disable_att=True)
    ### Attenuation is disabled, as in Kenda (2018)  
    ### We ensure a constant velocity with 
    AR.f_c = lambda z: 200*np.ones(z.shape)
    ### Check wavelength of ver to ensure precision of integration: 
    lambda_min = np.min(AR.f_c(AR.z_1_27_calc_m))/max(freq)
    dz_min = AR.dz_1_27_m
    print("Min wavelength of VER signal = {:.1f} m".format(lambda_min))
    print("Achieved vertical resolution = {:.1f} m".format(dz_min))
    if lambda_min<=2*dz_min:
        print("WARNING: vertical resolution of integration might be insufficient for desired frequency")

    ### To concord with Kenda or Sutin, the amplitude of the wave 
    ### must be 5 mm/s at 90 km or 40 mm/s at 100 km. 
    ### We therefore rescale VEL using the amplification function 
    AR.VEL = AR.VEL * ampl_at_start/AR.f_amplification(hstart)
    AR.DIS = AR.DIS * ampl_at_start/AR.f_amplification(hstart)

    ### Calculation of Airglow using the above framework  
    list_inorth, list_ieast = AR.iNN, AR.iEE
    AR.calculate_1_27_airglow(list_ieast, list_inorth, loc_save_idx=[], time_save = time, fourier_filtering=True, 
                               do_parallel=False, dir_save="./results_test/")
    ### Load saved data 
    dat = np.load("results_test/nightglow_dver_t.npy")
    dati = np.load("results_test/nightglow_I_t.npy")
    vz_z = dat[0,:,:,:,0]
    ver_z = dat[0,:,:,:,1]
    I1 = dati[0,:,:]
    ### Select altitude closest to 90 km or 100 km 
    iz = np.argmin(abs(AR.z_1_27_calc_m - hstart))
    print("Test altitude: index {:d}, {:.1f} km".format(iz, AR.z_1_27_calc_m[iz]/1e3) )

    ################################################################################################################
    ### Show waveforms at different frequencies 
    ################################################################################################################
    fig, (ax2, ax3, ax4) = plt.subplots(3,1, sharex = True, figsize=(7,9)) 
    cmap = plt.get_cmap("plasma")
    cols = [cmap(i) for i in np.linspace(0.2, 0.8, freq.size)]
    ### Propagated at 90 km 
    for fi, f in enumerate(freq):

        ### Starting waveform at 90 km 
        ax2.plot(time, vz_z[fi,iz,:], c=cols[fi], label="f={:.3g} Hz, T={:.3g} s".format(f, 1/f))
        if fi == freq.size-1:
            ax2.set_ylim(-1.5*ampl_at_start,1.5*ampl_at_start)
        ### Calculated VER 
        vmin = -np.max(np.abs(ver_z[fi,iz,:]))*1.1
        vmax = np.max(np.abs(ver_z[fi,iz,:]))*1.1
        # print(vmin, vmax)
        ax3.plot(time, ver_z[fi,iz,:]/AR.f_VER_1_27(AR.z_1_27_calc_m[iz])*100, c=cols[fi])
        if fi == freq.size-1:
            ax3b = ax3.twinx() 
            ax3b.plot(time, ver_z[fi,iz,:], ls=" ", color="w")    
            ax3.set_ylim(vmin/AR.f_VER_1_27(AR.z_1_27_calc_m[iz])*100, vmax/AR.f_VER_1_27(AR.z_1_27_calc_m[iz])*100)
            ax3b.set_ylim(vmin, vmax)
        ### Integrated Intensity
        imin = -np.max(np.abs(I1[fi,:]))*1.1
        imax = np.max(np.abs(I1[fi,:]))*1.1
        ax4.plot(time, I1[fi,:]/AR.I_background_nightglow*100, c=cols[fi])
        ### NOTE: No need to conversion to Rayleigh here, it is already done. 
        if fi== freq.size-1:
            ax4b = ax4.twinx() 
            ax4b.plot(time, I1[fi,:], c="k", ls=" ")
            ax4.set_ylim(imin/AR.I_background_nightglow*100, imax/AR.I_background_nightglow*100)
            ax4b.set_ylim(imin, imax)
    ###
    ax2.set_ylabel(r"$V_z$ at " + "{:.1f} km".format(AR.z_1_27_calc_m[iz]/1e3) + r" / [$m/s$]")
    ax3.set_ylabel("VER$_{1.27}$ pert. at " + "{:.1f} km".format(AR.z_1_27_calc_m[iz]/1e3) + r" / [%]")
    ax3b.set_ylabel(r"VER$_{1.27}$ pert. at " + "{:.1f} km".format(AR.z_1_27_calc_m[iz]/1e3) + r" / [$ph/m^3/s$]")
    ax4.set_ylabel("Relative Intensity pert. [%]")
    ax4b.set_ylabel("Intensity pert. [$Rayleigh$]")
    ax4.set_xlabel("Time / [$s$]")
    ax4.set_xlim(-10,tf)
    ax2.legend(frameon=False)
    for ax in  [ax2, ax3, ax4, ax3b, ax4b]:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2), useMathText=True)
    ###
    fig.align_labels() 
    fig.suptitle("NIGHTGLOW, calculated with Airglow framework")
    fig.savefig("./results_test/nightglow_wv_" + test + "_framework.pdf")
    fig.savefig("./results_test/nightglow_wv_" + test + "_framework.png", dpi=600)
    fig.tight_layout()
    ################################################################################################################

    

    ################################################################################################################
    ### Calculate dVER for sinusoid of 50 s 
    ################################################################################################################
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,7))
    fi_50 = 2
    ###
    ax1.plot(AR.f_VER_1_27(AR.z_1_27_calc_m), AR.z_1_27_calc_km, c="k") 
    ax1b = ax1.twiny()
    ax1b.plot(np.max(abs(ver_z[fi_50,:,:]), axis=1)/AR.f_VER_1_27(AR.z_1_27_calc_m)*100, AR.z_1_27_calc_km, c="r", label="Max Perturbation")
    ax1b.set_xlabel(r"Max. VER$_{1.27}$ perturbation / [% background]")
    ax1b.xaxis.label.set_color('red')
    ax1.set_xlabel("VER / ph/m3/s")
    ax1.set_ylabel("Altitude / km")
    ax1.set_xlim(0,5.5e11)
    ax1b.set_xlim(0,1)
    ###
    ax2.plot(AR.f_amplification(AR.z_1_27_calc_m)/AR.f_amplification(90e3), AR.z_1_27_calc_km, c="k", label=r"$\sqrt{\rho(90)c(90)/(\rho(z)c(z))}$") 
    ax2.set_xlabel("Amplification with respect to 90 km")
    ax2.legend(frameon=False, loc=4)
    ax2.set_xlim(0,70)
    ###
    im = ax3.pcolormesh(time, AR.z_1_27_calc_km, ver_z[fi_50,:,:], 
                        vmin=-np.max(np.abs(ver_z[fi_50,:,:])), 
                        vmax=np.max(np.abs(ver_z[fi_50,:,:])), rasterized=True)
    ax3.axhline(AR.z_1_27_calc_km[np.argmax(np.max(abs(ver_z[fi_50,:,:]), axis=1))], c="w", ls=":", label="Maximum perturbation")
    ax3.axhline(AR.z_1_27_calc_km[np.argmax(AR.f_VER_1_27(AR.z_1_27_calc_m))], c="w", ls="--", label=r"Maximum VER$_{1.27}$")
    fig.colorbar(im, ax=ax3, label=r"$\Delta$VER$_{1.27}$ [$ph/m^3/s$]")
    ax3.set_xlabel("Time / [$s$]")
    leg = ax3.legend(frameon=False, loc=2)
    for text in leg.get_texts():
        text.set_color('w') # Set all legend text to green
    # ax3.set_ylabel("Altitude / [$km$]")
    ax3.set_xlim(400,800)
    ax3.set_ylim(90,120)
    for ax in [ax1,ax2,ax3]:
        ax.set_ylim(90,120)
    fig.suptitle("NIGHTGLOW, calculated with Airglow framework")
    fig.savefig("./results_test/nightglow_2d_" + test + "_framework.pdf")
    fig.savefig("./results_test/nightglow_2d_" + test + "_framework.png", dpi=600)
    fig.tight_layout()
    ################################################################################################################


    ################################################################################################################
    ### PLOTTING THE DIFFERENT DIVERGENCE TERMS
    ################################################################################################################
    tp = 440 
    it = np.argmin(abs(time-tp))
    fig, (ax, axt) = plt.subplots(1,2,figsize=(8,7)) 
    ###
    # ax.plot( np.gradient(vz_z[fi_50,:,it], AR.z_1_27_calc_m), AR.z_1_27_calc_km, c="r", label=r"$\nabla \cdot v_z$")
    # ax.plot( np.gradient(AR.f_VER_1_27(AR.z_1_27_calc_m), AR.z_1_27_calc_m), AR.z_1_27_calc_km, c="k", label=r"$\nabla \cdot VER(z)$")
    # ax.plot( AR.f_dVER_1_27(AR.z_1_27_calc_m), AR.z_1_27_calc_km, c="k", label=r"$\nabla \cdot VER(z)$")
    ax.plot( AR.f_VER_1_27(AR.z_1_27_calc_m)*np.gradient(vz_z[fi_50,:,it], AR.z_1_27_calc_m), AR.z_1_27_calc_km, c="k", ls="-", label=r"$VER \;\nabla \cdot v_z$")
    ax.plot( np.gradient(AR.f_VER_1_27(AR.z_1_27_calc_m)*vz_z[fi_50,:,it], AR.z_1_27_calc_m), AR.z_1_27_calc_km, c="k", ls="--", label=r"$\nabla \cdot (VER \;v_z)$")
    ax.plot( integrate.cumulative_trapezoid(vz_z[fi_50,:,:], time, axis=1, initial=0)[:,it]  * AR.f_dVER_1_27(AR.z_1_27_calc_m), AR.z_1_27_calc_km, c="k", ls=":", label=r"$u_z \cdot \nabla VER$")
    ax.set_ylabel("Altitude / [$km$]")
    ax.set_ylabel("Divergence terms")
    ax.legend(frameon=False)
    ###
    axt.plot(time, 1/(2*np.pi/50) * integrate.trapezoid(AR.f_VER_1_27(AR.z_1_27_calc_m)[:,None]*np.gradient(vz_z[fi_50,:,:], AR.z_1_27_calc_m, axis=0), AR.z_1_27_calc_m, axis=0) ,
                     c="k", ls="-", label=r"Integrated, $1/\omega\;VER \;\nabla \cdot v_z$")
    axt.plot(time, 1/(2*np.pi/50) * integrate.trapezoid(-AR.f_dVER_1_27(AR.z_1_27_calc_m)[:,None]*vz_z[fi_50,:,:], AR.z_1_27_calc_m, axis=0) ,
                     c="purple", ls="--", label=r"Integrated, -$1/\omega\;v_z\cdot \nabla VER$")
    axt.plot(time, 1/(2*np.pi/50) * integrate.trapezoid( np.gradient(AR.f_VER_1_27(AR.z_1_27_calc_m)[:,None]*vz_z[fi_50,:,:], AR.z_1_27_calc_m, axis=0), AR.z_1_27_calc_m, axis=0), 
                     c="r", ls="--", label=r"Integrated $1/\omega\;\nabla \cdot (VER \;v_z)$")
    axt.plot(time, integrate.trapezoid(   integrate.cumulative_trapezoid(vz_z[fi_50,:,:], time, axis=1, initial=0)  * AR.f_dVER_1_27(AR.z_1_27_calc_m)[:,None], AR.z_1_27_calc_m,axis=0),
                     c="k", ls=":", label=r"$u_z \cdot \nabla VER$")
    ### WITH FILTER 
    ### lfilter(b, a, dver_vz_z, axis=1)
    # axt.plot(time, integrate.trapezoid(lfilter(AR.b, AR.a, AR.f_VER_1_27(AR.z_1_27_calc_m)[:,None]*np.gradient(vz_z[fi_50,:,:], AR.z_1_27_calc_m, axis=0), axis=1), AR.z_1_27_calc_m, axis=0) ,
    #                  c="k", ls="-", label=r"Integrated, $\tau\;VER \;\nabla \cdot v_z$")
    # axt.plot(time, integrate.trapezoid(lfilter(AR.b, AR.a,  np.gradient(AR.f_VER_1_27(AR.z_1_27_calc_m)[:,None]*vz_z[fi_50,:,:], AR.z_1_27_calc_m, axis=0), axis=1), AR.z_1_27_calc_m, axis=0), 
    #                  c="k", ls="--", label=r"Integrated $\tau\;\nabla \cdot (VER \;v_z)$")
    # axt.plot(time, integrate.trapezoid(   integrate.cumulative_trapezoid(vz_z[fi_50,:,:], time, axis=1, initial=0)  * AR.f_dVER_1_27(AR.z_1_27_calc_m)[:,None], AR.z_1_27_calc_m,axis=0),
    #                  c="k", ls=":", label=r"$u_z \cdot \nabla VER$")
    
    axt.set_xlabel("Time / [$s$]")
    axt.set_ylabel("Unfiltered intensity")
    axt.legend(frameon=False, loc=1)
    ###
    fig.tight_layout()
    # plt.show()
    # print("here 3")

    


    ################################################################################################################
    ### HANDMADE CALCUATION (analytical sinusoid): 
    ################################################################################################################    
    ### 0. Pick frequency 
    # f0s = np.array([0.2,0.1,0.02, 0.01])
    f0s = freq
    ### 1. Make up an altitude range  (must be wider than the VER region!)
    zrange = np.linspace(80e3,140e3,400)
    ### 2. Get frequencies 
    ff = np.fft.fftfreq(Nt, d=dt)
    om = ff*(2*np.pi)
    ### 4. Get amplification, normalize by the value at 90/100 km: 
    a_func = AR.f_amplification(zrange)[:,None]/AR.f_amplification(hstart)
    
    ### Function to generate a sinusoid at a specific frequency and its dVER 
    # def calculate_airglow_sinusoid(f0):
    #     ### 5. Defined derivative of tapered sinusoid (functions above) and normalize by amplitude 
    #     tVEL = tapered_sinusoid(time[None,:], zrange[:,None], f0, c=c) * ampl_at_start 
    #     tVEL_dz = tapered_sinusoid_dz(time[None,:], zrange[:,None], f0, c=c) * ampl_at_start 
    #     print("Max. of velocity signal at z = 90 km: ", np.max(tVEL[0,:]))

    #     ### 6. Get their fft
    #     fftVEL = np.fft.fft(tVEL, axis=1)
    #     fftVEL_dz = np.fft.fft(tVEL_dz, axis=1)

    #     ### 7. Calculate the desired dver in frequency domain (EQ 4.22): 
    #     tau = 4460 # s 

    #     ### OPTION 1: Calculate analytical derivative of sine. Can cause problems due to tapering.
    #     # fft_dver = -tau/(1+1j*om[None,:]*tau) * AR.f_VER_1_27(zrange)[:,None] * (fftVEL_dz*a_func  + fftVEL*np.gradient(a_func, zrange, axis=0) ) 
    #     ### OPTION 2: Calculate gradient of vel manually: 
    #     fft_dver = -tau/(1+1j*om[None,:]*tau) * AR.f_VER_1_27(zrange)[:,None] * (np.gradient(fftVEL, zrange, axis=0)*a_func  + fftVEL*np.gradient(a_func, zrange, axis=0) ) 
        
    #     ### 8. Back to time domain 
    #     dver = np.fft.ifft(fft_dver, axis=1).real

    #     ### 9. Remove linear trend. 
    #     start = dver[:,0][:,None]
    #     end   = dver[:,50][:,None]
    #     trend = np.linspace(0, 1, dver.shape[1])   
    #     trend = start + (end - start)/(trend[50]-trend[0]) * trend  # shape (Nz, Nt)
    #     dver = dver - trend

    #     ### 10. Integrated intensity 
    #     I = np.trapz(dver, zrange, axis=0 )
    #     return(tVEL, dver, I)
    # I_background = np.trapz(AR.f_VER_1_27(zrange), zrange, axis=0 )

    ### Even simpler method: define derivative of sinusoid explicitely 
    zatt = np.concatenate((np.linspace(0,zrange[0]-1e3,200),zrange))
    phase = np.exp(-2*1j*np.pi*ff[None,:]*integrate.cumulative_trapezoid(1/AR.f_c(zatt), zatt, initial=0)[:,None])
    dphase_dz = -2*1j*np.pi*ff[None,:]*1/AR.f_c(zatt)[:,None]*phase 
    ###
    ampl = AR.f_amplification(zatt)[:,None]/AR.f_amplification(hstart)
    dampl_dz = np.gradient(ampl, zatt, axis=0) 
    
    def calculate_airglow_perturbation(f0):
        if test=="kenda":
            tVEL = tapered_sinusoid(time[None,:], zatt[:,None]*0, f0, c=c) * ampl_at_start 
        elif test=="sutin":
            # tVEL = tapered_sinusoid(time[None,:], zatt[:,None]*0, f0, c=c) * ampl_at_start 
            tVEL = tapered_gaussian(time[None,:], zatt[:,None]*0, f0, c=c) * ampl_at_start
        fftVEL = np.fft.fft(tVEL, axis=1)
        ### VEL(z,f) = VEL(0,f)*exp(-2*1j*pi*f*int_0_z(1/c dz))*exp(-int_0_z(alpha dz)*ampl(z))
        ### dVEL(z,f)_dz = VEL(0,f)*(dphase_dz⋅ampl⋅att + dampl_dz⋅phase⋅att + datt_dz⋅phase⋅ampl)
        fft_dver = fftVEL*(dphase_dz*ampl + dampl_dz*phase)*AR.f_VER_1_27(zatt)[:,None]
        fft_dver = fft_dver[-zrange.size:,:]
        ###
        tVEL = np.fft.ifft(fftVEL*ampl*phase, axis=1).real
        tVEL = tVEL[-zrange.size:,:]
        ###
        tau = 4460 # s 
        dver = np.fft.ifft(fft_dver*-tau/(1+1j*om[None,:]*tau), axis=1).real
        ###
        I = np.trapz(fft_dver, zrange, axis=0 )
        I = np.fft.ifft(I*-tau/(1+1j*om*tau)).real
        ### 9. Remove linear trend. 
        start = dver[:,0][:,None]
        end   = dver[:,50][:,None]
        trend = np.linspace(0, 1, dver.shape[1])   
        trend = start + (end - start)/(trend[50]-trend[0]) * trend  # shape (Nz, Nt)
        dver = dver - trend
        ###
        start = I[0]
        end   = I[50]
        trend = np.linspace(0, 1, I.size)   
        trend = start + (end - start)/(trend[50]-trend[0]) * trend  # shape (Nz, Nt)
        I = I - trend
        ###
        return(tVEL, dver, I)
    I_background = np.trapz(AR.f_VER_1_27(zrange), zrange, axis=0 )


    ################################################################################################################
    ### Show waveforms at different frequencies 
    ################################################################################################################
    fig, (ax2, ax3, ax4) = plt.subplots(3,1, sharex = True, figsize=(7,9)) 
    cmap = plt.get_cmap("plasma")
    cols = [cmap(i) for i in np.linspace(0.2, 0.8, f0s.size)]
    ### To ensure we plot the same altitude as the other method 
    iz2 = np.argmin(abs(zrange-AR.z_1_27_calc_m[iz])) 

    for fi, f0 in enumerate(f0s):
        ### Do the calculation for f0 
        tVEL, dver, I = calculate_airglow_perturbation(f0)

        ### Starting waveform at 90 km 
        ax2.plot(time, tVEL[iz2,:], c=cols[fi], label="f={:.3g} Hz, T={:.3g} s".format(f0, 1/f0))
        if fi == f0s.size-1:
            ax2.set_ylim(-1.5*ampl_at_start,1.5*ampl_at_start)
        ### Calculated VER 
        vmin = -np.max(np.abs(dver[iz2,:]))*1.1
        vmax = np.max(np.abs(dver[iz2,:]))*1.1
        ax3.plot(time, dver[iz2,:]/AR.f_VER_1_27(zrange[iz2])*100, c=cols[fi])
        if fi == f0s.size-1:
            ax3b = ax3.twinx() 
            ax3b.plot(time, dver[iz2,:], ls=" ", color="w")    
            ax3.set_ylim(vmin/AR.f_VER_1_27(zrange[iz2])*100, vmax/AR.f_VER_1_27(zrange[iz2])*100)
            ax3b.set_ylim(vmin, vmax)
        ### Integrated Intensity
        imin = -np.max(np.abs(I))*1.1
        imax = np.max(np.abs(I))*1.1
        ax4.plot(time, I/I_background*100, c=cols[fi])
        if fi== f0s.size-1:
            ax4b = ax4.twinx() 
            ax4b.plot(time, I*factor_W_to_Rayleigh(1.27, dir="phRadiance_to_Rayleigh"), c="k", ls=" ")
            ax4.set_ylim(imin/I_background*100, imax/I_background*100)
            ax4b.set_ylim(imin*factor_W_to_Rayleigh(1.27, dir="phRadiance_to_Rayleigh"), imax*factor_W_to_Rayleigh(1.27, dir="phRadiance_to_Rayleigh"))
    ###
    ax2.set_ylabel(r"$V_z$ at " + "{:.1f} km".format(zrange[iz2]/1e3) + r" / [$m/s$]")
    ax3.set_ylabel("VER$_{1.27}$ pert. at " + "{:.1f} km".format(zrange[iz2]/1e3) + r" / [%]")
    ax3b.set_ylabel(r"VER$_{1.27}$ pert. at " + "{:.1f} km".format(zrange[iz2]/1e3) + r" / [$ph/m^3/s$]")
    ax4.set_ylabel("Relative Intensity pert. [%]")
    ax4b.set_ylabel("Intensity pert. [$Rayleigh$]")
    ax4.set_xlabel("Time / [$s$]")
    ax4.set_xlim(-10,tf)
    ax2.legend(frameon=False)
    for ax in  [ax2, ax3, ax4, ax3b, ax4b]:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2), useMathText=True)
    ###
    fig.align_labels() 
    fig.suptitle("NIGHTGLOW, calculated with Homemade sine framework")
    fig.savefig("./results_test/nightglow_wv_" + test + "_sine.pdf")
    fig.savefig("./results_test/nightglow_wv_" + test + "_sine.png", dpi=600)
    fig.tight_layout()
    ################################################################################################################
    

    ################################################################################################################
    ### Calculate for T=50s 
    ################################################################################################################
    tVEL, dver, I = calculate_airglow_perturbation(1/50)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,7))
    ###
    ax1.plot(AR.f_VER_1_27(zrange), zrange/1e3, c="k") 
    ax1b = ax1.twiny()
    ax1b.plot(np.max(abs(dver), axis=1)/AR.f_VER_1_27(zrange)*100, zrange/1e3, c="r", label="Max Perturbation")
    ax1b.set_xlabel(r"Max. VER$_{1.27}$ perturbation / [% background]")
    ax1b.xaxis.label.set_color('red')
    ax1.set_xlabel("VER / ph/m3/s")
    ax1.set_ylabel("Altitude / km")
    ax1.set_xlim(0,5.5e11)
    ax1b.set_xlim(0,1)
    ###
    ax2.plot(AR.f_amplification(zrange)/AR.f_amplification(90e3), zrange/1e3, c="k", label=r"$\sqrt{\rho(90)c(90)/(\rho(z)c(z))}$") 
    ax2.set_xlabel("Amplification with respect to 90 km")
    ax2.legend(frameon=False, loc=4)
    ax2.set_xlim(0,70)
    ###
    im = ax3.pcolormesh(time, zrange/1e3, dver[:,:], vmin=-np.max(np.abs(dver)), vmax=np.max(np.abs(dver)))
    ax3.axhline(zrange[np.argmax(np.max(abs(dver), axis=1))]/1e3, c="w", ls=":", label="Maximum perturbation")
    ax3.axhline(zrange[np.argmax(AR.f_VER_1_27(zrange))]/1e3, c="w", ls="--", label=r"Maximum VER$_{1.27}$")
    fig.colorbar(im, ax=ax3, label=r"$\Delta$VER$_{1.27}$ [$ph/m^3/s$]")
    ax3.set_xlabel("Time / [$s$]")
    leg = ax3.legend(frameon=False, loc=2)
    for text in leg.get_texts():
        text.set_color('w') # Set all legend text to green
    # ax3.set_ylabel("Altitude / [$km$]")
    ax3.set_xlim(400,800)
    ax3.set_ylim(90,120)
    for ax in [ax1,ax2,ax3]:
        ax.set_ylim(90,120)
    fig.suptitle("NIGHTGLOW, calculated with Homemade sine framework")
    fig.savefig("./results_test/nightglow_2d_" + test + "_sine.pdf")
    fig.savefig("./results_test/nightglow_2d_" + test + "_sine.png", dpi=600)
    fig.tight_layout()
    # plt.show()
    ################################################################################################################
    

def check_simple_perturbation_dayglow(test="kenda"):
    ### As in Balthasar Kenda's thesis, we send a sinusoidal perturbation with amplitude defined at 90 km 

    ### Time parameters 
    dt = 0.1               ### s
    tf = 1000              ### s
    time = np.arange(-tf/10, tf, dt)
    Nt = time.size

    ### Input waveform parameters 
    freq = np.array([0.2,0.1,0.02, 0.01]) ### Hz
    # freq = np.array([0.01]) ### Hz
    if test=="kenda":
        c = 200                ### Fixed vertical propagation velocity m/s
        hstart = 90e3          ### m   
        ampl_at_start = 5e-3   ### Ampl of Mw 6.5, 30degree distance, 90 km alt (kenda) 


    def tapered_sinusoid(t, z, f0, c=200):
        ### Tapered sinusoid propagating at speed c 
        sig = np.sin(2*np.pi*f0*(t - z/c))
        Nsine = int((1/f0)/dt)
        tp = tukey(Nsine, alpha=0.0)

        tap = np.zeros(t.shape)
        Nt0 = np.where(t<0)[0].size
        if np.isscalar(z):
            Nprop = int(z/c/dt)
            tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
            sig*=tap
        else:
            for i in range(z.size):
                Nprop = int(z[i,0]/c/dt)
                tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
                sig[i,:]*=tap
        return(sig)
    
    def tapered_sinusoid_dz(t, z, f0, c=200):
        ### Vertical gradient of tapered sinusoid 
        sig = -2*np.pi*f0/c*np.cos(2*np.pi*f0*(t - z/c))
        Nsine = int((1/f0)/dt)
        tp = tukey(Nsine, alpha=0.0)
        
        tap = np.zeros(t.shape)
        Nt0 = np.where(t<0)[0].size
        if np.isscalar(z):
            Nprop = int(z/c/dt)
            tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
            sig*=tap
            sig -= np.mean(sig)
        else:
            for i in range(z.size):
                Nprop = int(z[i,0]/c/dt)
                tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
                sig[i,:]*=tap
            sig -= np.mean(sig, axis=1)[:,None]
        return(sig)

    def tapered_sinusoid_dt(t, z, f0, c=200):
        ### Time derivative of tapered sinusoid
        sig = 2*np.pi*f0*np.cos(2*np.pi*f0*(t - z/c))
        Nsine = int((1/f0)/dt)
        tp = tukey(Nsine, alpha=0.0)
        
        tap = np.zeros(t.shape)
        Nt0 = np.where(t<0)[0].size
        if np.isscalar(z):
            Nprop = int(z/c/dt)
            tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
            sig*=tap
            sig -= np.mean(sig)
        else:
            for i in range(z.size):
                Nprop = int(z[i,0]/c/dt)
                tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
                sig[i,:]*=tap
            sig -= np.mean(sig, axis=1)[:,None]
        return(sig)

    ### To verify that the signal and its derivative is calculated corectly
    # fig, ax = plt.subplots() 
    # for fi, f in enumerate(freq):
    #     ax.plot(time, VEL[0,fi,:])
    #     # ax.plot(time, tapered_sinusoid(time, hstart, f), ls="--")
    #     ax.plot(time, tapered_sinusoid_dz(time, hstart, f), ls="--")
    #     ax.plot(time, np.gradient( tapered_sinusoid(time[None,:], hstart+np.linspace(-1e2,1e2,11)[:,None], f), hstart+np.linspace(-1e2,1e2,11), axis=0 )[5,:] , ls=":")
    # plt.show() 
    # quit()

    ### Construct ground sinusoid for our framework
    VEL = np.zeros((1,freq.size, Nt))
    DIS = np.zeros((1,freq.size, Nt))
    ### Construct ground sinusoid for our own framework. 
    ### This time the displacement is a sinusoid, the velocity its derivative.
    for fi, f0 in enumerate(freq):
        VEL[0,fi,:] = tapered_sinusoid_dt(time, 0e3, f0, c=c)
        DIS[0,fi,:] = tapered_sinusoid(time, 0e3, f0, c=c)
    ### Normalize velocity and displacement so that max(VEL)=1
    tvmax = np.max(np.abs(VEL), axis=2)
    VEL=VEL/tvmax[:,:,None] 
    DIS=DIS/tvmax[:,:,None] 

    north_shifts = freq 
    east_shifts = np.array([0. for i in range(north_shifts.size)])

    ### Prepare the SEISMO dictionay that is absorbed by the AirglowSignal class
    SEISMO ={"dt": dt,
             "t_new": time,
             "Nt": Nt,
             "VEL": VEL, 
             "DIS": DIS,
             "Nn" : north_shifts.size,  
             "Ne" : 1 ,
             "EE": east_shifts, 
             "NN" : north_shifts,
             "iEE" : [0 for i in freq],
             "iNN" : range(north_shifts.size),
             "gridded": False, 
    }

    ################################################################################################################
    ### FRAMEWORK CALCUATION (using above classes): 
    ################################################################################################################    
    AR = AirglowSignal(SEISMO, Nz = 500, do_plot=False, disable_att=False)
    ### Attenuation is NOT disabled, as in Kenda (2018)  
    ### However, we ensure a constant velocity with 
    AR.f_c = lambda z: 200*np.ones(z.shape)
    ### Check wavelength of ver to ensure precision of integration: 
    lambda_min = np.min(AR.f_c(AR.z_4_28_calc_m))/max(freq)
    dz_min = AR.dz_4_28_m
    print("Min wavelength of VER signal = {:.1f} m".format(lambda_min))
    print("Achieved vertical resolution = {:.1f} m".format(dz_min))
    if lambda_min<=2*dz_min:
        print("WARNING: vertical resolution of integration might be insufficient for desired frequency")

    ### To concord with Kenda or Sutin, the amplitude of the wave 
    ### must be 5 mm/s at 90 km or 40 mm/s at 100 km. 
    ### We therefore rescale VEL using the amplification function 
    AR.VEL = AR.VEL * ampl_at_start/AR.f_amplification(hstart)
    AR.DIS = AR.DIS * ampl_at_start/AR.f_amplification(hstart)

    ### Calculation of Airglow using the above framework  
    list_inorth, list_ieast = AR.iNN, AR.iEE
    AR.calculate_4_28_airglow(list_ieast, list_inorth, loc_save_idx=[], time_save = time, 
                               do_parallel=False, dir_save="./results_test/")
    ### Load saved data 
    dat = np.load("results_test/dayglow_dver_t.npy")
    dati = np.load("results_test/dayglow_I_t.npy")
    vz_z = dat[0,:,:,:,0]
    ver_z = dat[0,:,:,:,1]
    I1 = dati[0,:,:]
    ### Select altitude closest to 90 km or 100 km 
    iz = np.argmin(abs(AR.z_4_28_calc_m - hstart))
    print("Test altitude: index {:d}, {:.1f} km".format(iz, AR.z_4_28_calc_m[iz]/1e3) )

    ################################################################################################################
    fig, (ax2, ax3, ax4) = plt.subplots(3,1, sharex = True, figsize=(7,9)) 
    cmap = plt.get_cmap("plasma")
    cols = [cmap(i) for i in np.linspace(0.2, 0.8, freq.size)]
    ### Propagated at 90 km 
    for fi, f in enumerate(freq):
        ### Starting waveform at 90/100 km 
        ax2.plot(time, np.gradient(vz_z[fi,iz,:], time), c=cols[fi], label="f={:.3g} Hz, T={:.3g} s".format(f, 1/f))
        if fi == freq.size-1:
            ax2.set_ylim(-1.5*ampl_at_start,1.5*ampl_at_start)
        ### Calculated VER 
        vmin = -np.max(np.abs(ver_z[fi,iz,:]))*1.1
        vmax = np.max(np.abs(ver_z[fi,iz,:]))*1.1
        ax3.plot(time, ver_z[fi,iz,:]/AR.f_VER_4_28(AR.z_4_28_calc_m[iz])*100, c=cols[fi])
        if fi == freq.size-1:
            ax3b = ax3.twinx() 
            ax3b.plot(time, ver_z[fi,iz,:], ls=" ", color="w")    
            ax3.set_ylim(vmin/AR.f_VER_4_28(AR.z_4_28_calc_m[iz])*100, vmax/AR.f_VER_4_28(AR.z_4_28_calc_m[iz])*100)
            ax3b.set_ylim(vmin, vmax)
        ### Integrated Intensity
        imin = -np.max(np.abs(I1[fi,:]))*1.1
        imax = np.max(np.abs(I1[fi,:]))*1.1
        ax4.plot(time, I1[fi,:]/AR.I_background_dayglow*100, c=cols[fi])
        ### NOTE: No need to conversion to Rayleigh here, it is already done. 
        if fi== freq.size-1:
            ax4b = ax4.twinx() 
            ax4b.plot(time, I1[fi,:], c="k", ls=" ")
            ax4.set_ylim(imin/AR.I_background_dayglow*100, imax/AR.I_background_dayglow*100)
            ax4b.set_ylim(imin, imax)
    ###
    ax2.set_ylabel(r"$V_z$ at " + "{:.1f} km".format(AR.z_4_28_calc_m[iz]/1e3) + r" / [$m/s$]")
    ax3.set_ylabel("VER$_{4.28}$ pert. at " + "{:.1f} km".format(AR.z_4_28_calc_m[iz]/1e3) + r" / [%]")
    ax3b.set_ylabel(r"VER$_{4.28}$ pert. at " + "{:.1f} km".format(AR.z_4_28_calc_m[iz]/1e3) + r" / [$ph/m^3/s$]")
    ax4.set_ylabel("Relative Intensity pert. [%]")
    ax4b.set_ylabel("Intensity pert. [$Rayleigh$]")
    ax4.set_xlabel("Time / [$s$]")
    ax4.set_xlim(-10,tf)
    ax2.legend(frameon=False)
    for ax in  [ax2, ax3, ax4, ax3b, ax4b]:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2), useMathText=True)
    ###
    fig.align_labels() 
    fig.suptitle("DAYGLOW, calculated with Airglow framework")
    fig.savefig("./results_test/dayglow_wv_" + test + "_framework.pdf")
    fig.savefig("./results_test/dayglow_wv_" + test + "_framework.png", dpi=600)
    fig.tight_layout()
    ################################################################################################################


    

    ################################################################################################################
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,7))
    fi_50 = 2
    ###
    ax1.plot(AR.f_VER_4_28(AR.z_4_28_calc_m), AR.z_4_28_calc_km, c="k") 
    ax1b = ax1.twiny()
    ax1b.plot(np.max(abs(ver_z[fi_50,:,:]), axis=1)/AR.f_VER_4_28(AR.z_4_28_calc_m)*100, AR.z_4_28_calc_km, c="r", label="Max Perturbation")
    ax1b.set_xlabel(r"Max. VER$_{4.28}$ perturbation / [% background]")
    ax1b.xaxis.label.set_color('red')
    ax1.set_xlabel("VER / ph/m3/s")
    ax1.set_ylabel("Altitude / km")
    ax1.set_xlim(0,7e12)
    ###
    ax2.plot(AR.f_amplification(AR.z_4_28_calc_m)/AR.f_amplification(90e3), AR.z_4_28_calc_km, c="k", label=r"$\sqrt{\rho(90)c(90)/(\rho(z)c(z))}$") 
    ax2.set_xlabel("Amplification with respect to 90 km")
    ax2.legend(frameon=False, loc=4)
    ax2.set_xscale("log")
    ax2.set_xlim(1,1e4)
    ###
    fac=AR._factor_photons_watt(4.28, dir="ps_to_W")
    # print(fac)
    im = ax3.pcolormesh(time, AR.z_4_28_calc_km, ver_z[fi_50,:,:]*fac, 
                        vmin=-np.max(np.abs(ver_z[fi_50,:,:]))*fac, 
                        vmax=np.max(np.abs(ver_z[fi_50,:,:]))*fac, rasterized=True)
    ax3.axhline(AR.z_4_28_calc_km[np.argmax(np.max(abs(ver_z[fi_50,:,:]), axis=1))], c="w", ls=":", label="Maximum perturbation")
    ax3.axhline(AR.z_4_28_calc_km[np.argmax(AR.f_VER_4_28(AR.z_4_28_calc_m))], c="w", ls="--", label=r"Maximum VER$_{4.28}$")
    # fig.colorbar(im, ax=ax3, label=r"$\Delta$VER$_{4.28}$ [$ph/m^3/s$]")
    fig.colorbar(im, ax=ax3, label=r"$\Delta$VER$_{4.28}$ [$W/m^3$]")
    ax3.set_xlabel("Time / [$s$]")
    leg = ax3.legend(frameon=False, loc=3)
    for text in leg.get_texts():
        text.set_color('w') # Set all legend text to green
    # ax3.set_ylabel("Altitude / [$km$]")
    ax3.set_xlim(500,1000)
    ax3.set_ylim(90,150)
    for ax in [ax1,ax2,ax3]:
        ax.set_ylim(90,150)
    fig.suptitle("DAYGLOW, calculated with Airglow framework")
    fig.savefig("./results_test/dayglow_2d_" + test + "_framework.pdf", dpi=600)
    fig.savefig("./results_test/dayglow_2d_" + test + "_framework.png", dpi=600)
    fig.tight_layout()
    ################################################################################################################

    
    ################################################################################################################
    ### HANDMADE CALCUATION (analytical sinusoid): 
    ################################################################################################################    
    ### 0. Pick frequency 
    # f0s = np.array([0.2,0.1,0.02, 0.01])
    f0s = freq
    ### 1. Make up an altitude range (Must be wider than VER region!)
    zrange = np.linspace(90e3,160e3,200)
    ### 2. Get frequencies / omega 
    ff = np.fft.fftfreq(Nt, d=dt)
    fr_fft = abs(ff)
    om = ff*(2*np.pi)
    ### 4. Get amplification, normalize by the value at 90 km: 
    a_func = AR.f_amplification(zrange)[:,None]/AR.f_amplification(hstart)
    ### 5. Calculate attenuation term 
    zatt = np.concatenate((np.linspace(0,zrange[0]-1e3,50),zrange))
    FFver, ZZver2 = np.meshgrid(fr_fft, zatt  )
    attenuation = AR.f_alpha_2d((ZZver2, FFver))
    att_exp = np.exp(-integrate.cumulative_trapezoid(attenuation, zatt, axis=0))   ### Supposes Np/m 
    att_exp = att_exp[-zrange.size:,:]
    ### 5b. Redefine amplification, including attenuation
    a_func = a_func * att_exp 
    
    ### Function to generate a sinusoid at a specific frequency and its dVER 
    def calculate_airglow_perturbation(f0):
        ### 6. Defined derivative of tapered sinusoid (functions above) and normalize by amplitude 
        if test=="kenda":
            tVEL = tapered_sinusoid_dt(time[None,:], zrange[:,None], f0, c=c)  
            tVEL_dz = np.gradient(tapered_sinusoid_dz(time[None,:], zrange[:,None], f0, c=c), time, axis=1) 
        tvmax = np.abs(tVEL).max()
        tVEL = tVEL/tvmax * ampl_at_start
        tVEL_dz = tVEL_dz/tvmax * ampl_at_start
        tDIS = tapered_sinusoid(time[None,:], zrange[:,None], f0, c=c)/tvmax* ampl_at_start
        print("Max. of velocity signal at z = 90 km: ", np.max(tVEL[0,:]))
        
        ### 7. Get their fft
        fftDIS = np.fft.fft(tDIS, axis=1)

        ### 8. Calculate the desired dver in frequency domain (EQ 4.22): 
        ### Calculate gradient of vel manually. NOTE: a_func contains both amplification and attenuation  
        fft_dver = AR.alpha_t * AR.f_VER_4_28(zrange)[:,None] * (AR.f_gamma(zrange)[:,None]-1)* AR.f_t(zrange)[:,None]*\
                                (np.gradient(fftDIS, zrange, axis=0)*a_func  + fftDIS*np.gradient(a_func, zrange, axis=0) ) 
        ### 9. OPTIONAL: Add advection term (don't forget amplification of u): 
        fft_dver += fftDIS*AR.f_dVER_4_28(zrange)[:,None]*a_func
        
        ### 10. Back to time domain 
        dver = np.fft.ifft(fft_dver, axis=1).real

        ### 11. Remove linear trend. 
        start = dver[:,0][:,None]
        end   = dver[:,50][:,None]
        trend = np.linspace(0, 1, dver.shape[1])   
        trend = start + (end - start)/(trend[50]-trend[0]) * trend  # shape (Nz, Nt)
        dver = dver - trend
        ### 12. Integrated intensity 
        I = np.trapz(dver, zrange, axis=0 )
        return(tDIS, dver, I)
    I_background = np.trapz(AR.f_VER_4_28(zrange), zrange, axis=0 )


    ################################################################################################################
    ### Show waveforms at different frequencies 
    ################################################################################################################
    fig, (ax2, ax3, ax4) = plt.subplots(3,1, sharex = True, figsize=(7,9)) 
    cmap = plt.get_cmap("plasma")
    cols = [cmap(i) for i in np.linspace(0.2, 0.8, f0s.size)]
    ### Original vel shape
    iz2 = np.argmin(abs(zrange-AR.z_4_28_calc_m[iz]))   ### To ensure we plot the same altitude as the other method 
    ### Propagated at 90 km 
    for fi, f0 in enumerate(f0s):
        tDIS, dver, I = calculate_airglow_perturbation(f0)

        ### Starting waveform at 90 km 
        print("max displacement: ", np.max(np.abs(tDIS[iz2,:])) )
        ax2.plot(time, np.gradient(tDIS[iz2,:],time), c=cols[fi], label="f={:.3g} Hz, T={:.3g} s".format(f0, 1/f0))
        if fi == f0s.size-1:
            ax2.set_ylim(-1.5*ampl_at_start,1.5*ampl_at_start)
        ### Calculated VER 
        vmin = -np.max(np.abs(dver[iz2,:]))*1.1
        vmax = np.max(np.abs(dver[iz2,:]))*1.1
        ax3.plot(time, dver[iz2,:]/AR.f_VER_4_28(zrange[iz2])*100, c=cols[fi])
        if fi == f0s.size-1:
            ax3b = ax3.twinx() 
            ax3b.plot(time, dver[iz2,:], ls=" ", color="w")    
            ax3.set_ylim(vmin/AR.f_VER_4_28(zrange[iz2])*100, vmax/AR.f_VER_4_28(zrange[iz2])*100)
            ax3b.set_ylim(vmin, vmax)
        ### Integrated Intensity
        imin = -np.max(np.abs(I))*1.1
        imax = np.max(np.abs(I))*1.1
        ax4.plot(time, I/I_background*100, c=cols[fi])
        if fi== f0s.size-1:
            ax4b = ax4.twinx() 
            ax4b.plot(time, I*factor_W_to_Rayleigh(4.28, dir="phRadiance_to_Rayleigh"), c="k", ls=" ")
            ax4.set_ylim(imin/I_background*100, imax/I_background*100)
            ax4b.set_ylim(imin*factor_W_to_Rayleigh(4.28, dir="phRadiance_to_Rayleigh"), imax*factor_W_to_Rayleigh(4.28, dir="phRadiance_to_Rayleigh"))
    ###
    ax2.set_ylabel(r"$V_z$ at " + "{:.1f} km".format(zrange[iz2]/1e3) + r" / [$m/s$]")
    ax3.set_ylabel("VER$_{4.28}$ pert. at " + "{:.1f} km".format(zrange[iz2]/1e3) + r" / [%]")
    ax3b.set_ylabel(r"VER$_{4.28}$ pert. at " + "{:.1f} km".format(zrange[iz2]/1e3) + r" / [$ph/m^3/s$]")
    ax4.set_ylabel("Relative Intensity pert. [%]")
    ax4b.set_ylabel("Intensity pert. [$Rayleigh$]")
    ax4.set_xlabel("Time / [$s$]")
    ax4.set_xlim(-10,tf)
    ax2.legend(frameon=False)
    for ax in  [ax2, ax3, ax4, ax3b, ax4b]:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2), useMathText=True)
    ###
    fig.align_labels() 
    fig.suptitle("DAYGLOW, calculated with Homemade sine framework")
    fig.savefig("./results_test/dayglow_wv_" + test + "_sine.pdf")
    fig.savefig("./results_test/dayglow_wv_" + test + "_sine.png", dpi=600)
    fig.tight_layout()
    ################################################################################################################
    

    ################################################################################################################
    ### Calculate for T=50s 
    ################################################################################################################
    tVEL, dver, I = calculate_airglow_perturbation(1/50)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,7))
    ###
    ax1.plot(AR.f_VER_4_28(zrange), zrange/1e3, c="k") 
    ax1b = ax1.twiny()
    ax1b.plot(np.max(abs(dver), axis=1)/AR.f_VER_4_28(zrange)*100, zrange/1e3, c="r", label="Max Perturbation")
    ax1b.set_xlabel(r"Max. VER$_{4.28}$ perturbation / [% background]")
    ax1b.xaxis.label.set_color('red')
    ax1.set_xlabel("VER / ph/m3/s")
    ax1.set_ylabel("Altitude / km")
    ax1.set_xlim(0,7e12)
    ###
    ax2.plot(AR.f_amplification(zrange)/AR.f_amplification(90e3), zrange/1e3, c="k", label=r"$\sqrt{\rho(90)c(90)/(\rho(z)c(z))}$") 
    ax2.set_xlabel("Amplification with respect to 90 km")
    ax2.set_xscale("log")
    ax2.legend(frameon=False, loc=4)
    ax2.set_xlim(1,1e4)
    ###
    print(np.max(np.abs(dver)),np.max(np.abs(dver))*fac)
    im = ax3.pcolormesh(time, zrange/1e3, dver[:,:]*fac, 
                        vmin=-np.max(np.abs(dver))*fac, 
                        vmax=np.max(np.abs(dver))*fac, rasterized=True)
    ax3.axhline(zrange[np.argmax(np.max(abs(dver), axis=1))]/1e3, c="w", ls=":", label="Maximum perturbation")
    ax3.axhline(zrange[np.argmax(AR.f_VER_4_28(zrange))]/1e3, c="w", ls="--", label=r"Maximum VER$_{4.28}$")
    # fig.colorbar(im, ax=ax3, label=r"$\Delta$VER$_{4.28}$ [$ph/m^3/s$]")
    fig.colorbar(im, ax=ax3, label=r"$\Delta$VER$_{4.28}$ [$W/m^3$]")
    ax3.set_xlabel("Time / [$s$]")
    leg = ax3.legend(frameon=False, loc=3)
    for text in leg.get_texts():
        text.set_color('w') # Set all legend text to green
    # ax3.set_ylabel("Altitude / [$km$]")
    ax3.set_xlim(500,1000)
    ax3.set_ylim(90,150)
    for ax in [ax1,ax2,ax3]:
        ax.set_ylim(90,150)
    fig.suptitle("DAYGLOW, calculated with Homemade sine framework")
    fig.savefig("./results_test/dayglow_2d_" + test + "_sine.pdf", dpi=600)
    fig.savefig("./results_test/dayglow_2d_" + test + "_sine.png", dpi=600)
    fig.tight_layout()
    # plt.show()
    ################################################################################################################
    

def check_Lognonne_2016():
    ### Calculation of Integrated VER (in Rayleigh) at different epicentral distances
    ### For a Mw 6.5 (M0 = 10e19 Nm) earthquake.
    ### Epicentral distances = [15, 30, 45, 60]

    ### First, define the grid of locations. 
    gridded       = False
    r_venus = 6051.8e3  ### m 
    north_shifts  = np.array([15., 30., 45., 60.])*2*np.pi*r_venus/360
    east_shifts   = np.array([0. for i in range(north_shifts.size)])
    print(north_shifts)

    ### Option for Pyrocko 
    opt_synthetics = dict(
        ### Options for source 
        mw = 6.5,              ### if none: We only get the Green's function
        depth = 30e3,          ### Only one depth
        strike = 45.,           ### Default mechanism 
        dip =  45., 
        rake = 45.,
        stf_type = None,       ### Dirac source 
        #stf_type = 'triangle', 
        # stf_type = 'sinus', 
        # effective_duration = 9.,
        ###  
        ### Options for store
        base_folder='/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/',
        ### Old option, single store 
        #store_id = 'GF_venus_Cold100_qssp',
        #store_id = 'GF_venus_Cold100_qssp_grid',
        ### Give store names, and min and max valid distance 
        #store_ids_dists = [('GF_venus_Cold100_qssp_grid',0e3,500e3),('GF_venus_Cold100_qssp_grid_mid',500e3,8000e3)],
        store_ids_dists = [('GF_venus_Cold100_atten_qssp_nearfield',0e3,50e3),('GF_venus_Cold100_atten_qssp',50e3,8000e3)],
        ###
        ### Options for grid 
        north_shifts = north_shifts, 
        east_shifts = east_shifts,
        gridded=gridded
    )

    ### INITIALIZE SEISMOGRAM CLASS 
    ### we build seismograms over grid
    ### 
    SEISMO = Seismograms(**opt_synthetics)
    print("Total grid size: ", SEISMO.NN.size)

    ### [1584.3e3, 3168.7e3, 4753.0e3 6337.4e3]
    ns, es       = 1584.3e3, 0e3
    ### Plot one of the waveforms for check
    fig = SEISMO.plot_traces(ns, es, do_interpolate=True)

    SEISMO.arrange_interpolate_synthetics(tmax=60*60, dt=0.1)

    ### Load airglow class 
    AIRGLOW = AirglowSignal(SEISMO, Nz=1000)

    dir_save="./results_test/"
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    dir_save += "singletrace_"
    list_inorth, list_ieast = AIRGLOW.iNN, [0 for i in range(AIRGLOW.Nn)]
    # print(list_inorth, list_ieast)

    ### Calculation of the Nightglow 
    AIRGLOW.calculate_1_27_airglow(list_ieast, list_inorth, loc_save_idx=[],
                                    do_parallel=True, 
                                    fourier_filtering=True,   ### Use time filtering 
                                    dir_save = dir_save,
                                    time_save = AIRGLOW.t_new) ### Save all timesteps 

    ### Calculation of the Dayglow
    AIRGLOW.calculate_4_28_airglow(list_ieast, list_inorth, loc_save_idx=[],
                            do_parallel=True, 
                            dir_save=dir_save, 
                            time_save = AIRGLOW.t_new) ### Save all timesteps 
    
    ####################################################################################################
    ### FIGURE 
    ####################################################################################################
    c_dayglow = "orangered"
    c_nightglow = "forestgreen"
    ### Load VER grids:
    I_nightglow = np.load(dir_save + "nightglow_I_t.npy")
    VER_Vz_nightglow = np.load(dir_save + "nightglow_dver_t.npy")
    I_dayglow = np.load(dir_save + "dayglow_I_t.npy")
    VER_Uz_dayglow = np.load(dir_save + "dayglow_dver_t.npy")

    time = AIRGLOW.t_new
    dt_airglow = SEISMO.dt
    ###
    z_ver_127 = AIRGLOW.z_1_27_calc_m  ### Always in meter
    z_ver_428 = AIRGLOW.z_4_28_calc_m  ### Always in meter
    ### Background VER 
    VER_127 = AIRGLOW.f_VER_1_27(z_ver_127) 
    VER_428 = AIRGLOW.f_VER_4_28(z_ver_428)
    ### Background Intensity 
    I_background_nightglow = AIRGLOW.I_background_nightglow
    I_background_dayglow   = AIRGLOW.I_background_dayglow


    ##############################################################
    ### Create figure
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(4, 2, figure=fig, width_ratios=[4, 1])

    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)] 
    axv1 = fig.add_subplot(gs[1, 1])
    axv2 = fig.add_subplot(gs[3, 1])
    # ax3b = axes[3].twinx()   ### Plot as purcentage of background intensity 
    # ax1b = axes[1].twinx()   ### Plot as purcentage of background intensity 
    
    colors = ["k", "r", "b", "g"]
    dR = 20 ### Rayleighs
    dR2 = 20e4
    fmin, fmax = 0.001, 0.04
    R_threshold = 1600
    ##############################################################

    ### Select a location: 
    # ns, es       = 3000e3, 0e3
    for i, (ns,es) in enumerate(zip(north_shifts, east_shifts)):
        idx = np.argmin(np.sqrt((AIRGLOW.NN-ns)**2+(AIRGLOW.EE-es)**2)) 
        i_east, i_north = 0, idx
        print(i_east, i_north)

        if i==0:
            axes[0].plot(time, AIRGLOW.VEL[i_east, i_north,:], c="k", lw=1)
            axes[2].plot(time, AIRGLOW.DIS[i_east, i_north,:], c="k", lw=1)

        ### 
        ### Filter I between fmin and fmax: 
        I_nightglow_filt = butter_filter(I_nightglow[i_east, i_north, :], 1/dt_airglow, fmin,fmax, order=4)
        axes[1].plot(time, I_nightglow_filt+dR*i, c=colors[i], lw=1, label="{:.0f}km, {:.0f}°".format(ns/1e3, ns*180/(np.pi*r_venus)))
        ###
        # axes[1].axhline(dR*i + R_threshold , c=colors[i], lw=1, ls=":")
        # axes[1].axhline(dR*i - R_threshold , c=colors[i], lw=1, ls=":")
    
        ###
        ### Filter I between fmin and fmax: 
        I_dayglow_filt = butter_filter(I_dayglow[i_east, i_north, :], 1/dt_airglow, fmin,fmax, order=4)
        axes[3].plot(time, I_dayglow_filt+dR2*i, c=colors[i], lw=1)
        ###
        # axes[3].axhline(dR2*i + I_background_dayglow , c=colors[i], lw=1, ls=":")
        # axes[3].axhline(dR2*i - I_background_dayglow , c=colors[i], lw=1, ls=":")
    ###
    axes[0].set_ylabel(r"Ground Vel. / [$m/s$]")
    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    axes[2].set_ylabel(r"Ground Disp. / [$m$]")
    axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ###
    axes[1].set_ylabel(r"1.27$\mu m$ Intensity / [$R$]")
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ###
    axes[3].set_ylabel(r"4.28$\mu m$ Intensity / [$R$]")
    axes[3].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ###
    axes[1].legend(framealpha=0.8, edgecolor="none", loc=1, title="Distance")

    ### Plot VER profiles 
    axv1.fill_betweenx(z_ver_127/1e3, 0, VER_127, edgecolor="k", facecolor=c_nightglow, alpha=0.4)
    axv1.set_ylabel(r"Altitude / [$km$]")
    axv1.set_xlabel(r"1.27$\mu m$ VER / [$ph/m^3/s$]")
    ###
    axv2.fill_betweenx(z_ver_428/1e3, 0, VER_428, edgecolor="k", facecolor=c_dayglow, alpha=0.4)
    axv2.set_ylabel(r"Altitude / [$km$]")
    axv2.set_xlabel(r"4.28$\mu m$ VER / [$ph/m^3/s$]")

    ###
    axes[-1].set_xlabel("Time / [$s$]")
    for ax in axes:
        ax.set_xlim(0,60*60)
    axv1.set_xlim(0, 6e11)
    axv2.set_xlim(0, 7e12)
    for ax in [axv1, axv2]:
        ax.set_ylim(80,160)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.xaxis.get_offset_text().set_position((1.2, 1.0))  # (x, y) in axis coordinates
        ax.xaxis.get_offset_text().set_horizontalalignment('left')
        ax.xaxis.get_offset_text().set_verticalalignment('bottom')
    for ax in axes[:-1]:
        ax.set_xticklabels([])

    fig.suptitle("Seismic and Airglow signals for Mw 6.5 earthquake, filtered between [{:.3g}, {:.3g}] Hz".format(fmin, fmax))
    fig.align_labels()
    fig.subplots_adjust(hspace=0.4, wspace=0.2, bottom=0.08, top=0.93)
    ###
    fig.savefig(dir_save + "Nightglow_Dayglow_traces_PL2016_dirac.png", dpi=300)

    
def minimal_example(test_waveform=False, test_sinusoid=False):

    def tapered_sinusoid(t, z, f0, c=200):
        ### Tapered sinusoid propagating at speed c 
        sig = np.sin(2*np.pi*f0*(t - z/c))
        Nsine = int((1/f0)/dt)
        tp = tukey(Nsine, alpha=0.0)

        tap = np.zeros(t.shape)
        Nt = t.size
        Nt0 = np.where(t<0)[0].size
        if np.isscalar(z):
            Nprop = int(z/c/dt)
            tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
            sig*=tap
        else:
            for i in range(z.size):
                Nprop = int(z[i,0]/c/dt)
                tap = np.pad(tp,(int(Nt0+Nprop),int(Nt-Nt0-tp.size-Nprop)))
                sig[i,:]*=tap
        return(sig)
    

    from scipy.fft import next_fast_len
    ### Calculation of Integrated VER (in Rayleigh) at different epicentral distances
    ### For a Mw 6.5 (M0 = 10e19 Nm) earthquake.
    ### Epicentral distances = [15, 30, 45, 60]

    ### First, define the grid of locations. 
    gridded       = False
    r_venus = 6051.8e3  ### m 
    north_shifts  = np.array([15., 30., 45., 60.])*2*np.pi*r_venus/360
    east_shifts   = np.array([0. for i in range(north_shifts.size)])

    # ### Option for Pyrocko 
    opt_synthetics = dict(
        ### Options for source 
        mw = 6.5,              ### if none: We only get the Green's function
        depth = 30e3,          ### Only one depth
        strike = 45.,           ### Default mechanism 
        dip =  45., 
        rake = 45.,
        stf_type = None,       ### Dirac source 
        #stf_type = 'triangle', 
        # stf_type = 'sinus', 
        # effective_duration = 9.,
        ###  
        ### Options for store
        base_folder='/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/',
        ### Old option, single store 
        #store_id = 'GF_venus_Cold100_qssp',
        #store_id = 'GF_venus_Cold100_qssp_grid',
        ### Give store names, and min and max valid distance 
        #store_ids_dists = [('GF_venus_Cold100_qssp_grid',0e3,500e3),('GF_venus_Cold100_qssp_grid_mid',500e3,8000e3)],
        store_ids_dists = [('GF_venus_Cold100_atten_qssp_nearfield',0e3,50e3),('GF_venus_Cold100_atten_qssp',50e3,8000e3)],
        ###
        ### Options for grid 
        north_shifts = north_shifts, 
        east_shifts = east_shifts,
        gridded=gridded
    )

    ### INITIALIZE SEISMOGRAM CLASS 
    ### we build seismograms over grid
    ### 
    SEISMO = Seismograms(**opt_synthetics)
    print("Total grid size: ", SEISMO.NN.size)

    ### [1584.3e3, 3168.7e3, 4753.0e3 6337.4e3]
    ns, es       = 1584.3e3, 0e3
    ### Plot one of the waveforms for check
    fig = SEISMO.plot_traces(ns, es, do_interpolate=True)

    SEISMO.arrange_interpolate_synthetics(tmax=60*60, dt=0.5)

    ### Load airglow class 
    AIRGLOW = AirglowSignal(SEISMO, Nz=1000)

    dir_save="./results_minimal_example/"
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    for i in range(4):
        np.savetxt(dir_save + "time_velocity_displacement_Mw_{:.1f}_{:.0f}deg.dat".format(6.5, north_shifts[i]*180/(np.pi*r_venus) ),
                    np.column_stack((AIRGLOW.t_new, AIRGLOW.VEL[0,i,:], AIRGLOW.DIS[0,i,:])) , header='Time [s] , Velocity [m/s], Displacement [m/s]')
    
    zver = np.linspace(0,160e3,1000)
    np.savetxt(dir_save + "altitude_nightglow_dayglow_ph_s_m3.dat", 
                    np.column_stack((zver, AIRGLOW.f_VER_1_27(zver), AIRGLOW.f_VER_4_28(zver))), header="Altitude [m], VER 1.27 [ph/s/m3], VER 4.28 [ph/s/m3]" )

    np.savetxt(dir_save + "atmosphere_z_T_rho_c_gamma.dat",
               np.column_stack((zver, AIRGLOW.f_t(zver), AIRGLOW.f_rho(zver), AIRGLOW.f_c(zver), AIRGLOW.f_gamma(zver))), 
               header = "Altitude [m], Temperature [K], Density [kg/m3], Sound speed [m/s], Heat capacity ratio gamma")

    dir_save="./results_minimal_example/"
    ### Load atmosphere: 
    dat_atm = np.loadtxt(dir_save + "atmosphere_z_T_rho_c_gamma.dat")
    alt_atm = dat_atm[:,0]    ### meters
    T_atm = dat_atm[:,1]      ### K
    rho_atm = dat_atm[:,2]    ### kg/m3
    c_atm = dat_atm[:,3]      ### m/s
    gamma_atm = dat_atm[:,4]  ### unit 

    ### Load VER (Converted to ph/s/m3)
    dat_ver = np.loadtxt(dir_save + "altitude_nightglow_dayglow_ph_s_m3.dat")
    alt_ver = dat_ver[:,0]    ### meters
    ver_1_27 = dat_ver[:,1]      ### ph/s/m3
    ver_4_28 = dat_ver[:,2]    ### ph/s/m3

    if test_waveform:
        ### Load waveform (Calculated for Mw 6.5 at different distances)
        dist = np.array([15., 30., 45., 60.])
        vel = [] 
        dis = []
        for i, d in enumerate(dist): 
            dat_vu = np.loadtxt(dir_save + "time_velocity_displacement_Mw_6.5_{:.0f}deg.dat".format(d))
            vel.append(dat_vu[:,1])
            dis.append(dat_vu[:,2])
            time = dat_vu[:,0]
        vel = np.array(vel)  ### Velocity shape [distance, time]
        dis = np.array(dis)  ### Displacement shape [distance, time]
        Nd = dist.size
        dt = time[1]-time[0]
    elif test_sinusoid:
        ### Calculate a sinusoid excitation 
        ### Scale it to 5e-3 at 90 km altitude, 30 deg distance 
        freq = [0.2, 0.1, 0.02, 0.01]
        vel = [] 
        dis = [] 
        dt = 0.5
        time = np.arange(-1000,1000,dt)
        for fi, f0 in enumerate(freq):
            vel.append(tapered_sinusoid(time, 0e3, f0))
            # dis.append(integrate.cumulative_trapezoid(tapered_sinusoid(time, 0e3, f0), time, initial=0))
            dis.append(tapered_sinusoid(time, 0e3, f0)/(2*np.pi*f0))
        vel = np.array(vel)  ### Velocity shape [frequency, time]
        dis = np.array(dis)  ### Displacement shape [frequency, time]
    
        ### To concord with BK, the amplitude of the wave must be 5 mm/s at 90 km. 
        ### We therefore rescale vel and dis using the amplification function 
        # ampl_at_start = 5e-3    ### m/s
        # i90 = np.argmin(abs(alt_atm-90e3))
        # ampl_90 = np.sqrt(rho_atm[0]*c_atm[0] / (rho_atm[i90]*c_atm[i90]))

        ### To concord with Sutin (2018), the amplitude of the wave must be 4 cm/s at 100 km. 
        ampl_at_start = 4e-2    ### Ampl of Mw 6.5, 10degree distance, 100 km alt (sutin 2018) 
        i90 = np.argmin(abs(alt_atm-100e3))
        ampl_90 = np.sqrt(rho_atm[0]*c_atm[0] / (rho_atm[i90]*c_atm[i90]))

        vel = vel * ampl_at_start/ampl_90
        dis = dis * ampl_at_start/ampl_90

    Nz = alt_atm.size 
    Nt = time.size

    ### Propagate velocity / Displacement upward 
    ### STEP1: Pad displacement + velocity with zeros to avoid problems with FFT 
    dpad = next_fast_len(Nt*2, real=True) - Nt
    long_vel = np.pad(vel, ((0,0),(0, dpad )), mode='constant')
    long_dis = np.pad(dis, ((0,0),(0, dpad )), mode='constant')
    Ntpad = long_vel.shape[1]

    ### STEP2. Take fft 
    fft_vel = np.fft.fft(long_vel, axis=1)
    fft_dis = np.fft.fft(long_dis, axis=1)
    freqsi = np.fft.fftfreq(n=Ntpad, d=dt)

    ### STEP3. Delay
    phase_shift_z = np.zeros((Nz, Ntpad), dtype = np.complex64)
    ### Integrate travel time from zero to z 
    travel_time = integrate.cumulative_trapezoid(1/c_atm, alt_atm, initial=0)
    for jz in range(Nz): 
        phase_shift_z[jz,:] = np.exp(-2 * np.pi * freqsi * 1j * travel_time[jz] )
    fft_vel_delayed = fft_vel[:,None,:]*phase_shift_z[None,:,:]
    fft_dis_delayed = fft_dis[:,None,:]*phase_shift_z[None,:,:]

    # fig, ax = plt.subplots() 
    # vel_delayed = np.fft.ifft(fft_vel_delayed, axis=2)
    # vel_delayed_norm = vel_delayed/np.max(np.abs(vel_delayed), axis=2)[:,:,None]
    # for i, z in enumerate(alt_atm):
    #     ax.plot(time, vel_delayed_norm[0,i,:]+ z/1e3, c="k", lw=1)
    

    ### STEP4. Amplify 
    amplification = np.sqrt( (rho_atm[0][None]*c_atm[0][None]) / (rho_atm*c_atm) )
    fft_vel_amplified = fft_vel_delayed * amplification[None,:,None]
    fft_dis_amplified = fft_dis_delayed * amplification[None,:,None]

    ### STEP5. Attenuate 
    ### We don't apply attenuation here 
    ### But it would be something like 
    ### attenuation = grid[Nz, Nt] in Np/m 
    ### att_exp = np.exp(-integrate.cumulative_trapezoid(attenuation, alt_atm, axis=0))   

    ### STEP6. For 1.27 airglow: apply filter 
    tau = 4460    ### seconds
    fft_vel_filtered = fft_vel_amplified*-(tau/(1 + 1j*2*np.pi*freqsi[None,None,:]*tau)) 

    ### STEP7. Calculate dver at all altitudes 
    ### First calculate d/dz(VER*v_z) --> amplitudes will be zero after integration 
    # fft_dver = np.gradient(fft_vel_filtered*ver_1_27[None,:,None], alt_atm, axis=1)
    ### First calculate VER*d/dz(v_z)
    fft_dver_1_27 = np.gradient(fft_vel_filtered, alt_atm, axis=1)*ver_1_27[None,:,None]
    ### Then go back to time domain 
    dver_1_27 = np.fft.ifft(fft_dver_1_27, axis=2).real
    ### Remove zero-padding 
    dver_1_27 = dver_1_27[:,:,:-Nt]

    ### For 4.28 airglow: define the temperature sensitivity 
    alpha_t = 0.01  ### 1% change if dVER/VER for 1 K 
    fft_dver_4_28 = alpha_t * ver_4_28[None,:,None]* (gamma_atm[None,:,None]-1)*T_atm[None,:,None] * np.gradient(fft_dis_amplified, alt_atm, axis=1)
     ### Then go back to time domain 
    dver_4_28 = np.fft.ifft(fft_dver_4_28, axis=2).real
    ### Remove zero-padding 
    dver_4_28 = dver_4_28[:,:,:-Nt]

    ### STEP8. Calculate light intensity 
    I_photons_1_27 = integrate.trapezoid(dver_1_27, alt_atm, axis=1)
    ### Convert to rayleigh 
    I_rayleigh_1_27 = I_photons_1_27*4*np.pi/1e10
    ### Remove integration trend 
    start = I_rayleigh_1_27[:,0][:,None]
    end   = I_rayleigh_1_27[:,50][:,None]
    trend = np.linspace(0, 1, Nt)   
    trend = start + (end - start)/(trend[50]-trend[0]) * trend  
    I_rayleigh_1_27 = I_rayleigh_1_27 - trend

    I_photons_4_28 = integrate.trapezoid(dver_4_28, alt_atm, axis=1)
    ### Convert to rayleigh 
    I_rayleigh_4_28 = I_photons_4_28*4*np.pi/1e10

    ### STEP9. Calculate background 
    I_background_1_27 = integrate.trapezoid(ver_1_27, alt_atm)*4*np.pi/1e10
    I_background_4_28 = integrate.trapezoid(ver_4_28, alt_atm)*4*np.pi/1e10

    
    
    ### STEP10. Plot 
    def make_figure_waveform():
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(4, 2, figure=fig, width_ratios=[4, 1])

        axes = [fig.add_subplot(gs[i, 0]) for i in range(4)] 
        axv1 = fig.add_subplot(gs[1, 1])
        axv2 = fig.add_subplot(gs[3, 1])
        ax3b = axes[3].twinx()   ### To plot as purcentage of background intensity 
        ax1b = axes[1].twinx()   ### To plot as purcentage of background intensity 
        
        colors = ["k", "r", "b", "g"]
        c_dayglow = "orangered"
        c_nightglow = "forestgreen"
        dR = 20 ### Rayleighs
        dR2 = 20e4
        dv = 1e-4
        du = 3e-4
        fmin, fmax = 0.001, 0.04
        ##############################################################

        ### Loop on distances from epicenter 
        for i, d in enumerate(dist):
            
            # if i==0:
            axes[0].plot(time, vel[i,:]+dv*i, c="k", lw=1)
            axes[2].plot(time, dis[i,:]+du*i, c="k", lw=1)

            ### Filter I between fmin and fmax: 
            # I_nightglow_filt = butter_filter(I_rayleigh_1_27[i, :], 1/dt, fmin,fmax, order=4)
            I_nightglow_filt = I_rayleigh_1_27[i, :]# butter_filter(I_rayleigh_1_27[i, :], 1/dt, fmin,fmax, order=4)
            axes[1].plot(time, I_nightglow_filt +dR*i, c=colors[i], lw=1, 
                         label="{:.0f}km, {:.0f}°".format(d/180*(np.pi*r_venus)/1e3, d))
            ax1b.plot(time, (I_nightglow_filt +dR*i)/I_background_1_27*100, ls="")
            ###
        
            ### Filter I between fmin and fmax: 
            # I_dayglow_filt = butter_filter(I_rayleigh_4_28[i, :], 1/dt, fmin,fmax, order=4)
            I_dayglow_filt = I_rayleigh_4_28[i, :]# butter_filter(I_rayleigh_4_28[i, :], 1/dt, fmin,fmax, order=4)
            axes[3].plot(time, I_dayglow_filt +dR2*i, c=colors[i], lw=1)
            ax3b.plot(time, (I_dayglow_filt +dR2*i)/I_background_4_28*100, ls="")
            ###
            
        ###
        axes[0].set_ylabel(r"Ground Vel. / [$m/s$]")
        axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        axes[2].set_ylabel(r"Ground Disp. / [$m$]")
        axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ###
        axes[1].set_ylabel(r"1.27$\mu m$ Intensity / [$R$]")
        axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax1b.set_ylabel(r"1.27$\mu m$ Intensity Pert. / [%]")
        ax1b.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
        ###
        axes[3].set_ylabel(r"4.28$\mu m$ Intensity / [$R$]")
        axes[3].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax3b.set_ylabel(r"4.28$\mu m$ Intensity Pert. / [%]")
        ax3b.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
        ###
        axes[1].legend(framealpha=0.8, edgecolor="none", loc=1, title="Distance")

        ### Plot VER profiles 
        axv1.fill_betweenx(alt_atm/1e3, 0, ver_1_27, edgecolor="k", facecolor=c_nightglow, alpha=0.4)
        axv1.set_ylabel(r"Altitude / [$km$]")
        axv1.set_xlabel(r"1.27$\mu m$ VER / [$ph/m^3/s$]")
        ###
        axv2.fill_betweenx(alt_atm/1e3, 0, ver_4_28, edgecolor="k", facecolor=c_dayglow, alpha=0.4)
        axv2.set_ylabel(r"Altitude / [$km$]")
        axv2.set_xlabel(r"4.28$\mu m$ VER / [$ph/m^3/s$]")

        ###
        axes[-1].set_xlabel("Time / [$s$]")
        for ax in axes:
            ax.set_xlim(0,60*60)
        axv1.set_xlim(0, 6e11)
        axv2.set_xlim(0, 7e12)
        for ax in [axv1, axv2]:
            ax.set_ylim(80,160)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.xaxis.get_offset_text().set_position((1.2, 1.0))  # (x, y) in axis coordinates
            ax.xaxis.get_offset_text().set_horizontalalignment('left')
            ax.xaxis.get_offset_text().set_verticalalignment('bottom')
        for ax in axes[:-1]:
            ax.set_xticklabels([])

        fig.suptitle("Seismic and Airglow signals for Mw 6.5 earthquake, filtered between [{:.3g}, {:.3g}] Hz".format(fmin, fmax))
        fig.align_labels()
        fig.subplots_adjust(hspace=0.4, wspace=0.35, bottom=0.08, top=0.93)
        ###
        # fig.savefig(dir_save + "Nightglow_Dayglow_traces_PL2016_dirac.png", dpi=300)

    
    def make_figure_sinusoid():
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(4, 2, figure=fig, width_ratios=[4, 1])

        axes = [fig.add_subplot(gs[i, 0]) for i in range(4)] 
        axv1 = fig.add_subplot(gs[1, 1])
        axv2 = fig.add_subplot(gs[3, 1])
        ax3b = axes[3].twinx()   ### To plot as purcentage of background intensity 
        ax1b = axes[1].twinx()   ### To plot as purcentage of background intensity 
        
        colors = ["k", "r", "b", "g"]
        c_dayglow = "orangered"
        c_nightglow = "forestgreen"
        dR = 0#20 ### Rayleighs
        dR2 = 0#20e4
        dv = 0#1e-4
        du = 0#3e-4
        fmin, fmax = 0.001, 0.04
        ##############################################################

        ### Loop on distances from epicenter 
        for i, f in enumerate(freq):
            
            # if i==0:
            axes[0].plot(time, vel[i,:]+dv*i, c=colors[i], lw=1)
            axes[2].plot(time, dis[i,:]+du*i, c=colors[i], lw=1)

            ### Filter I between fmin and fmax: 
            # I_nightglow_filt = butter_filter(I_rayleigh_1_27[i, :], 1/dt, fmin,fmax, order=4)
            I_nightglow_filt = I_rayleigh_1_27[i, :]# butter_filter(I_rayleigh_1_27[i, :], 1/dt, fmin,fmax, order=4)
            axes[1].plot(time, I_nightglow_filt +dR*i, c=colors[i], lw=1, 
                         label="f={:.3g} Hz, T={:.3g} s".format(f, 1/f))
            ax1b.plot(time, (I_nightglow_filt +dR*i)/I_background_1_27*100, ls="")
            ###
        
            ### Filter I between fmin and fmax: 
            # I_dayglow_filt = butter_filter(I_rayleigh_4_28[i, :], 1/dt, fmin,fmax, order=4)
            I_dayglow_filt = I_rayleigh_4_28[i, :]# butter_filter(I_rayleigh_4_28[i, :], 1/dt, fmin,fmax, order=4)
            axes[3].plot(time, I_dayglow_filt +dR2*i, c=colors[i], lw=1)
            ax3b.plot(time, (I_dayglow_filt +dR2*i)/I_background_4_28*100, ls="")
            ###
            
        ###
        axes[0].set_ylabel(r"Ground Vel. / [$m/s$]")
        axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        axes[2].set_ylabel(r"Ground Disp. / [$m$]")
        axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ###
        axes[1].set_ylabel(r"1.27$\mu m$ Intensity / [$R$]")
        axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax1b.set_ylabel(r"1.27$\mu m$ Intensity Pert. / [%]")
        ax1b.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
        ###
        axes[3].set_ylabel(r"4.28$\mu m$ Intensity / [$R$]")
        axes[3].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax3b.set_ylabel(r"4.28$\mu m$ Intensity Pert. / [%]")
        ax3b.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
        ###
        axes[1].legend(framealpha=0.8, edgecolor="none", loc=1, title="Distance")

        ### Plot VER profiles 
        axv1.fill_betweenx(alt_atm/1e3, 0, ver_1_27, edgecolor="k", facecolor=c_nightglow, alpha=0.4)
        axv1.set_ylabel(r"Altitude / [$km$]")
        axv1.set_xlabel(r"1.27$\mu m$ VER / [$ph/m^3/s$]")
        ###
        axv2.fill_betweenx(alt_atm/1e3, 0, ver_4_28, edgecolor="k", facecolor=c_dayglow, alpha=0.4)
        axv2.set_ylabel(r"Altitude / [$km$]")
        axv2.set_xlabel(r"4.28$\mu m$ VER / [$ph/m^3/s$]")

        ###
        axes[-1].set_xlabel("Time / [$s$]")
        for ax in axes:
            ax.set_xlim(-100, time.max())
        axv1.set_xlim(0, 6e11)
        axv2.set_xlim(0, 7e12)
        for ax in [axv1, axv2]:
            ax.set_ylim(80,160)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.xaxis.get_offset_text().set_position((1.2, 1.0))  # (x, y) in axis coordinates
            ax.xaxis.get_offset_text().set_horizontalalignment('left')
            ax.xaxis.get_offset_text().set_verticalalignment('bottom')
        for ax in axes[:-1]:
            ax.set_xticklabels([])

        fig.suptitle("Seismic and Airglow signals for sinusoid perturbation")
        fig.align_labels()
        fig.subplots_adjust(hspace=0.4, wspace=0.35, bottom=0.08, top=0.93)
        ###
        # fig.savefig(dir_save + "Nightglow_Dayglow_traces_PL2016_dirac.png", dpi=300)

    
    if test_waveform:
        make_figure_waveform()
    elif test_sinusoid:
        make_figure_sinusoid()



# =========================================================================================================
if __name__ == '__main__':
# =========================================================================================================

    ### TEST SCALER FUNCTIONS 
    # compute_airglow_scaler_new()
    # compute_airglow_scaler_sine()
    compute_airglow_scaler_new(mw = 1, strike=45, dip=45, rake=45, do_plot=True, effect="ampl",
                              store_ids_dists = [('GF_venus_Cold100_atten_qssp_nearfield',0e3,50e3),('GF_venus_Cold100_atten_qssp',50e3,8000e3)])  
                                ### Check effect of amplification curves 
    # compute_airglow_scaler_Hots(mw=None, strike=45, dip=45, rake=45, do_plot=True, effect=None, tit ="", 
                            # store_ids_dists = [('GF_venus_Hot10_atten_qssp_nearfield',0e3,50e3),('GF_venus_Hot10_atten_qssp',50e3,8000e3)])
                                ## Plot all the Hots together

    ### TEST SINUSOID PERTURBATIONS 
    # check_simple_perturbation_nightglow(test="sutin")
    # check_simple_perturbation_nightglow(test="kenda")
    # check_simple_perturbation_dayglow()

    ### OTHER TESTS 
    # check_Lognonne_2016()
    # minimal_example(test_waveform=True)
    # minimal_example(test_sinusoid=True)

    plt.show()


    