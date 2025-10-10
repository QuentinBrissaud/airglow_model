###
import pandas as pd
import numpy as np
import sys
###
from scipy import interpolate, integrate
from scipy import signal
from scipy.signal import fftconvolve, lfilter, cont2discrete, butter, sosfilt 
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
        vmin = np.mean(wf)-1*np.std(wf)
        vmax = np.mean(wf)+1*np.std(wf)
        
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
gridded = None
dir_save = None 

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
def _factor_W_to_Rayleigh(L, bandwidth = 0.03, dir="specRadiance_to_Rayleigh"):
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
def init_worker_nightlow(_list_of_locations, _fft_vzs, _att_exp, _amplification, _phase_shift_z,
                _f_VER_1_27,_f_dVER_1_27, _z_1_27_calc_m, _fourier_filtering,
                _b, _a, _loc_save, _itime_save, _gridded, _tf_phase_nightglow, _dir_save):
    global fft_vzs, att_exp, amplification, phase_shift_z
    global f_VER_1_27, f_dVER_1_27, z_1_27_calc_m, fourier_filtering, tf_phase_nightglow
    global b, a
    global list_of_locations, loc_save, itime_save, gridded, dir_save

    list_of_locations = _list_of_locations
    fft_vzs = _fft_vzs
    att_exp = _att_exp
    amplification = _amplification
    phase_shift_z = _phase_shift_z
    f_VER_1_27 = _f_VER_1_27
    f_dVER_1_27 = _f_dVER_1_27
    z_1_27_calc_m = _z_1_27_calc_m
    fourier_filtering = _fourier_filtering
    b = _b
    a = _a
    loc_save = _loc_save
    itime_save = _itime_save
    gridded = _gridded
    tf_phase_nightglow = _tf_phase_nightglow
    dir_save = _dir_save


def init_worker_dayglow(_list_of_locations, _fft_uzs, _att_exp, _amplification, _phase_shift_z,
                _z_4_28_calc_m, _fac_temperature, _f_VER_4_28, _loc_save, _itime_save, _gridded, _dir_save):
    global fft_uzs, att_exp, amplification, phase_shift_z
    global z_4_28_calc_m, fac_temperature, f_VER_4_28
    global list_of_locations, loc_save, itime_save, gridded, dir_save

    list_of_locations = _list_of_locations
    fft_uzs = _fft_uzs
    att_exp = _att_exp
    amplification = _amplification
    phase_shift_z = _phase_shift_z
    z_4_28_calc_m = _z_4_28_calc_m
    f_VER_4_28 = _f_VER_4_28
    fac_temperature = _fac_temperature
    loc_save = _loc_save
    itime_save = _itime_save
    gridded = _gridded
    dir_save = _dir_save 


# =========================================================================================================
### Function to propagate a group seismograms up in the atmosphere (simple model)
def propagate_attenuate(fft, i_east, i_north, att, ampl, psz):
    ### Apply attenuation and amplification at all z 
    ### shape attenuation, phase_shift: (Nz, Nw). Shape velocity, fft: (Ne, Nn, Nz, Nw)

    att_vz = fft[i_east, i_north, np.newaxis, :] * att
    ampl_vz_z = att_vz * ampl[:, np.newaxis]

    ### Delay at altitude z 
    fft_vz_z = ampl_vz_z *psz

    ### Back to time domain: inverse FFT to get vz_z for these altitudes
    vz_z = np.real(sfft.ifft(fft_vz_z, axis=1))  # shape: (2, Nt)

    ### Remove mean (to help filtering?) 
    #vz_z -= np.mean(vz_z, axis=1)[:,np.newaxis]

    ### Filter high frequency and low frequency 
    #vz_z = butter_filter(vz_z, 1/0.5, 0.001, 0.1, axis=1)
    #f = sfft.fftfreq(vz_z.shape[1], d=0.5)
    #fft_vz_z *= butterworth_bandpass_response(f, 0.001, 0.1, 4)[np.newaxis, :]

    return(vz_z, fft_vz_z)


# =========================================================================================================
### Function to transform velocity at a specific altitude into VER perturbation 
def velocity_to_dVER_nightglow(vz_z, fft_vz_z, z_1_27_calc_m, f_VER_1_27, f_dVER_1_27, b, a, tf_phase_nightglow=None, fourier_filtering=False, test=False):

    fver_alt = f_VER_1_27(z_1_27_calc_m)[:,np.newaxis]
    fdver_alt = f_dVER_1_27(z_1_27_calc_m)[:,np.newaxis]
    #print(fdver_alt)

    ### Filter at all altitudes 
    if not fourier_filtering:
        ### Compute VER and its vertical gradient (numpy gradient) TIME DOMAIN 
        ver_vz = fver_alt * vz_z  # shape: (Nz, Nt)
        ### VERSION WITH SMOOTH GRADIENT 
        dver_vz_z = fver_alt * np.gradient(vz_z, z_1_27_calc_m, axis=0) + fdver_alt*vz_z
        ### VERSION OF PL 
        #dver_vz_z = np.gradient(ver_vz, z_1_27_calc_m, axis=0)
        ### VERSION OF BK 
        #dver_vz_z = fver_alt * np.gradient(vz_z, z_1_27_calc_m, axis=0)

        dver_z = lfilter(b, a, dver_vz_z, axis=1)
    else:
        ### Compute VER and its vertical gradient (numpy gradient) FOURIER DOMAIN 
        ver_vz = fver_alt * fft_vz_z  # shape: (Nz, Nt)        
        ### VERSION OF PL 
        dver_vz_z = np.gradient(ver_vz, z_1_27_calc_m, axis=0)
        ### VERSION OF BK 
        #dver_vz_z = fver_alt * np.gradient(vz_z, z_1_27_calc_m, axis=0)

        dver_z = sfft.ifft(tf_phase_nightglow * dver_vz_z, axis=1).real

        ### Simple linear detrend 
        ### dver_z = signal.detrend(dver_z, type="linear", axis=1) ### way too slow 
        start = dver_z[:,0][:,None]
        end   = dver_z[:,50][:,None]
        trend = np.linspace(0, 1, dver_z.shape[1])   
        trend = start + (end - start)/(trend[50]-trend[0]) * trend  # shape (Nz, Nt)
        dver_z = dver_z - trend
    
    if test:
        ### TEST 
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

    ### Ensure dVER starts at zero 
    dver_z -= dver_z[:,0][:,np.newaxis]

    return(dver_z)


# =========================================================================================================
### Function to transform velocity at a specific altitude into VER perturbation 
def temperature_perturbation(uz_z, z_4_28_calc_m, fac_temperature):

    ### Divergence of U * temperature factor 
    dver_z = fac_temperature[:,None] * np.gradient(uz_z, z_4_28_calc_m, axis=0) 
    ### Tet with P.Y. F. expression (here uz_z is actually v)
    # dver_z = fac_temperature[:,None] * uz_z 

    return(dver_z)


# =========================================================================================================
### Function to integrate over line of sight (simple model)
def integrate_line_of_sight(dver_z, z_calc_m, wavelength):
    ### For now, the LOS is a simple vertical line 
    #amp_dayglow = np.trapz((dVER_ad+1*dVER_tr), x=alts_dayglow, axis=1)/np.trapz(f_VER_dayglow(alts_dayglow), x=alts_dayglow,)

    ### Luminosity of airglow perturbation 
    ### dver_z shape (Nz,Nt)
    amp_airglow = np.trapz(dver_z, x=z_calc_m, axis=0) # /np.trapz(f_VER(alts_dayglow), x=alts_dayglow,)

    ### Convert to Rayleigh
    #amp_airglow *= _factor_W_to_Rayleigh(wavelength, dir="Radiance_to_Rayleigh")  ### for VER in W/m3
    amp_airglow *= _factor_W_to_Rayleigh(wavelength, dir="phRadiance_to_Rayleigh")  ### for VER in ph/s/m3

    ### Can then be converted into total airglow luminosity by adding the integral of ver(z) 

    return(amp_airglow)


# =========================================================================================================
### Wrapper function for calculating NIGHTglow at one location (outside of class to be parallelisable)
# =========================================================================================================
def nightglow_at_location(i_en, list_of_locations, fft_vzs, att_exp, amplification, phase_shift_z,f_VER_1_27, f_dVER_1_27,
                        z_1_27_calc_m, fourier_filtering, b,a, loc_save, itime_save, gridded, tf_phase_nightglow, dir_save):
    
    i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]

    # if (i_east, i_north) == loc_save[0]:
    #     print(i_en, i_east, i_north)
    
    vz_z, fft_vz_z = propagate_attenuate(fft_vzs, i_east, i_north, att_exp, amplification, phase_shift_z)
    ###
    dver_z = velocity_to_dVER_nightglow(vz_z, fft_vz_z, z_1_27_calc_m, f_VER_1_27, f_dVER_1_27, b, a, 
                                        tf_phase_nightglow=tf_phase_nightglow, fourier_filtering= fourier_filtering)
    
    amp_nightglow = integrate_line_of_sight(dver_z, z_1_27_calc_m, 1.27)
            
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

    return(vz_z[:,itime_save], dver_z[:,itime_save], amp_nightglow[itime_save])


# =========================================================================================================
### Wrapper function for calculating DAYglow at one location (outside of class to be parallelisable)
# =========================================================================================================
def dayglow_at_location(i_en, list_of_locations, fft_uzs, att_exp, amplification, phase_shift_z, 
                        z_4_28_calc_m, fac_temperature, f_VER_4_28, loc_save, itime_save, gridded, dir_save):
    
    i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]

    # if (i_east, i_north) == loc_save[0]:
    #     print(i_en, i_east, i_north)
    
    uz_z, fft_uz_z = propagate_attenuate(fft_uzs, i_east, i_north, att_exp, amplification, phase_shift_z)
    ###
    dver_z = temperature_perturbation(uz_z, z_4_28_calc_m, fac_temperature)
    ### With advection of VER term (BK):
    ad_ver = uz_z * np.gradient(f_VER_4_28(z_4_28_calc_m), z_4_28_calc_m)[:,None]
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
    return(uz_z[:,itime_save], dver_z[:,itime_save], amp_dayglow[itime_save])
    


def worker_func_nightglow(i):
    # location = list_of_locations[i]
    return nightglow_at_location(i, list_of_locations, fft_vzs, att_exp, amplification, 
                               phase_shift_z, f_VER_1_27, f_dVER_1_27, z_1_27_calc_m, 
                               fourier_filtering, b, a, loc_save, itime_save, gridded, tf_phase_nightglow, dir_save)


def worker_func_dayglow(i):
    # location = list_of_locations[i]
    return dayglow_at_location(i, list_of_locations, fft_uzs, att_exp, amplification, 
                               phase_shift_z, z_4_28_calc_m, fac_temperature, f_VER_4_28, 
                               loc_save, itime_save, gridded, dir_save)


# =========================================================================================================
class AirglowSignal:
# =========================================================================================================

    def __init__(self, SEISMO, Nz = 40, do_plot=False):
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
        self.__dict__.update(SEISMO.__dict__)

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
            self.f_alpha, self.f_alpha_2d, self.f_amplification = self._load_absorption_amplification(file_attenuation_kenda=file_attenuation_kenda, do_plot=do_plot)
        else:
            self.f_alpha, self.f_alpha_2d, self.f_amplification = self._load_absorption_amplification(dir_attenuation_GA=dir_attenuation_GA, do_plot=do_plot)

        ### Definitions for the calculation of 1_27 micrometer nightglow 
        self.tau = 4460 # s
        self.b, self.a = self._def_filter_nightglow()
        self.Nz = Nz    # Number of altitude points for gradients and integrations. 
        self.z_1_27_calc_m = np.linspace(self.z_1_27_min, self.z_1_27_max, self.Nz)  # in meters, always 
        self.z_1_27_calc_km = self.z_1_27_calc_m / 1e3        
        self.dz_1_27_m = np.diff(self.z_1_27_calc_m)[0]
        ### For calculation of cumulated attenuation: 
        self.z_att_1_27_m = np.concatenate((np.arange(self.z_1_27_min,0, -self.dz_1_27_m)[1:][::-1], self.z_1_27_calc_m)) 
        self.I_background_nightglow = integrate_line_of_sight(self.f_VER_1_27(self.z_1_27_calc_m), self.z_1_27_calc_m, 1.27)

        ### Definitions for the calculation of 4_28 micrometer nightglow         
        self.alpha_t = 0.01    ### VERY IMPORTANT: 1% sensitivity to temperature variations 
        self.z_4_28_calc_m = np.linspace(self.z_4_28_min, self.z_4_28_max, self.Nz)  # in meters, always 
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


    def _factor_W_to_Rayleigh(self, L, bandwidth = 0.03, dir="specRadiance_to_Rayleigh"):
        ### bandwidth should be in micrometer if Radiance is in W/m2/micrometer/sr
        ### Note: In B Kenda's thesis, its 4pi.(Ep)^-1 and not (4pi.Ep)⁻1 that is correct.
        ### Correct definition of the Rayleigh: 1R = (1/4pi) * 1e10 photons/s/cm2/steradians 
        Ep = self.hp*self.c/(L*1e-6)    #### Energy of photon at wavelength L 

        if dir=="specRadiance_to_Rayleigh":
            ### If radiance is initially in W/m2/micrometer/sr 
            factor = bandwidth/Ep*4*np.pi*1e-10 
            return(factor)
        elif dir=="Radiance_to_Rayleigh":
            ### If radiance is initially in W/m2/sr 
            factor = 1/Ep*4*np.pi*1e-10 
            return(factor)
        elif dir=="phRadiance_to_Rayleigh":
            ### If radiance is initially in ph/s/m2/sr 
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

    
    def _load_absorption_amplification(self, file_attenuation_kenda=None, dir_attenuation_GA=None, do_plot=False):

        if file_attenuation_kenda is not None:
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
        num = np.polymul(num, num2)
        den = np.polymul(den, den2)
        ### Do a second multiplication to get second order 
        num = np.polymul(num, num2)
        den = np.polymul(den, den2)

        system_d = cont2discrete((num, den), self.dt, method='bilinear')
        b, a = system_d[0].flatten(), system_d[1].flatten()
        return(b,a)


    def calculate_1_27_airglow(self, list_ieast, list_inorth, loc_save_idx=None, loc_save_EN = None, 
                               time_save = None, fourier_filtering=False, 
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
        save_wavefield = np.zeros((self.Ne, self.Nn, self.Nz, len(time_save),2 ))
        save_intensity_dver = np.zeros((self.Ne, self.Nn,len(time_save)))

        ### Define frequencies 
        freqsi = sfft.fftfreq(d=self.dt, n=self.Nt)
        freqsp = abs(freqsi)

        ### Fourier transform of seismograms 
        fft_vzs = sfft.fft(self.VEL, axis=2)

        ### Pre-calculate the phase shift corresponding to the delay with altitude 
        self.phase_shift_z = np.zeros((self.Nz, self.Nt), dtype = np.complex64)
        ### Integrated travel time from zero to airglow altitudes 
        self.travel_time = self.dz_1_27_m * np.cumsum(1/self.f_c(self.z_att_1_27_m))
        self.travel_time = self.travel_time[-self.Nz:]
        for jz, zz in enumerate(self.z_1_27_calc_m):
            ### Constant propagation velocity og 300 m/s
            #self.phase_shift_z[jz,:] = np.exp(-2 * np.pi * freqsi * 1j * zz / 0.3)

            ### Integrated propagation velocity from zero to altitude z  
            self.phase_shift_z[jz,:] = np.exp(-2 * np.pi * freqsi * 1j * self.travel_time[jz] )

        ### If using fourier filtering, prepare the filter: 
        if fourier_filtering:
            self.tf_phase_nightglow = -(self.tau/(1+1j*2*np.pi*freqsi[None,:]*self.tau)) 
            ### Set no gain at low frequencies 
            ### PB: Sets mean to zero but trend is still there
            ### After detrending + resetting VER(0)=0, this effectively doesn't do anything... 
            # self.tf_phase_nightglow[:,0] = 0.0 + 0.0j
            
            ### Better option: Applying a high-pass filter to VER(t): H_hp(i omega) = iomega / (i omega + omega_c)
            self.tf_phase_nightglow *= (1j*freqsi[None,:] / (1j*freqsi[None,:] + 1e-4))**2
        else:
            self.tf_phase_nightglow = None


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
        att_exp = np.exp(-self.dz_1_27_m*np.cumsum(attenuation, axis=0))   ### Supposes Np/m 
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
                # i_en = 3360
                # save_wavefield, save_intensity_dver = self.nightglow_at_location(i_en, 
                #                                                                 fft_vzs, fourier_filtering, 
                #                                                                 save_wavefield, save_intensity_dver, 
                #                                                                 loc_save, itime_save)
                vz_z_it, dver_z_it, amp_nightglow_it = nightglow_at_location(i_en, list_of_locations, fft_vzs, self.att_exp, self.amplification, 
                                                                          self.phase_shift_z,self.f_VER_1_27,self.f_dVER_1_27,
                                                                        self.z_1_27_calc_m, fourier_filtering, self.b, self.a, 
                                                                        loc_save_idx, itime_save, self.gridded, self.tf_phase_nightglow, dir_save)
                if self.gridded:
                    i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]
                    save_wavefield[i_east, i_north, :,:,0] = vz_z_it    ### Save velocity waveform 
                    save_wavefield[i_east, i_north, :,:,1] = dver_z_it  ### Save dVER at altitude z 
                    save_intensity_dver[i_east, i_north,:] = amp_nightglow_it  ### Save Intensity at altitude z

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
                                            initargs=(list_of_locations, fft_vzs, self.att_exp, self.amplification, self.phase_shift_z,
                                                    self.f_VER_1_27, self.f_dVER_1_27, self.z_1_27_calc_m, fourier_filtering,
                                                    self.b, self.a, 
                                                    loc_save_idx, itime_save, self.gridded, self.tf_phase_nightglow, dir_save)
                                                ) as p:
                
                results = list(tqdm(p.imap(worker_func_nightglow, list_indices), total=len(list_indices), bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}' ))
                t2 = ptime.time()
            print("Time for airglow calculation: {:.1f} s".format(t2-t0))

            ### Store wavefields 
            # if self.gridded: 
            print("Re-aranging gridded wavefield...")
            for i_en, r in enumerate(tqdm(results, total=len(results), bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}' ) ):
                i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]
                    
                vz_z_it = r[0]
                dver_z_it = r[1]
                amp_nightglow_it = r[2]
                save_wavefield[i_east, i_north, :,:,0] = vz_z_it         ### Save velocity waveform at all requested altitudes and times
                save_wavefield[i_east, i_north, :,:,1] = dver_z_it       ### Save dVER at all requested altitudes and times
                save_intensity_dver[i_east, i_north,:] = amp_nightglow_it  ### Save Intensity at requested times  

            
        ### Save the full wavefield, but only at certain times 
        print("Saving gridded wavefield...")
        np.save(dir_save + "nightglow_dver_t", save_wavefield)
        np.save(dir_save + "nightglow_I_t", save_intensity_dver)
        print("Grid save completed.")


    def calculate_4_28_airglow(self, list_ieast, list_inorth, loc_save_idx=None, loc_save_EN = None,
                                 time_save = None, 
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
        save_wavefield = np.zeros((self.Ne, self.Nn, self.Nz, len(time_save),2 ))
        save_intensity_dver = np.zeros((self.Ne, self.Nn,len(time_save)))

        ### Define frequencies 
        freqsi = sfft.fftfreq(d=self.dt, n=self.Nt)
        freqsp = abs(freqsi)

        ### Fourier transform of seismograms 
        fft_uzs = sfft.fft(self.DIS, axis=2)
        # fft_vzs = sfft.fft(self.VEL, axis=2)

        ### Pre-calculate the phase shift corresponding to the delay with altitude 
        self.phase_shift_z = np.zeros((self.Nz, self.Nt), dtype = np.complex64)
        ### Integrated travel time from zero to airglow altitudes 
        self.travel_time = self.dz_4_28_m * np.cumsum(1/self.f_c(self.z_att_4_28_m))
        self.travel_time = self.travel_time[-self.Nz:]
        for jz, zz in enumerate(self.z_4_28_calc_m):
            ### Constant propagation velocity og 300 m/s
            #self.phase_shift_z[jz,:] = np.exp(-2 * np.pi * freqsi * 1j * zz / 0.3)

            ### Integrated propagation velocity from zero to altitude z  
            self.phase_shift_z[jz,:] = np.exp(-2 * np.pi * freqsi * 1j * self.travel_time[jz] )

        ### Grid of amplification 
        self.amplification = self.f_amplification(self.z_4_28_calc_m)

        ### Grid of attenuation: OPTION WITHOUT CUMULATIVE SUM 
        #FFver, ZZver = np.meshgrid(freqsp, self.z_1_27_calc_km)
        #attenuation = self.f_alpha_2d((ZZver, FFver))
        ### Exponential of attenuation (supposing Np/km)
        #self.att_exp = np.exp(-self.z_1_27_calc_km[:,np.newaxis]*attenuation)   ### meter or kilometer ? Cumulative sum or not ? 

        ### NOTE: Cumulative sum works only if we are starting from z=0
        FFver, ZZver2 = np.meshgrid(freqsp, self.z_att_4_28_m)
        attenuation = self.f_alpha_2d((ZZver2, FFver))
        att_exp = np.exp(-self.dz_4_28_m*np.cumsum(attenuation, axis=0))   ### Supposes Np/m 
        self.att_exp = att_exp[-self.Nz:]
        # fig, ax = plt.subplots() 
        # for i in range(self.Nz):
        #     ax.plot(freqsp, self.att_exp[i,:])    

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
                uz_z_it, dver_z_it, amp_dayglow_it = dayglow_at_location(i_en, list_of_locations, fft_uzs, self.att_exp, self.amplification, 
                                                                          self.phase_shift_z,self.z_4_28_calc_m, self.fac_temperature, self.f_VER_4_28, 
                                                                          loc_save_idx, itime_save, self.gridded, dir_save)
                
                i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]
                save_wavefield[i_east, i_north, :,:,0] = uz_z_it    ### Save dispalcement waveform 
                save_wavefield[i_east, i_north, :,:,1] = dver_z_it  ### Save dVER at altitude z 
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
                                            initargs=(list_of_locations, fft_uzs, self.att_exp, self.amplification, self.phase_shift_z,
                                                    self.z_4_28_calc_m, self.fac_temperature, self.f_VER_4_28, loc_save_idx, itime_save, self.gridded, dir_save)
                                                ) as p:
                
                results = list(tqdm(p.imap(worker_func_dayglow, list_indices), total=len(list_indices), bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}' ))
                t2 = ptime.time()
            print("Time for dayglow calculation: {:.1f} s".format(t2-t0))

            ### Store wavefields 
            # if self.gridded: 
            print("Re-aranging gridded dayglow wavefield...")
            for i_en, r in enumerate(tqdm(results, total=len(results), bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}' ) ):
                i_east, i_north = list_of_locations[i_en][0], list_of_locations[i_en][1]
                    
                uz_z_it = r[0]
                dver_z_it = r[1]
                amp_airglow_it = r[2]
                save_wavefield[i_east, i_north, :,:,0] = uz_z_it         ### Save displacement waveform at all requested altitudes and times
                save_wavefield[i_east, i_north, :,:,1] = dver_z_it       ### Save dVER at all requested altitudes and times
                save_intensity_dver[i_east, i_north,:] = amp_airglow_it  ### Save Intensity at requested times  

            
        ### Save the full wavefield, but only at certain times 
        print("Saving gridded dayglow wavefield...")
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
        # fig.savefig("./Figures/Seismic_to_Nightglow.png")
        

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
        # fig.savefig("./Figures/Seismic_to_Dayglow.png")
        

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
### New AIRGLOW SCALER 
# =========================================================================================================
def compute_airglow_scaler_new(mw=None, strike=45, dip=45, rake=45, do_plot=True):
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
        store_ids_dists = [('GF_venus_Cold100_qssp_grid',0e3,500e3),('GF_venus_Cold100_qssp_grid_mid',500e3,8000e3)],
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
    AIRGLOW = AirglowSignal(SEISMO)

    ### Now we compute the AIRGLOW at all locations and timesteps. 
    ### NOTE : This can be quite heavy ! 
    ### List of all north and east indices:
    dir_save="./results_detectability/"
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    list_inorth, list_ieast = AIRGLOW.iNN, AIRGLOW.iEE
    AIRGLOW.calculate_1_27_airglow(list_ieast, list_inorth, loc_save_idx=[],
                                   do_parallel=True, 
                                   fourier_filtering=False,   ### Use time filtering 
                                   dir_save = dir_save,
                                   time_save = AIRGLOW.t_new) ### Save all timesteps 
    ### Calculation of the Dayglow
    AIRGLOW.calculate_4_28_airglow(list_ieast, list_inorth, loc_save_idx=[],
                                   do_parallel=True, 
                                   dir_save=dir_save, 
                                   time_save = AIRGLOW.t_new) ### Save all timesteps 

    I_nightglow = np.load(dir_save + "nightglow_I_t.npy")
    I_dayglow = np.load(dir_save + "dayglow_I_t.npy")
    

    ### Now, we make some frequency bins 
    # freq_bins = np.logspace(np.log10(1e-3), np.log10(5e-1), 5)
    fmean = [10**-3, 10**-2, 10**-1, 10**0]
    freq_bins = [None, 10**-2.5, 10**-1.5, 10**-0.5, None]  ### Centered around 1e-2, 1e-1, 1. 
    f_targets = []
    for ibin, (binleft, binright) in enumerate(zip(freq_bins[:-1], freq_bins[1:])):
        f_targets += [[binleft, binright]]
    print(" Filter bins: ", f_targets)

    scaling_airglow = pd.DataFrame()
    ### We loop over locations and store the max amplitude in a dataframe: 
    for f1, f2 in tqdm(f_targets, disable=True):

        waveform_nightglow_filt = butter_filter(I_nightglow, 1/dt_airglow, f1,f2, order=4, axis=2)
        perturb_nightglow_filt = waveform_nightglow_filt/AIRGLOW.I_background_nightglow*100

        waveform_dayglow_filt = butter_filter(I_dayglow, 1/dt_airglow, f1,f2, order=4, axis=2)
        perturb_dayglow_filt = waveform_dayglow_filt/AIRGLOW.I_background_dayglow*100
        
        # fig, ax = plt.subplots() 
        # ax.plot(I_t[5,5,:])
        # ax.plot(waveform_nightglow[5,5,:])

        for (ies, ins) in zip(AIRGLOW.iEE.ravel(), AIRGLOW.iNN.ravel()):
            es, ns = AIRGLOW.EE[ies, ins], AIRGLOW.NN[ies,ins]
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

    if do_plot:
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
        fig.savefig(dir_save + "Airglow_scaler.png", dpi=300)

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
        ### Note mf: Going there
        f_vz, f_dvz = vz_and_dzv
        shape_init = times_rescaled.shape
        #dzv = f_dvz(times_rescaled.ravel(), ALTS_DAYGLOW.ravel(), grid=False).reshape(shape_init)
        pts = np.stack([times_rescaled.ravel(), ALTS_DAYGLOW.ravel()], axis=-1)
        dzv = f_dvz(pts).reshape(shape_init)
        
    #vzdzver = vz*dVERnightglowdz(ALTS_DAYGLOW)
    
    tf_phase_nightglow = -(tau/(1+1j*omega[:,None]*tau)) 
    signal = f_VER(ALTS_DAYGLOW)*dzv
    ### Code test mf 
    # fig, ax = plt.subplots()
    # plt.plot(ALTS_DAYGLOW[0,:], dzv[1000,:]*1e3, label="dvz/dz")
    # plt.plot(ALTS_DAYGLOW[0,:], f_vz(pts).reshape(shape_init)[1000,:], label="vz")
    # plt.legend()
    # plt.xlim(90,120)
    # plt.show()
    # brrout 
    
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
        b   = -tau * (1.0 - a)              # zero
        
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