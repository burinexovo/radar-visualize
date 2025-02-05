# %%
import os
import struct
import numpy as np
from pyart.config import get_metadata
from pyart.core.radar import Radar
# import cwb_radar
from pyart.graph import RadarMapDisplayBasemap
import matplotlib.pyplot as plt

# def make_empty_cwb_ppi_radar(cwb_radar_object):
# %%


def read_cwb_radar_sweep(fname):
    """
    Return an Radar object, representing a PPI scan.

    Parameters
    ----------
    ngates : int
        Number of gates per ray.
    rays_per_sweep : int
        Number of rays in each PPI sweep.
    nsweeps : int
        Number of sweeps.

    Returns
    -------
    radar : Radar
        Radar object with no fields, other parameters are set to default
        values.
    transforming CWB data into object	

    """
    cwb_radar_object = radar(fname)

    ngates = cwb_radar_object.ngate
    rays_per_sweep = cwb_radar_object.nray
    nsweeps = 1

    nrays = rays_per_sweep * nsweeps

    time = get_metadata('time')
    _range = get_metadata('range')
    latitude = get_metadata('latitude')
    longitude = get_metadata('longitude')
    altitude = get_metadata('altitude')
    sweep_number = get_metadata('sweep_number')
    sweep_mode = get_metadata('sweep_mode')
    fixed_angle = get_metadata('fixed_angle')
    sweep_start_ray_index = get_metadata('sweep_start_ray_index')
    sweep_end_ray_index = get_metadata('sweep_end_ray_index')
    azimuth = get_metadata('azimuth')
    elevation = get_metadata('elevation')

    # fields = {}
    # if ("cref" in fname):
    if ("ref_qc" in fname):
        varname = 'corrected_reflectivity'
    elif ("ref_raw" in fname):
        varname = 'reflectivity'
    elif ("vel" in fname):
        varname = 'velocity'
    elif ("phi" in fname):
        varname = 'differential_phase'
    elif ("zdr" in fname):
        varname = 'differential_reflectivity'
    elif ("rho" in fname):
        varname = 'cross_correlation_ratio'
    elif ('cref' in fname):
        varname = 'composite_reflectivity'

    data_dict = get_metadata(varname)
    data_dict['data'] = np.array(cwb_radar_object.data, dtype='float32')
    fields = {varname: data_dict}
    scan_type = 'ppi'
    metadata = {'instrument_name': cwb_radar_object.name}

    time['data'] = np.arange(nrays, dtype='float64')
#    time['units'] = 'seconds since 1989-01-01T00:00:01Z'
    time['units'] = 'seconds since '+str(cwb_radar_object.yyyy) +\
        '-'+format(cwb_radar_object.mm, '02d') +\
        '-'+format(cwb_radar_object.dd, '02d') +\
        'T'+format(cwb_radar_object.hh, '02d') +\
        ':'+format(cwb_radar_object.mn, '02d') +\
        ':'+format(cwb_radar_object.ss, '02d')+'Z'
    _range['data'] = np.linspace(cwb_radar_object.gate_start, cwb_radar_object.gate_start +
                                 cwb_radar_object.gate_sp*(ngates-1), ngates).astype('float32')

    latitude['data'] = np.array([cwb_radar_object.rlat], dtype='float64')
    longitude['data'] = np.array([cwb_radar_object.rlon], dtype='float64')
    altitude['data'] = np.array([cwb_radar_object.radar_elev], dtype='float64')

    sweep_number['data'] = np.arange(nsweeps, dtype='int32')
    sweep_mode['data'] = np.array(['azimuth_surveillance'] * nsweeps)
    fixed_angle['data'] = np.array(
        [cwb_radar_object.theta] * nsweeps, dtype='float32')
    sweep_start_ray_index['data'] = np.arange(0, nrays, rays_per_sweep,
                                              dtype='int32')
    sweep_end_ray_index['data'] = np.arange(rays_per_sweep - 1, nrays,
                                            rays_per_sweep, dtype='int32')

  #  azimuth['data'] = np.arange(nrays, dtype='float32')
    azimuth['data'] = np.linspace(cwb_radar_object.azm_start, cwb_radar_object.azm_start +
                                  cwb_radar_object.azm_sp*(nrays-1), nrays, dtype='float32')
#    azimuth['data'] = np.linspace(cwb_radar_object.azm_start,cwb_radar_object.azm_start+0.5*(nrays-1),nrays, dtype='float32')
    elevation['data'] = np.array(
        [cwb_radar_object.theta] * nrays, dtype='float32')

    return Radar(time, _range, fields, metadata, scan_type,
                 latitude, longitude, altitude,
                 sweep_number, sweep_mode, fixed_angle, sweep_start_ray_index,
                 sweep_end_ray_index,
                 azimuth, elevation,
                 instrument_parameters=None)


class radar:
    def __init__(self, fname):

        if '.gz' in fname:
            import gzip
            print('unzip files')
            f = gzip.open(fname).read()
        else:
            f = open(fname, "rb").read()
        header_end = 160  # after  160 units is data, before it is header info
        # get header in format char(len=16)+36integers
        header = struct.unpack('<16s36i', f[0:header_end])
        print('header:', header)
        info = np.array(header[1:-1])
        if (len(str(header[0], 'utf-8')) == 12):
            self.name = str(header[0], 'utf-8')[0:16:4]
        else:
            self.name = str(header[0][0:4], 'utf-8')
        # self.name='test'
        self.h_scale = info[0]
        info_flt = info/info[0]
        self.radar_elev = info_flt[1]
        self.rlat = info_flt[2]
        print('rlat:', self.rlat)
        self.rlon = info_flt[3]
        print('rlon:', self.rlon)
        self.yyyy = int(info_flt[4])
        self.mm = int(info_flt[5])
        self.dd = int(info_flt[6])
        self.hh = int(info_flt[7])
        self.mn = int(info_flt[8])
        self.ss = int(info_flt[9])
        self.nyquist = info_flt[10]
        self.nivcp = int(info_flt[11])
        self.itit = info[12]
        self.theta = info_flt[13]       # elevation angle
        self.nray = int(info_flt[14])
        self.ngate = int(info_flt[15])
        self.azm_start = info_flt[16]
        # self.azm_sp = info_flt[17]
        self.azm_sp = 360./self.nray
        self.gate_start = info_flt[18]   # gate start is m
        self.gate_sp = info_flt[19]  # unit=m
        self.var_scale = info_flt[20]
        self.var_miss = info[21]
        self.rf_miss = -44400
        n = self.nray*self.ngate
        data = struct.unpack('<'+str(n)+'i', f[header_end:])
        self.data = np.reshape(data/self.var_scale,
                               (self.nray, self.ngate), order='C')
        print('data_shape:', self.data.shape)

# %%
