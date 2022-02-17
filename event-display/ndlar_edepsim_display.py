#!/usr/bin/env python3

import h5py
import numpy as np
import yaml
import sys
import argparse

import plotly.graph_objects as go
from plotly import subplots

from particle import Particle

from larndsim import consts
from collections import defaultdict
from larndsim.consts import detector, physics

_default_path_to_geometry = "larndsim/detector_properties/ndlar-module.yaml"
_default_path_to_pixels = "larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"

class DetectorGeometry():
    def __init__(self, detector_properties, pixel_layout):
        self.detector_properties = detector_properties
        self.pixel_layout = pixel_layout
        self.geometry = {}
        self.io_group_io_channel_to_tile = {}
        self.tile_positions = None
        self.tile_orientations = None
        self.tpc_offsets = None
        self.load_geometry()
        consts.load_properties(detector_properties,pixel_layout)

    @staticmethod
    def rotate_pixel(pixel_pos, tile_orientation):
        return pixel_pos[0]*tile_orientation[2], pixel_pos[1]*tile_orientation[1]

    def load_geometry(self):
        geometry_yaml = yaml.load(open(self.pixel_layout), Loader=yaml.FullLoader)

        pixel_pitch = geometry_yaml['pixel_pitch']
        chip_channel_to_position = geometry_yaml['chip_channel_to_position']
        self.tile_orientations = geometry_yaml['tile_orientations']
        self.tile_positions = geometry_yaml['tile_positions']
        tile_indeces = geometry_yaml['tile_indeces']
        tile_chip_to_io = geometry_yaml['tile_chip_to_io']
        xs = np.array(list(chip_channel_to_position.values()))[:, 0] * pixel_pitch
        ys = np.array(list(chip_channel_to_position.values()))[:, 1] * pixel_pitch
        x_size = max(xs)-min(xs)+pixel_pitch
        y_size = max(ys)-min(ys)+pixel_pitch

        tile_geometry = {}

        with open(self.detector_properties) as df:
            detprop = yaml.load(df, Loader=yaml.FullLoader)

        self.tpc_offsets = detprop['tpc_offsets']
        for tile in tile_chip_to_io:
            tile_orientation = self.tile_orientations[tile]
            tile_geometry[tile] = self.tile_positions[tile], self.tile_orientations[tile]

            for chip in tile_chip_to_io[tile]:
                io_group_io_channel = tile_chip_to_io[tile][chip]
                io_group = io_group_io_channel//1000
                io_channel = io_group_io_channel % 1000
                self.io_group_io_channel_to_tile[(io_group, io_channel)] = tile

            for chip_channel in chip_channel_to_position:
                chip = chip_channel // 1000
                channel = chip_channel % 1000

                try:
                    io_group_io_channel = tile_chip_to_io[tile][chip]
                except KeyError:
                    continue

                io_group = io_group_io_channel // 1000
                io_channel = io_group_io_channel % 1000
                x = chip_channel_to_position[chip_channel][0] * \
                    pixel_pitch - x_size / 2 + pixel_pitch / 2
                y = chip_channel_to_position[chip_channel][1] * \
                    pixel_pitch - y_size / 2 + pixel_pitch / 2

                x, y = self.rotate_pixel((x, y), tile_orientation)
                x += self.tile_positions[tile][2]
                y += self.tile_positions[tile][1]
                self.geometry[(io_group, io_channel, chip, channel)] = x, y

    def get_z_coordinate(self, io_group, io_channel, time):
        tile_id = self.get_tile_id(io_group, io_channel)

        z_anode = self.tile_positions[tile_id][0]
        drift_direction = self.tile_orientations[tile_id][0]
        return z_anode + time * detector.V_DRIFT * drift_direction

    def get_tile_id(self, io_group, io_channel):
        if (io_group, io_channel) in self.io_group_io_channel_to_tile:
            tile_id = self.io_group_io_channel_to_tile[io_group, io_channel]
        else:
            return np.nan

        return tile_id

drawn = []
fig = go.Figure(drawn)
color_dict = {11: '#3f90da',
              -11: '#92dadd',
              13: '#b2df8a',
              -13: '#33a02c',
              22: '#b15928',
              2212: '#bd1f01',
              -2212: '#e76300',
              -211: '#cab2d6',
              211: '#6a3d9a',
              2112: '#555555',
              1000010020: 'blue'}
color_dict = defaultdict(lambda: 'gray', color_dict)

def plot_geometry():
    _drawn_objects = []

    for ix in range(0,detector.TPC_BORDERS.shape[0],2):
        for i in range(2):
            for j in range(2):
                _drawn_objects.append(go.Scatter3d(x=(detector.TPC_BORDERS[ix][0][j],detector.TPC_BORDERS[ix][0][j]),
                                                  y=(detector.TPC_BORDERS[ix][2][0],detector.TPC_BORDERS[ix+1][2][0]),
                                                  z=(detector.TPC_BORDERS[ix][1][i],detector.TPC_BORDERS[ix][1][i]),
                                                  showlegend=False,
                                                  hoverinfo='skip',
                                                  opacity=0.5,
                                                  mode='lines',line={'color':'gray'}))

                _drawn_objects.append(go.Scatter3d(x=(detector.TPC_BORDERS[ix][0][j],detector.TPC_BORDERS[ix][0][j]),
                                                  y=(detector.TPC_BORDERS[ix+i][2][0],detector.TPC_BORDERS[ix+i][2][0]),
                                                  z=(detector.TPC_BORDERS[ix][1][0],detector.TPC_BORDERS[ix][1][1]),
                                                  showlegend=False,
                                                  hoverinfo='skip',
                                                  opacity=0.5,
                                                  mode='lines',line={'color':'gray'}))
                #print(detector.TPC_BORDERS[ix][1][j],detector.TPC_BORDERS[ix][1][j])
                _drawn_objects.append(go.Scatter3d(x=(detector.TPC_BORDERS[ix][0][0],detector.TPC_BORDERS[ix][0][1]),
                                                  y=(detector.TPC_BORDERS[ix+i][2][0],detector.TPC_BORDERS[ix+i][2][0]),
                                                  z=(detector.TPC_BORDERS[ix][1][j],detector.TPC_BORDERS[ix][1][j]),
                                                  showlegend=False,
                                                  hoverinfo='skip',
                                                  opacity=0.5,
                                                  mode='lines',line={'color':'gray'}))

        #xx = np.linspace(detector.TPC_BORDERS[ix][0][0], detector.TPC_BORDERS[ix][0][1], 2)
        #zz = np.linspace(detector.TPC_BORDERS[ix][1][0], detector.TPC_BORDERS[ix][1][1], 2)
        #xx,zz = np.meshgrid(xx,zz)

        #single_color=[[0.0, 'rgb(200,200,200)'], [1.0, 'rgb(200,200,200)']]
        #z_cathode = (detector.TPC_BORDERS[ix][2][0]+detector.TPC_BORDERS[ix+1][2][0])/2

        #cathode_plane=dict(type='surface', x=xx, y=np.full(xx.shape, z_cathode), z=zz,
        #                   opacity=0.15,
        #                   hoverinfo='skip',
        #                   text='Cathode',
        #                   colorscale=single_color,
        #                   showlegend=False,
        #                   showscale=False)

        #_drawn_objects.append(cathode_plane)

        annotations_x = [(p[0][0]+p[0][1])/2 for p in detector.TPC_BORDERS]
        annotations_y = [p[1][1]for p in detector.TPC_BORDERS]
        annotations_z = [(p[2][0]+p[2][1])/2 for p in detector.TPC_BORDERS]

        annotations_label = ["(%i,%i)" % (ip//2+1,ip%2+1) for ip in range(detector.TPC_BORDERS.shape[0])]
        module_annotations = go.Scatter3d(
            mode='text',
            x=annotations_x,
            z=annotations_y,
            y=annotations_z,
            text=annotations_label,
            opacity=0.5,
            textfont=dict(
                color='gray',
                size=8
            ),
            showlegend=False,
        )
        _drawn_objects.append(module_annotations)
    return _drawn_objects

camera = dict(
    eye=dict(x=-1.7,y=0.3,z=1.1)
)

fig.update_layout(scene_camera=camera,
                  uirevision=True,
                  margin=dict(l=0, r=0, t=5),
                  legend={"y" : 0.8},
                  scene=dict(
                      xaxis=dict(backgroundcolor="white",
                                 showspikes=False,
                                 showgrid=True,
                                 title='z [cm]'),##note this is x every where else in the code
                                #  range=(detector.TPC_BORDERS[0][0][0],detector.TPC_BORDERS[0][0][1])),
                      yaxis=dict(backgroundcolor="white",
                                 showgrid=True,
                                 showspikes=False,
                                 title='x [cm]'), ##note this is y every where else in the code
                                #  range=(detector.TPC_BORDERS[0][2][0],detector.TPC_BORDERS[1][2][0])),
                      zaxis=dict(backgroundcolor="white",
                                 showgrid=True,
                                 showspikes=False,
                                 title='y [cm]'),##note this is z every where else in the code
                                #  range=(detector.TPC_BORDERS[0][1][0],detector.TPC_BORDERS[0][1][1])),
                  ))

def plot_hits(segments, min_time, max_time):
    z_offset = 0
    x_offset = 0
    y_offset = 0
    drawn_objects = []
    for itrk in range(len(segments)):
        segment = segments[itrk]
        if segment['t'] < min_time or segment['t'] > max_time: continue
       # latex_name = Particle.from_pdgid(segment['pdgId']).latex_name
        #html_name = Particle.from_pdgid(segment['pdgId']).html_name.replace('&pi;','&#960;').replace('&gamma;','&#947;')
        line = go.Scatter3d(x=np.linspace(segment['z_start'] + x_offset, segment['z_end'] + x_offset, 5),
                            y=np.linspace(segment['x_start'] + y_offset, segment['x_end'] + y_offset, 5),
                            z=np.linspace(segment['y_start'] + z_offset, segment['y_end'] + z_offset, 5),
                            mode='lines',
                            hoverinfo='text',
                            text='<br>pdg: %i<br>dE: %f<br>dEdx: %f<br>time: %f' % (segment['pdgId'], segment['dE'], segment['dEdx'], segment['t'] ),
                            opacity=0.7,
                            # legendgrouptitle_text=r'$%s$' % latex_name,
                            customdata=['segment_%i' % itrk],
                            line=dict(
                                color=color_dict[segment['pdgId']],
                                width=4
                            ))
        drawn_objects.append(line)
    return drawn_objects
    

def plot_tracks(tracks, min_time, max_time):
    z_offset = 0
    x_offset = 0
    y_offset = 0
    pdgs = []
    drawn_objects = []
    for itrk in range(len(tracks)):
        track = tracks[itrk]
        if track['t_start'] < min_time or track['t_start'] > max_time: continue

        latex_name = Particle.from_pdgid(track['pdgId']).latex_name
        html_name = Particle.from_pdgid(track['pdgId']).html_name.replace('&pi;','&#960;').replace('&gamma;','&#947;')
        line = go.Scatter3d(x=np.linspace(track['xyz_start'][2]/10 + x_offset, track['xyz_end'][2]/10 + x_offset, 5),
                            y=np.linspace(track['xyz_start'][0]/10 + y_offset, track['xyz_end'][0]/10 + y_offset, 5),
                            z=np.linspace(track['xyz_start'][1]/10 + z_offset, track['xyz_end'][1]/10 + z_offset, 5),
                            mode='lines',
                            hoverinfo='text',
                            name=r'$%s$' % latex_name,
                            text='<br>trackId: %i<br>eventID: %i<br>start_time: %f<br>p:%f' % (itrk, track['eventID'], track['t_start'],np.linalg.norm(track['pxyz_start'])),
                            opacity=0.3,
                            legendgroup='%i_%i' % (0,track['pdgId']),
                            # legendgrouptitle_text=r'$%s$' % latex_name,
                            customdata=['track_%i' % itrk],
                            showlegend=track['pdgId'] not in pdgs,
                            line=dict(
                                color=color_dict[track['pdgId']],
                                width=2
                            ))
        
        if track['pdgId'] not in pdgs:
            pdgs.append(track['pdgId'])
        drawn_objects.append(line)
        drawn_objects.append(line)

    return drawn_objects

def main(filename, min_time, max_time, path_to_geometry, path_to_pixels, event_ids, draw_mc=False):
    my_geometry = DetectorGeometry(path_to_geometry, path_to_pixels)
    datalog = h5py.File(filename, 'r')
    tracks = datalog['trajectories']#[:250]
    segments = datalog['segments']
    good_event_ids = set()
    #get event ids corresponding to this time
    if event_ids is None:
        for itrk in range(len(tracks)):
            track = tracks[itrk]
            if track['t_start'] < min_time: continue
            if track['t_end'] > max_time: continue
            good_event_ids.add(track['eventID'])
    else:
        good_event_ids = set([int(ev_id) for ev_id in event_ids.split(',')])

    print('Displaying events:', good_event_ids)

    event_id_mask_tracks = [0]*len(tracks)
    event_id_mask_segs = [0]*len(segments)

    for ev_id in good_event_ids:
        new_track_mask = tracks['eventID']==ev_id
        new_segs_mask = segments['eventID']==ev_id

        event_id_mask_segs = np.logical_or(event_id_mask_segs, new_segs_mask)
        event_id_mask_tracks = np.logical_or(event_id_mask_tracks, new_track_mask)

    good_tracks = tracks[event_id_mask_tracks]
    good_segments = segments[event_id_mask_segs]

    track_ids = np.unique(tracks['trackID'])[1:]
    segment_ids = np.unique(segments['trackID'])[1:]
    #drawn_objects.extend(plot_tracks(track_ids, 0))
    if not draw_mc:
        drawn_objects = plot_hits(good_segments, min_time, max_time)
    if draw_mc:
        drawn_objects = plot_tracks(good_tracks, min_time, max_time)
    drawn_objects.extend(plot_geometry())
    fig.add_traces(drawn_objects)

    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-i', type=str, help='''edep-sim file converted to hdf5''')
    parser.add_argument('--path_to_geometry', default=_default_path_to_geometry, type=str, help='''Filename with module geometry info''')
    parser.add_argument('--path_to_pixels', default=_default_path_to_pixels, type=str, help='''Filename with pixel layout''')
    parser.add_argument('--min_time', '-n', type=float, default=0, help='''Min time to start plotting''')
    parser.add_argument('--max_time', '-x', type=float, default=999999999, help='''Max time to end plotting''')
    parser.add_argument('--draw_mc', '-mc', action='store_true', help='''Flag to draw mc truth''')
    parser.add_argument('--event_ids', type=str, default=None, help='''EventIds to draw, comma separated list''')
    args = parser.parse_args()
    c = main(**vars(args))

