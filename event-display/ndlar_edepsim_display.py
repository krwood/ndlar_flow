#!/usr/bin/env python3

import h5py
import numpy as np
import yaml
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import plotly.graph_objects as go
from plotly import subplots

import plotly.io as pio
#pio.kaleido.scope.mathjax = None


from particle import Particle

from larndsim import consts
from collections import defaultdict
from larndsim.consts import detector, physics

_default_path_to_geometry = "../../larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
_default_path_to_pixels = "../../larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"

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

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer):
        self.draw(renderer)
        return 0


drawn = []
fig = go.Figure(drawn)
#def get_color
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
    x=[]
    y=[]
    z=[]
    for ix in range(0,detector.TPC_BORDERS.shape[0],2):
        for i in range(2):
            for j in range(2):
                x.append(detector.TPC_BORDERS[ix][0][j])
                x.append(detector.TPC_BORDERS[ix][0][j])
                y.append(detector.TPC_BORDERS[ix][2][0])
                y.append(detector.TPC_BORDERS[ix+1][2][0])
                z.append(detector.TPC_BORDERS[ix][1][i])
                z.append(detector.TPC_BORDERS[ix][1][i])

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

    #print(min(x), max(x))
    
    #print(min(y), max(y))
    #print(min(z), max(z))
    return _drawn_objects

def get_geometry_frame():
    lines = []
    for ix in range(0,detector.TPC_BORDERS.shape[0],2):
        for i in range(2):
            for j in range(2):
            
                lines.append([(detector.TPC_BORDERS[ix][0][j],detector.TPC_BORDERS[ix][0][j]),(detector.TPC_BORDERS[ix][2][0],detector.TPC_BORDERS[ix+1][2][0]), (detector.TPC_BORDERS[ix][1][i],detector.TPC_BORDERS[ix][1][i])])
                lines.append([(detector.TPC_BORDERS[ix][0][j],detector.TPC_BORDERS[ix][0][j]), (detector.TPC_BORDERS[ix+i][2][0],detector.TPC_BORDERS[ix+i][2][0]), (detector.TPC_BORDERS[ix][1][0],detector.TPC_BORDERS[ix][1][1])])
                lines.append([(detector.TPC_BORDERS[ix][0][0],detector.TPC_BORDERS[ix][0][1]), (detector.TPC_BORDERS[ix+i][2][0],detector.TPC_BORDERS[ix+i][2][0]), (detector.TPC_BORDERS[ix][1][j],detector.TPC_BORDERS[ix][1][j]) ])

        #cathode_plane=dict(type='surface', x=xx, y=np.full(xx.shape, z_cathode), z=zz,
        #                   opacity=0.15,
        #                   hoverinfo='skip',
        #                   text='Cathode',
        #                   colorscale=single_color,
        #                   showlegend=False,
        #                   showscale=False)


       

    #print(min(x), max(x))
    #print(min(y), max(y))
    #print(min(z), max(z))
    return lines

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


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def set_axes_equal(ax, bounds):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = bounds[0]
    y_limits = bounds[1]
    z_limits = bounds[2]

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_hits_plt(segments, min_time, max_time, figfile):
    xs, ys, zs, cs, ss = [], [], [], [], []
    fid_evtids = []
    color_list = ['#ff0000','#00ff00','#0000ff','#ff00ff','#00ffff','#59d354','#5954d8','#aaa5bf','#d3ce87','#ddba87','#bc9e82','#c6997c','#bf8277','#ce5e60','#aa8e93','#a5777a','#936870','#d35954','#9200ff','#84c1a3','#89a8a0','#829e8c','#adbcc6']

    new_color_dict = {186186 : '#ff0000',
                        7432 : '#00ff00',
                        163063 : '#0000ff',
                        20377 : '#ff00ff', 
                        168301 : '#00ffff',
                        181493 : '#59d354', 
                        23327 : '#5954d8', 
                        240206 : '#aaa5bf', 
                        95820 : '#d3ce87', 
                        150201 : '#ddba87'}
    
    new_color_dict = defaultdict(lambda: 'gray', new_color_dict)
    npts = 10
    z_bounds = [-356.7, 356.7]
    y_bounds = [-148.613, 155.387]
    x_bounds = [413.72, 916.68]
    bounds = [x_bounds, y_bounds, z_bounds]

    for itrk in range(len(segments)):
        #if not itrk%10 == 0: continue
      #  print(itrk, '/', len(segments))
        segment = segments[itrk]
        if segment['t'] < min_time or segment['t'] > max_time:
            continue

        disp = np.array([segment['z_start']-segment['z_end'], segment['x_start']-segment['x_end'], segment['y_start']-segment['y_end']])
        npts = int(np.linalg.norm(disp)*20)

        xs += list(np.linspace(segment['z_start'], segment['z_end'], npts))
        ys += list(np.linspace(segment['x_start'], segment['x_end'], npts))
        zs += list(np.linspace(segment['y_start'], segment['y_end'], npts))
        ss += list([0.3]*npts)

        if segment['eventID'] in fid_evtids:
            this_color = color_list[fid_evtids.index(segment['eventID'])]
        elif segment['fiducial']:
            this_color = color_list[len(fid_evtids)]
            fid_evtids += [segment['eventID']]
        else:
            #this_color = 'gray'
            this_color = 'black'
        cs += list([this_color]*npts)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs, c=cs, s=ss, marker=',', linewidths=0)

    lines = get_geometry_frame()
    for line in lines:
        if (abs(line[2][0]-line[2][1])>0.5):
             print("y:",line[2][0],line[2][1])
        fxs = np.linspace(line[0][0], line[0][1], 100)
        fys = np.linspace(line[1][0], line[1][1], 100)
        fzs = np.linspace(line[2][0], line[2][1], 100)
        ax.plot3D(fxs, fys, fzs, color='gray', linewidth=0.5, alpha=0.7)

    # draw arrow for beam direction
    a = Arrow3D([100.,300.], [0.,0.], 
                [10.,-10.], mutation_scale=20, 
                lw=3, arrowstyle="-|>", color="purple")
    a.set_zorder(-1)
    ax.add_artist(a)
    ax.text(120.,5.,5.,r'$\nu$-beam',(1,0,-0.1),color='purple',weight='bold')
    

    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    set_axes_equal(ax, bounds)

    ax.set_xlabel('z [cm]',size=12,weight='bold')
    ax.set_ylabel('x [cm]',size=12,weight='bold')
    ax.set_zlabel('y [cm]',size=12,weight='bold')

    # Now set color to white (or whatever is "invisible")
    #ax.xaxis.pane.set_edgecolor('w')
    #ax.yaxis.pane.set_edgecolor('w')
    #ax.zaxis.pane.set_edgecolor('w')
    #ax.set_axis_off()

    fig.subplots_adjust(left=-0.9,right=1.9,bottom=0.05,top=1.05)

    # Bonus: To get rid of the grid as well:
    ax.grid(False)
    plt.savefig(figfile)
    #plt.show()

    ax.view_init(90,-90)
    plt.savefig(figfile+"_birdseye")

    ax.view_init(0,-90)
    plt.savefig(figfile+"_sideview")


def plot_hits(segments, min_time, max_time):
    z_offset = 0
    x_offset = 0
    y_offset = 0
    drawn_objects = []
    for itrk in range(len(segments)):
        #if not itrk%10 == 0: continue
      #  print(itrk, '/', len(segments))
        segment = segments[itrk]
        if segment['t'] < min_time or segment['t'] > max_time:
            continue
       # latex_name = Particle.from_pdgid(segment['pdgId']).latex_name
        #html_name = Particle.from_pdgid(segment['pdgId']).html_name.replace('&pi;','&#960;').replace('&gamma;','&#947;')
        line = go.Scatter3d(x=np.linspace(segment['z_start'] + x_offset, segment['z_end'] + x_offset, 2),
                            y=np.linspace(segment['x_start'] + y_offset, segment['x_end'] + y_offset, 2),
                            z=np.linspace(segment['y_start'] + z_offset, segment['y_end'] + z_offset, 2),
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

def in_range(val, _range):
    return (val > _range[0] and val < _range[1])

def all_in_range(vals, _range):
    return all([in_range(val, _range[i]) for i, val in enumerate(vals)])

def any_in_range(vals, _range):
    return any([in_range(val, _range[i]) for i, val in enumerate(vals)])

def select_in_range(pt_list, _range):
    return [pt for pt in pt_list if in_range(pt, _range)]

def coerce_in_range(_start_pt, _end_pt, start_direction, bounds, scale):
    start_pt = [val/scale for val in _start_pt]
    end_pt = [val/scale for val in _end_pt]
    if all_in_range(start_pt, bounds) and all_in_range(end_pt, bounds):
        return start_pt[0], start_pt[1], start_pt[2], end_pt[0], end_pt[1], end_pt[2], True

    xs = select_in_range(np.linspace(start_pt[0], end_pt[0], 100), bounds[0])
    ys = select_in_range(np.linspace(start_pt[1], end_pt[1], 100), bounds[1])
    zs = select_in_range(np.linspace(start_pt[2], end_pt[2], 100), bounds[2])

    if len(xs) < 2 or len(ys) < 2 or len(zs) < 2:
        xs = select_in_range(np.linspace(start_pt[0], end_pt[0], 1000), bounds[0])
        ys = select_in_range(np.linspace(start_pt[1], end_pt[1], 1000), bounds[1])
        zs = select_in_range(np.linspace(start_pt[2], end_pt[2], 1000), bounds[2])
        if len(xs) < 2 or len(ys) < 2 or len(zs) < 2:
            return 0,0,0,0,0,0, False

    return min(xs), min(ys), min(zs), max(xs), max(ys), max(zs), True
    
def plot_tracks(tracks, min_time, max_time):
    z_offset = 0
    x_offset = 0
    y_offset = 0
    pdgs = []
    drawn_objects = []
    x_bounds = [-356.7, 356.7]
    y_bounds = [-148.613, 155.387]
    z_bounds = [413.72, 916.68]
    bounds = [x_bounds, y_bounds, z_bounds]

    new_color_dict = {5 : 'green', 8 : 'green', 9 : 'green', 0 : 'green'}
    new_color_dict = defaultdict(lambda: 'gray', new_color_dict)


    #first_track_mask = tracks['trackID']==0 
    #event_vertex = tracks['eventID','xyz_start'][first_track_mask] 

    for itrk in range(len(tracks)):
        track = tracks[itrk]
        if track['t_start'] < min_time or track['t_start'] > max_time: continue
        if track['pdgId'] in [2112, 14, 14, 16]: continue

        latex_name = Particle.from_pdgid(track['pdgId']).latex_name

        ##############
        ## enforcing tracks are in range
        ##
        x_start, y_start, z_start, x_end, y_end, z_end, flag = coerce_in_range(track['xyz_start'], track['xyz_end'], track['pxyz_start'], bounds, 10)
        if not flag: continue

        html_name = Particle.from_pdgid(track['pdgId']).html_name.replace('&pi;','&#960;').replace('&gamma;','&#947;')
        line = go.Scatter3d(x=np.linspace(z_start, z_end, 2),
                            y=np.linspace(x_start, x_end, 2),
                            z=np.linspace(y_start, y_end, 2),
                            mode='lines',
                            hoverinfo='text',
                            name=r'$%s$' % latex_name,
                            text='<br>pdgId: %i<br>trackId: %i<br>eventID: %i<br>start_time: %f<br>p:%f' % (track['pdgId'], itrk, track['eventID'], track['t_start'],np.linalg.norm(track['pxyz_start'])),
                            opacity=0.3,
                            legendgroup='%i_%i' % (0,track['pdgId']),
                            # legendgrouptitle_text=r'$%s$' % latex_name,
                            customdata=['track_%i' % itrk],
                            showlegend=track['pdgId'] not in pdgs,
                            line=dict(
                                color=new_color_dict[track['eventID']],
                                width=3
                            ))
        
        if track['pdgId'] not in pdgs:
            pdgs.append(track['pdgId'])
        drawn_objects.append(line)
        drawn_objects.append(line)

    return drawn_objects

def main(filename, min_time, max_time, path_to_geometry, path_to_pixels, event_ids, draw_plotly=False, draw_mc=False, still=False):
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

    #good_event_ids = [0]

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
    figfile = filename.split(".h5",1)[0]
    if not draw_plotly:

        plot_hits_plt(good_segments, min_time, max_time, figfile)
        return

    if not draw_mc:
        drawn_objects = plot_hits(good_segments, min_time, max_time)
    if draw_mc:
        drawn_objects = plot_tracks(good_tracks, min_time, max_time)
    drawn_objects.extend(plot_geometry())
    print('drawing', len(drawn_objects), 'traces')
    print('adding traces to display...')
    fig.add_traces(drawn_objects)
    print('done')

    if not still:
        print('Rendering display...')
        fig.write_html('plot.html', auto_open=True)
    else:
        print('Creating image...')
        pio.kaleido.scope.mathjax = None
        pio.write_image(fig, file='test.png', format="png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-i', type=str, help='''edep-sim file converted to hdf5''')
    parser.add_argument('--path_to_geometry', default=_default_path_to_geometry, type=str, help='''Filename with module geometry info''')
    parser.add_argument('--path_to_pixels', default=_default_path_to_pixels, type=str, help='''Filename with pixel layout''')
    parser.add_argument('--min_time', '-n', type=float, default=0, help='''Min time to start plotting''')
    parser.add_argument('--max_time', '-x', type=float, default=999999999, help='''Max time to end plotting''')
    parser.add_argument('--draw_mc', '-mc', action='store_true', help='''Flag to draw mc truth''')
    parser.add_argument('--still', action='store_true', help='''Flag to write image instead of interactive display''')
    parser.add_argument('--event_ids', type=str, default=None, help='''EventIds to draw, comma separated list''')
    parser.add_argument('--draw_plotly', action='store_true', help='''Use plotly instead of matplotlib''')
    args = parser.parse_args()
    c = main(**vars(args))

