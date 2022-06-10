#!/usr/bin/env python
'''
This module is for plotting HI 21 cm data from the HI4PI survey:
2016A&A...594A.116H

The file mapper_settings.yaml contains most of the adjustable parameters
for making plots, and should be located in the same directory as this code.


'''

__author__ = "David M. French"
__email__ = "dfrench@stsci.edu"

import os, glob, sys
import shutil
import argparse
import warnings
import csv

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table, hstack, vstack, Column

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import yaml

from spectral_cube import SpectralCube
from spectral_cube.utils import SpectralCubeWarning
warnings.filterwarnings(action='ignore', category=SpectralCubeWarning,
                        append=True)

from reproject import reproject_interp, reproject_from_healpix

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def find_hi4pi_fits_name(coords):
    """ Identifies the HI4Pi cube that the coordinate lives in

    Inputs:
        coords: central coordinate in format [RA, Dec] in degrees (J2000)
    """
    # load in key
    hkey = Table.read("cubes_eq.dat", format="ascii")
    ra_list = np.array(hkey["col1"], dtype="float")
    dec_list = np.array(hkey["col2"], dtype="float")
    fits_list = hkey["col4"]

    c_all = SkyCoord(ra_list * u.deg, dec_list * u.deg, frame="icrs")
    c_coord = SkyCoord(coords[0] * u.deg, coords[1] * u.deg, frame="icrs")
    idx, d2d, d3d = c_coord.match_to_catalog_sky(c_all)

    return fits_list[idx]


def make_velocity_axis(h):
    """ Creates the velocity axis given a pyfits header. Assumes the third
    axis is the velocity axis in km/s using the radio definition.

    Inputs:
        h: header
    """
    array = (np.arange(h["NAXIS3"]) - h["CRPIX3"] + 1) * h["CDELT3"] + h["CRVAL3"]
    return array/1000.


# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# --- this is the main plotting function

def plot_maps(coords, size, vrange, settings):
    """ Extracts cutout from HI4Pi cube based on coordinate,
    field size, and velocity range and makes plots:

    Inputs:
        coords: central coordinate in format [RA, Dec] in degrees (J2000)
        size: field size (width /2.) in degrees
        vrange: velocity range in format [vmin, vmax]
        settings: yaml file containing mapper settings
    """

    # --- directory where all the HI4Pi cubes are downloaded to
    data_dir = settings['data_dir']

    if data_dir == '/hi4pi/':
        current_dir = os.getcwd()
        hi4pi_dir = current_dir + data_dir
    else:
        hi4pi_dir = data_dir

    print('Searching for data in: ',hi4pi_dir)
    # hi4pi_dir = '/Users/dfrench/hvc_metallicity/hi4pi/'

    # --- find the HI4Pi cube of interest
    filename = hi4pi_dir + find_hi4pi_fits_name(coords)

    # --- open the cube
    cube = SpectralCube.read(filename)
    cube = cube.with_spectral_unit(u.km / u.s)

    # --- define the WCS
    wcs_ref = cube.wcs
    header = cube.wcs.to_header()

    # --- identify the pixel location of the coordinate
    source = wcs_ref.all_world2pix(coords[0], coords[1], 0, 0)
    # print('coords: ',coords)
    # print('source: ',source)

    # --- convert to nearest integer for extracting sub-cube below
    # --- this is the center
    xval = int(source[0])
    yval = int(source[1])
    # print('xval: ',xval)
    # print('yval: ',yval)
    # print()

    # --- central spectrum
    spec_cen = cube[:, yval, xval].value

    # --- compute the size of the field in pixel units
    size_pix = int(size / np.abs(header['CDELT1']))

    # --- compute the velocity axis of the cube
    vlsr = cube.spectral_axis.value

    # --- identify the range from vrange
    vlsr_selected = np.argwhere((vlsr > vrange[0]) & (vlsr <= vrange[1])).ravel()

    # --- extract velocity restricted cube
    cube_restricted = cube[np.nanmin(vlsr_selected):np.nanmax(vlsr_selected),:,:]

    # --- extract sub_cube by indexing
    sub_cube = cube[np.nanmin(vlsr_selected):np.nanmax(vlsr_selected),yval-size_pix:yval+size_pix+1,xval-size_pix:xval+size_pix+1]
#     sub_cube = get_subcube(coords, size, vrange)

    # --- identify the pixel location of the coordinate in the sub_cube
    sub_wcs_ref = sub_cube.wcs
    sub_source = sub_wcs_ref.all_world2pix(coords[0], coords[1], 0, 0)

    # --- convert to nearest integer for extracting sub-cube below
    sub_xval = int(sub_source[0])
    sub_yval = int(sub_source[1])

# ------------------------------------------------------------------------------
    if settings['save_spectrum']:
        save_name = 'hi4pi_spec_ra_{}_dec_{}_vrange_{}_{}.csv'.format(\
            str(coords[0]).replace('.','p'), str(coords[1]).replace('.','p'), int(vrange[0]), int(vrange[1]))

        save_file = os.path.join(settings['save_dir'], save_name)
        fieldnames = ('vlsr', 'spec')
        output = np.array([vlsr, spec_cen])

        with open(save_file, 'w') as file:
            writer = csv.writer(file)

            writer.writerow(fieldnames)
            writer.writerows(output.T)

# ------------------------------------------------------------------------------
    # --- creating the figure
    plots_to_make = settings['plots_to_make']
    num_plots = len(plots_to_make)

    # --- this defines the row and column order the plots should be made in
    # --- this order makes things look nice no matter which plots are made
    row_order = [0, 0, 1, 1, 0, 1, 0, 1]
    col_order = [0, 1, 0, 1, 2, 2, 3, 3]

    if num_plots >2:
        nrows = 2
    else:
        nrows = 1

    ncols = int(np.ceil(num_plots/nrows))

    xfigsize, yfigsize = 18*(ncols/4), 6*(nrows/2)

    # --- now make the figure
    fig = plt.figure(figsize=(xfigsize, yfigsize), constrained_layout=False)

    # --- shrink the colorbar by this fraction
    cmap_shrink=1

    # --- create this pad between the figure and colorbar
    cmap_pad=0.02

    # --- force this aspect ratio for the plots
    aspect_ratio = 1

    # --- define the grid and spacing for the plots
    width = 16.0
    height = 9.0
    left = 0.125
    right = 0.9
    bottom = 0.1
    top = 0.9
    wspace = 0.10
    hspace = 0.38

    spec = gridspec.GridSpec(ncols=ncols,
                            nrows=nrows,
                            figure=fig,
                            left=left,
                            right=right,
                            top=top,
                            bottom=bottom,
                            hspace=hspace,
                            wspace=wspace)


    # --- now loop through and make each plot in order
    axs = [] # all the axes are added to this list, but not currently used
    for i, plot in enumerate(plots_to_make):
        print('i, plot: ',i, plot)
        row = row_order[i]
        col = col_order[i]
        if 'nhi_zoom' == plot:
            ax_nhi_zoom = fig.add_subplot(spec[row, col], projection=sub_cube.wcs, slices=("x", "y", 0))

            # --- convert coordinates and ticks to degrees
            lon=ax_nhi_zoom.coords[0]
            lat=ax_nhi_zoom.coords[1]
            lon.set_major_formatter('dd')
            lat.set_major_formatter('dd')

            # --- show the image
            nhi_zoom = sub_cube.moment(order=0) * 0.01823
            im_nhi_zoom = ax_nhi_zoom.imshow(nhi_zoom.value,
                                             origin='lower',
                                             cmap=settings['cm_nhi_zoom'],
                                             aspect=aspect_ratio)

            # --- plot a little red 'x'
            if settings['mark_target']:
                ax_nhi_zoom.scatter(sub_xval, sub_yval, marker='x', color='red')


            # --- colorbar
            plt.colorbar(im_nhi_zoom, ax=ax_nhi_zoom, label=r'$N({\rm HI})\rm\,[10^{20}\,cm^{-2}]$',
                         location='right', shrink=cmap_shrink, pad=cmap_pad)

            # --- set title and axes labels
            ax_nhi_zoom.set_title('N(HI) (zoom)')
            ax_nhi_zoom.set_xlabel('Right Ascension')
            ax_nhi_zoom.set_ylabel('Declination')

            axs.append(ax_nhi_zoom)

        elif 'nhi' == plot:
            ax_nhi = fig.add_subplot(spec[row, col], projection=cube.wcs, slices=("x", "y", 0))

            # --- show the image
            # im_nhi = ax_nhi.imshow(np.nansum(cube[np.nanmin(vlsr_selected):np.nanmax(vlsr_selected), : , :], axis=0),
            #                        origin='lower',
            #                        cmap=settings['cm_nhi'],
            #                        aspect=aspect_ratio)
            nhi_cube = cube_restricted.moment(order=0) * 0.01823
            im_nhi = ax_nhi.imshow(nhi_cube.value,
                                   origin='lower',
                                   cmap=settings['cm_nhi'],
                                   aspect=aspect_ratio)

            # --- convert coordinates and ticks to degrees
            lon=ax_nhi.coords[0]
            lat=ax_nhi.coords[1]
            lon.set_major_formatter('dd')
            lat.set_major_formatter('dd')

            # --- colorbar
            plt.colorbar(im_nhi, ax=ax_nhi, label=r'$N({\rm HI})\rm\,[10^{20}\,cm^{-2}]$',
                         location='right', shrink=cmap_shrink, pad=cmap_pad)

            # --- add rectangle to plot to show zoom region
            rect_kwargs = {'linewidth':1.0,
                          'edgecolor':'red',
                          'fill':False}
            rect = Rectangle((xval-size_pix, yval-size_pix), size_pix*2, size_pix*2,**rect_kwargs)
            ax_nhi.add_patch(rect)

            # --- plot an 'x' on the target coordinates
            if settings['mark_target']:
                ax_nhi.scatter(xval, yval, marker='x', color='red')

            # --- set title and axes labels
            ax_nhi.set_title('N(HI)')
            ax_nhi.set_xlabel('Right Ascension')
            ax_nhi.set_ylabel('Declination')

            axs.append(ax_nhi)

        elif 'velocity' == plot:
            ax_vel = fig.add_subplot(spec[row, col], projection=cube.wcs, slices=("x", "y", 0))

            # --- compute the first moment cube
            vel_cube = cube[np.nanmin(vlsr_selected):np.nanmax(vlsr_selected),:,:]
            vel = vel_cube.moment(order=1)

            # --- show the image
            im_vel = ax_vel.imshow(vel.value,
                                   origin='lower',
                                   cmap=settings['cm_velocity'],
                                   aspect=aspect_ratio,
                                   vmin=vrange[0],
                                   vmax=vrange[1])

            # --- plot an 'x' on the target coordinates
            if settings['mark_target']:
                ax_vel.scatter(xval, yval, marker='x', color='red')

            # --- colorbar
            plt.colorbar(im_vel, ax=ax_vel, label=r'$\rm <Velocity>\, [km\,s^{-1}]$',
                         location='right', shrink=cmap_shrink, pad=cmap_pad)

            # --- convert coordinates and ticks to degrees
            lon=ax_vel.coords[0]
            lat=ax_vel.coords[1]
            lon.set_major_formatter('dd')
            lat.set_major_formatter('dd')

            # --- set the title and axes labels
            ax_vel.set_title('Velocity')
            ax_vel.set_xlabel('Right Ascension')
            ax_vel.set_ylabel('Declination')

            axs.append(ax_vel)

        elif 'velocity_zoom' == plot:
            ax_vel_zoom = fig.add_subplot(spec[row, col], projection=sub_cube.wcs, slices=("x", "y", 0))

            # --- compute the first moment of the zoomed-in cube
            vel_zoom = sub_cube.moment(order=1)
            try:
                im_vel_zoom = ax_vel_zoom.imshow(vel_zoom.value,
                                                 origin='lower',
                                                 cmap=settings['cm_velocity_zoom'],
                                                 aspect=aspect_ratio,
                                                 vmax=vrange[1],
                                                 vmin=vrange[0])
            except IndexError as e:
                print('Could not plot with given vmax and vmin, reverting to defaults.')
                im_vel_zoom = ax_vel_zoom.imshow(vel_zoom.value,
                                                 origin='lower',
                                                 cmap=settings['cm_velocity_zoom'],
                                                 aspect=aspect_ratio)

            # --- plot an 'x' on the target coordinates
            if settings['mark_target']:
                ax_vel_zoom.scatter(sub_xval, sub_yval, marker='x', color='red')

            # --- colorbar
            plt.colorbar(im_vel_zoom, ax=ax_vel_zoom, label=r'$\rm <Velocity>\, [km\,s^{-1}]$',
                         location='right', shrink=cmap_shrink, pad=cmap_pad)

            # --- convert coordinates and ticks to degrees
            lon=ax_vel_zoom.coords[0]
            lat=ax_vel_zoom.coords[1]
            lon.set_major_formatter('dd')
            lat.set_major_formatter('dd')

            # --- set the title and axes labels
            ax_vel_zoom.set_title('Velocity (zoom)')
            ax_vel_zoom.set_xlabel('Right Ascension')
            ax_vel_zoom.set_ylabel('Declination')

            axs.append(ax_vel_zoom)

        elif 'dispersion' == plot:
#             ax_disp = plt.subplot2grid((nrows,ncols), (row, col), projection=cube.wcs, slices=('x', 'y', 0))
            ax_disp = fig.add_subplot(spec[row, col], projection=cube.wcs, slices=("x", "y", 0))

            # --- compute the second moment cube
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # --- the Spectral Cube commands for second moment don't work
                # --- so do it manually
                disp = cube.moment(order=2)
                disp = np.sqrt(disp.value)

                if settings['dispersion_type'] == 'sigma':
                    # disp_map_zoom = sub_cube.linewidth_sigma()
                    disp_map = disp
                else:
                    # disp_map_zoom = sub_cube.linewidth_fwhm()
                    disp_map = disp * np.sqrt(8 * np.log(2))

            # -- show the image
            im_disp = ax_disp.imshow(disp_map,
                                     origin='lower',
                                     cmap=settings['cm_dispersion'],
                                     aspect=aspect_ratio,
#                                      vmax=500,
                                     )

            # --- plot an 'x' on the target coordinates
            if settings['mark_target']:
                ax_disp.scatter(xval, yval, marker='x', color='red')

            # --- colorbar
            if settings['dispersion_type']:
                plt.colorbar(im_disp, ax=ax_disp, label=r'$\rm <\sigma>\, [km\,s^{-1}]$',
                             location='right', shrink=cmap_shrink, pad=cmap_pad)
            else:
                plt.colorbar(im_disp, ax=ax_disp, label=r'$\rm <FWHM>\, [km\,s^{-1}]$',
                             location='right', shrink=cmap_shrink, pad=cmap_pad)

            # --- convert coordinates and ticks to degrees
            lon=ax_disp.coords[0]
            lat=ax_disp.coords[1]
            lon.set_major_formatter('dd')
            lat.set_major_formatter('dd')

            # --- set the title and axes labels
            ax_disp.set_title('Dispersion')
            ax_disp.set_xlabel('Right Ascension')
            ax_disp.set_ylabel('Declination')

            axs.append(ax_disp)

        elif 'dispersion_zoom' == plot:
            ax_disp_zoom = fig.add_subplot(spec[row, col], projection=sub_cube.wcs, slices=("x", "y", 0))

            # --- compute the second moment sub-cube
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # --- the Spectral Cube commands for second moment don't work
                # --- so do it manually
                disp_zoom = sub_cube.moment(order=2)
                disp_zoom = np.sqrt(disp_zoom.value)

                if settings['dispersion_type'] == 'sigma':
                    # disp_map_zoom = sub_cube.linewidth_sigma()
                    disp_zoom_map = disp_zoom
                else:
                    # disp_map_zoom = sub_cube.linewidth_fwhm()
                    disp_zoom_map = disp_zoom * np.sqrt(8 * np.log(2))


            # --- show the image
            im_disp_zoom = ax_disp_zoom.imshow(disp_zoom_map,
                                               origin='lower',
                                               cmap=settings['cm_dispersion_zoom'],
                                               aspect=aspect_ratio)

            # --- plot an 'x' on the target coordinates
            if settings['mark_target']:
                ax_disp_zoom.scatter(sub_xval, sub_yval, marker='x', color='red')

            # --- colorbar
            if settings['dispersion_type'] == 'sigma':
                plt.colorbar(im_disp_zoom, ax=ax_disp_zoom, label=r'$\rm <\sigma>\, [km\,s^{-1}]$',
                             location='right', shrink=cmap_shrink, pad=cmap_pad)
            else:
                plt.colorbar(im_disp_zoom, ax=ax_disp_zoom, label=r'$\rm <FWHM>\, [km\,s^{-1}]$',
                             location='right', shrink=cmap_shrink, pad=cmap_pad)

            # --- convert coordinates and ticks to degrees
            lon=ax_disp_zoom.coords[0]
            lat=ax_disp_zoom.coords[1]
            lon.set_major_formatter('dd')
            lat.set_major_formatter('dd')

            # --- set the title and axes labels
            ax_disp_zoom.set_title('Dispersion (zoom)')
            ax_disp_zoom.set_xlabel('Right Ascension')
            ax_disp_zoom.set_ylabel('Declination')

            axs.append(ax_disp_zoom)

        elif 'spectrum' == plot:
            ax_spec = fig.add_subplot(spec[row, col])

            # --- plot the spectrum
            ax_spec.plot(vlsr, spec_cen, color='black')

            # --- overplot the zoom-in velocity region in red
            ax_spec.plot(sub_cube.spectral_axis.value, sub_cube[:, sub_xval, sub_yval], color='red', lw=3)

            # --- adjust the limits and tick locations
            try:
                ax_spec.set_ylim(-1,max(spec_cen)+10)
            except ValueError as e:
                print('Could not set y limits to given value, reverting to defaults.')

            ax_spec.set_xlim(-200, 200)
            ax_spec.yaxis.tick_right()
            ax_spec.yaxis.set_label_position("right")

            # --- set the axis labels
            ax_spec.set_title('Target Spectrum')
            # ax_spec.set_ylabel(r'$T_B\rm\,K$')
            ax_spec.set_xlabel(r'$v_{\rm LSR}\rm\,[km\,s^{-1}]$')
            # ax_spec.set_xlabel('Velocity (km/s)')
            ax_spec.set_ylabel(r'$\rm T_B $ (K)')

            axs.append(ax_spec)

        elif 'spectrum_zoom' == plot:
            ax_spec_zoom = fig.add_subplot(spec[row, col])

            # --- compute the zoom-in spectrum
            vels = sub_cube.spectral_axis.value
            temps = sub_cube[:, size_pix, size_pix]

            # --- plot the zoom-in spectrum
            ax_spec_zoom.plot(sub_cube.spectral_axis.value, sub_cube[:, sub_xval, sub_yval], color='red', lw=3)

            # --- adjust the limits and tick locations
            try:
                ax_spec_zoom.set_ylim(0,max(temps.value)*1.1)
            except ValueError as e:
                print('Could not set y limits to given value, reverting to defaults.')

            ax_spec_zoom.set_xlim(vrange[0], vrange[1])
            ax_spec_zoom.yaxis.tick_right()
            ax_spec_zoom.yaxis.set_label_position("right")

            # --- set the title and axes labels
            ax_spec_zoom.set_title('Target Spectrum (zoom)')
            # ax_spec_zoom.set_ylabel(r'$T_B\rm\,K$')
            ax_spec_zoom.set_xlabel(r'$v_{\rm LSR}\rm\,[km\,s^{-1}]$')
            # ax_spec_zoom.set_xlabel('Velocity (km/s)')
            ax_spec_zoom.set_ylabel(r'$\rm T_B $ (K)')

            axs.append(ax_spec_zoom)


        else:
            print('Undefined plot type: ',plot)
            print('Exiting...')
            sys.exit()

    fig.tight_layout()

    return fig

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", dest="coords", type=float, nargs=2,
                        help="RA and Dec to center plots on. E.g., '-c ra dec'")
    parser.add_argument("-s", dest="size", type=float, default=2,
                        help="Size of zoom-in maps in degrees.")
    parser.add_argument("-v", dest='vrange', type=float, nargs=2,
                        default=[-100,100],
                        help="What velocity range to use? E.g., '-v -150 -40' ")
    parser.add_argument('-f', dest='targ_file', type=str,
                        help="Enter the name of a csv file containing the above information to plot for multiple targets.")
    parser.add_argument('--cubes', dest='identify_cubes', action="store_true",
                        default=False,
                        help="This option prints out the needed data cubes and does not generate any plots.")
    parser.add_argument('--spec', dest='save_spectrum', action="store_true",
                        default=False,
                        help="This option saves the extracted spectrum to file.")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    if args.targ_file == None:
        if args.identify_cubes == None:
            if args.coords == None:
                raise ValueError('Need to specify coordinates')
            if args.vrange == None:
                raise ValueError("Need to specify velocity range")
        else:
            if args.coords == None:
                raise ValueError('Need to specify coordinates')
    return args

def main(args):

    # --- read in the settings file
    yaml_file = os.path.join(os.getcwd(), 'mapper_settings.yaml')
    with open(yaml_file) as file:
        settings = yaml.safe_load(file)

    save_dir = settings['save_dir']

    # --- over-ride the settings file if save_spectrum is specified
    if args.save_spectrum:
        settings['save_spectrum'] = True

    # --- parse the target file if there is one
    if args.targ_file !=None:
        ra, dec, s, vlow, vhigh = np.loadtxt(args.targ_file, unpack=True, delimiter=',', ndmin=2)
        coords = np.array([ra, dec]).T
        vrange = np.array([vlow, vhigh]).T
        size = np.array(s)
    else:
        coords = [args.coords]
        size = [args.size]
        vrange = [args.vrange]

    if args.identify_cubes:
        print('The following data cubes are needed: ')
        print()

    for c, s, v in zip(coords, size, vrange):

        if args.identify_cubes:
            cube = find_hi4pi_fits_name(c)
            print(cube)

        else:
            print('Now creating map for coords={}, size={}, vrange={}'.format(c, s, v))
            fig = plot_maps(c, s, v, settings)

            save_name = 'hi4pi_map_ra_{}_dec_{}_vrange_{}_{}.pdf'.format(str(c[0]).replace('.','p'), str(c[1]).replace('.','p'), int(v[0]), int(v[1]))
            save_file = os.path.join(save_dir, save_name)

            fig.savefig(save_file, bbox_inches = 'tight')


    print()
    if args.identify_cubes:
        print('Please download the needed data cubes from here: http://cdsarc.u-strasbg.fr/ftp/J/A+A/594/A116/CUBES/EQ2000/')
        print("Save them in the /hi4pi/ folder located in the same directory as this code.")
        print()
    print("Finished")
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    args = parser()
    main(args)
