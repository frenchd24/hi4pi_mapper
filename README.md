# HI4PI Mapper
A Python tool for creating plots of HI4PI data.

Run without any commands for basic usage help:

$ python hi4pi_mapper.py
usage: hi4pi_mapper.py [-h] [-c COORDS COORDS] [-s SIZE] [-v VRANGE VRANGE]
                       [-f TARG_FILE] [--cubes]

optional arguments:
  -h, --help        show this help message and exit
  -c COORDS COORDS  RA and Dec to center plots on. E.g., '-c ra dec'
  -s SIZE           Size of zoom-in maps in degrees.
  -v VRANGE VRANGE  What velocity range to use? E.g., '-v -150 -40'
  -f TARG_FILE      Enter a file containing the above information to plot for
                    multiple targets.
  --cubes           Enter the name of a csv file containing the above
                    information to plot for multiple targets.


INSTALLING:
To install requirements for HI4PI Mapper, run the following:
$ pip install -r requirements.txt

INPUT FILE:
You can generate plots for a list of coordinates by input a target file. This should be a txt file with the following data format:
ra,dec,zoom_size,velocity_min,velocity_max

Example for 2 targets:
-37,-50,1,-50,40
20,20,2,-150,-40

An example file "targets.txt" is included.
