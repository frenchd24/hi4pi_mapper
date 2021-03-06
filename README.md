# HI4PI Mapper
A Python tool for creating plots of HI4PI data.


Run without any commands for basic usage help:

$ python hi4pi_mapper.py\
usage: hi4pi_mapper.py [-h] [-c COORDS COORDS] [-s SIZE] [-v VRANGE VRANGE]
                       [-f TARG_FILE] [--cubes] [--spec]

optional arguments:\
  -h, --help        show this help message and exit\
  -c COORDS COORDS  RA and Dec to center plots on. E.g., '-c ra dec'\
  -s SIZE           Size of zoom-in maps in degrees.\
  -v VRANGE VRANGE  What velocity range to use? E.g., '-v -150 -40'\
  -f TARG_FILE      Enter the name of a csv file containing the above\
                    information to plot for multiple targets.\
  --cubes           This option prints out the needed data cubes and does not\
                    generate any plots.\
  --spec            This option saves the extracted spectrum to file.\


## INSTALLING:
To install requirements for HI4PI Mapper, run the following:\
$ pip install -r requirements.txt



## INPUT FILE:
You can generate plots for a list of coordinates by input a target file. This should be a txt file with the following data format:\
ra,dec,zoom_size,velocity_min,velocity_max

Example for 2 targets:\
-37,-50,1,-50,40\
20,20,2,-150,-40

An example file "targets.txt" is included.



## EXAMPLES:
Generate a single plot around the coordinate ra=50, dec=50, with velocity bounds (-100, 100):\
$ python hi4pi_mapper.py -c 50 50 -v -100 100

Generate the same plot but with the zoom-in windows having size=3 degrees:\
$ python hi4pi_mapper.py -c 50 50 -v -100 100 -s 3

Generate multiple plots from a target file containing coordinates, zoom-in size, and velocity bounds:\
$ python hi4pi_mapper.py -f targets.txt

Just print out a list of all the data cubes you'll need to download to create the plots you want:\
$ python hi4pi_mapper.py -f targets.txt --cubes

The --cubes command works for all the examples above (i.e., single or multiple plots).

The data cubes should be downloaded from here: http://cdsarc.u-strasbg.fr/ftp/J/A+A/594/A116/CUBES/EQ2000/ \
Save them in the /hi4pi/ folder located in the same directory as this code.



Reference: 2016A&A...594A.116H
