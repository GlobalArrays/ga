# python libs
from getopt import getopt
import sys

# 3rd party libs
from PIL import Image

# my libs
from random_values import *

nPt = 8 # default
expo = 2 # default distance exponent (Euclidean)
nGrid = 512 # image size (in each dimension)
useGain = False
profile = False
help = False

imgOutput = True
rawOutput = False
img_name = "distmap.gif"
raw_name = "distmap.dat"
usage = """
distmap.py [options]

  -h        help (prints this message)
  -g        use gain instead of numpy
  -e expo   set exponent (default: 2 (Euclidean))
  -o out    output filename (default: distmap.gif)
  -r raw    raw output filename (default: distmap.dat)
  -p points number of points (default: 8)
  -s seed   for random number generation
  -w width  image size e.g. Width X Width square
  -d        whether to enable profiling

"""

(optsvals, args) = getopt(sys.argv[1:], 'hge:o:r:p:s:w:d')
for (opt, val) in optsvals:
    if opt == '-h':
        help = True
    elif opt == '-g':
        useGain = True
    elif opt == '-e':
        expo = float(val)
    elif opt == '-o':
        img_name = str(val)
        imgOutput = True
    elif opt == '-r':
        raw_name = str(val)
        rawOutput = True
    elif opt == '-p':
        nPt = int(val)
    elif opt == '-s':
        myrandseed(int(val));
    elif opt == '-w':
        nGrid = int(val)
    elif opt == '-d':
        profile = True

if useGain:
    from ga import gain as np
    from ga.gain import me
    if help:
        if not me:
            print usage
        sys.exit()
else:
    import numpy as np
    if help:
        print usage
        sys.exit()
my_dtype = np.float32


# generate nPt random points within the grid
pt = []
for k in range(nPt):
    x = nGrid * myrandf()
    y = nGrid * myrandf()
    pt.append((x, y))

def dist((x, y), (x0, y0)):

    """Returns the Euclidean distance between two (2D) points."""

    return ((x-x0)**expo + (y-y0)**expo)**(1.0/expo)

def dist_array((x0, y0)):

    """Returns an array whose elements are the distance (in units of
    rows and columns) to a given point (x0, y0)."""

    f = lambda x, y, x0=x0, y0=y0: dist((x, y), (x0, y0))
    a = np.fromfunction(f, (nGrid, nGrid), dtype=my_dtype)

def main():
    # distMin[i,j] is the distance from pixel (i,j) to the closest 'pt'.
    distMin = dist_array(pt[0])
    for i in range(1, nPt):
        distMin = np.minimum(dist_array(pt[i]), distMin)
    
    pxlMin = 0
    pxlMax = np.maximum.reduce(distMin)
    
    # scale and offset distances to lie between 0 and 1
    distMinScaled = (distMin - pxlMin) / (pxlMax - pxlMin)
    
    # (gray) pixel values lie between 0 and 255
    pxls = 255 * distMinScaled
    
    if (imgOutput or rawOutput) and useGain:
        if nodeid() == 0:
            pxls = pxls.__array__()
        else:
            return
    
    #print pxls
    
    if imgOutput:
        im_r = Image.frombuffer("F", pxls.shape, pxls, "raw", "F", 0, 1).convert("L")
        im_r.save(img_name)
    
    if rawOutput:
        f = open(raw_name, "wb")
        f.write(buffer(pxls.astype(np.float32)))
        f.close()

if __name__ == '__main__':
    if profile:
        import cProfile
        if useGain:
            cProfile.run("main()", "distmap.prof" + str(nodeid()))
        else:
            cProfile.run("main()", "distmap.prof")
    else:
        main()

