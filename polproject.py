#!/usr/bin/env python

'''
        polproject.py
        
        Usage: python polproject.py [input-image] [[output-image]]
        
        Inputs:
                input-image: String, FITS filename. The image to be projected into polar coordinates by interpolation.
                
                output-image: Optional. String, FITS filename If provided, the polar projection of the input image will
                              be saved to this filename or path.
        
        Outputs: Prints infomration about the intensity as a function of phase and the contrast as a function of radius to
                 STDOUT and Matplotlib pyplot windows. Optionally, a FITS file containing the polar projection of the input, with
                 name given by [output-image].
                 
        TODO:    More descriptive and pythonic function names.
                 
        Created by E. Monson on 8/9/2018.
'''

import numpy as np
from astropy.io import fits
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

# Change these to make the figures look how you like.
matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['font.monospace'] = ['Courier','Andale Mono']
matplotlib.rcParams['lines.markersize'] = 8
matplotlib.rcParams['mathtext.default'] = 'regular'

__all__ = ["cart2pol","pol2cart","polarProject","azimuthalScan","contrastAsRadius","spiralTemplate","toImCoords"]

##=========================================================================================================
# Functions
##=========================================================================================================
def cart2pol(x,y):
    '''
        Project floats (or vectors or matrices of floats) x and y into polar coordinates.
    '''
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y,x)
    return r,th
# End definition

def pol2cart(r,th):
    '''
       Project floats (or vectors or matrices of floats) r and th into Cartesian coordinates.
    '''
    x = r*np.cos(th)
    y = r*np.sin(th)
    return x,y
# End definition

def polarProject(data,origin,order=1):
    '''
        Project an image (`data`) from cartesian coordinates into polar coordinates,
        about the origin given by `origin` and with interpolation given by `order`.
        
        data: 2D numpy floating-point array containing image intensity data
        
        origin: integer tuple containing origin of polar coordinate system
        
        order: integer, interpolation order when projecting into new coordinates.
               Defaults to 1, bilinear interpolation.
    '''
    rows,cols = data.shape
    # Make a Cartesian coordinate grid with its origin at `origin`
    x,y = np.meshgrid(np.arange(cols),np.arange(rows))
    x -= origin[0]
    y -= origin[1]
    # Project that coordinate grid into polar coordinates to determine the maximum radius
    r,th = cart2pol(x,y)
    # Make an evenly spaced grid of polar coordinates...
    thgrid, rgrid = np.meshgrid(np.linspace(th.min(),th.max(),cols),np.linspace(r.min(),r.max(),rows))
    # And project that back into Cartesian coordinates.
    xnew,ynew = pol2cart(rgrid,thgrid)
    # Convert those cartesian coordinates back to image grid coordinates
    xnew += origin[0]
    ynew += origin[1]
    # map_coordinates expects coordinates to have shape (N,2)
    # map_coordinates is also looking for coords in (row,col) order, so we swap x and y here. Otherwise
    # the theta axis of the plot will be off by pi/2.
    newcoords = np.vstack((ynew.flatten(),xnew.flatten()))
    # Sample the image at the new gridpoints and interpolate (order = 1 is bilinear).
    interpolated = map_coordinates(data,newcoords,order=order)
    # This ends up giving us a list of values, so we'll reshape it to end up with an image  of the same size,
    # with r on the y-axis and theta on the x-axis.
    newimage = interpolated.reshape((rows,cols))
    
    return newimage, r, th
# End definition

def azimuthalScan(polimage,pmin=0,pmax=None):
    '''
       Given an image with intensity as a function of r and theta, calculate the total intensity per each value
       of theta.
       
       polimage: 2D numpy floating-point array containing image intensity data as a function of polar coordinates.
       
       pmax: high pixel index for the scan. Defaults to the top edge of the image.
       
       pmin: low pixel index for the scan. Defaults to 0.
    '''
    
    rows, cols = polimage.shape

    if (pmax == None):
        pmax = rows
    # End if

    i = 0
    result = np.zeros((cols,2))
    for column in polimage.T:
        intens = np.sum(column[pmin:pmax])
        # Converting from [0,2pi] to [-pi,pi] this way should be okay since the columns of the image are linear
        # in theta.
        phase = 2*np.pi*(float(i)/rows) - np.pi
        result[i,0], result[i,1] = phase, intens
        i+=1
    # End loop

    return result
# End definition

def contrastAsRadius(polimage,rmax,n,pmin=0,pmax=None):
    '''
       A function for finding the contrast as a function of the radial coordinate.
       
       polimage: an image projected into polar coordinates.
       
       rmax: the maximum radial extent of the polar-projected image.
       
       n: the number of pixels to integrate the flux over at each step.
       
       pmin: low pixel index for the scan. Defaults to n/2.
       
       pmax: high pixel index for the scan. Defaults to rows - n/2
    '''

    rows, cols = polimage.shape
    
    dn = n//2
    
    if (pmax == None or rows - pmax < dn):
        pmax = rows-dn
    # End if
    
    if (pmin < dn):
        pmin = dn
    # End if

    result = np.zeros((pmax - pmin,4))
    i = 0
    for pix in range(pmin,pmax):
        intens_phase = azimuthalScan(polimage,pix-dn,pix+dn)
        max = intens_phase[:,1].max()
        min = intens_phase[:,1].min()
        contrast = max/min
        # convert pixel coordinates to physical coordinates
        r = float(pix)/rows*rmax
        result[i,0],result[i,1],result[i,2],result[i,3] = r,contrast,max,min
        i+=1
    # End loop

    return result
# End definition

def spiralTemplate(shape,rmax,pitch,thmax=2*np.pi,phase=0):
    '''
       Generate a single logarithmic spiral in polar coordinates, and return its coordinates
       as a numpy array with shape[0] rows and 2 columns. Column 1 is theta, column 2 is r.
       
       Implemented, currently unused.
    '''
    rows,cols = shape
    # Ideally the points should be very closely spaced.
    # Becuase of the way we're wrapping angles into [-pi,pi], we need to reach into negative
    # angles here. It's weird.
    th = np.linspace(-thmax,0,rows)
    th = th.reshape((rows,1))
    prad = np.pi/180.*pitch
    b = np.tan(-prad)
    # This scale factor definition is from Jazmin's old spiral overlay code.
    a = (1.15*rmax)/np.exp(np.tan(np.abs(prad))*thmax)
    t = th + phase
    r = a*np.exp(b*t)
    # I need the theta values to wrap at the edge...
    # Taking the inverse tangent of the tangent with arctan2 (preserves phase)
    # should work.
    t = np.arctan2(np.sin(t),np.cos(t))
    arm = np.hstack((t,r))
    
    # The thing to do with this spiral template is to sweep it through the polar image at all phases at a given
    # radius and do something like sum(map_coordinates(polar_image,arm)) to determine the flux along the arm template.
    # That way we can interpolate and we're actually getting information from a 2D region of the image rather than a 1D one.
    # Problems: points are more closely spaced near the bottom (bulge/bar) region of the image than at the ends. We want to
    # sample the image in equally spaced intervals along the spiral.
    
    return arm
# End definition

def toImCoords(rth,shape,rmax):
    '''
       Given an array of (theta,r) coordinates, convert them to image coordinates (row,col) corresponding
       to rows and columns of a polar-projected image with shape and maximum r extent rmax.
       Note that the resulting coordinates need not (and will not) be integer values; rather they
       will correspond to locations between pixels that we'll later interpolate at.
    '''
    rows, cols = shape
    i = rth[:,1]/rmax*rows
    j = (rth[:,0]+np.pi)/(2*np.pi)*cols
    # I shouldn't have to reshape like this
    i = i.reshape((len(rth[:,0]),1))
    j = j.reshape((len(rth[:,1]),1))
    return np.hstack((i,j))
# End definition

##=========================================================================================================
# Main
##=========================================================================================================
def main():
    '''
       python polplot.py [input-image] [output-image]
       Take a fits image from the command line and display it in cartesian and polar coordinates
       on a log intensity scale.
       Optionally, given a filename [output-image] the script will save the projected image to the given filename.
    '''
    
    data = fits.getdata(sys.argv[1])
    
    # We'll make two plots on the same figure with the image on top and the azimuthal profile on the bottom.
    fig,ax = plt.subplots(2,figsize=(10,6),dpi=120,sharex=True)
    rows,cols = data.shape
    cx = cols//2
    cy = rows//2

    # Project the data into polar coordinates.
    newimage,r,th = polarProject(data,(cx,cy))
    
    print("\n")
    scanlow = int(raw_input("Lower radial limit for scan: "))
    scanhigh = int(raw_input("Upper radial limit for scan: "))
    
    # Convert the radial limits into row coordinate limits
    pmin = int(float(scanlow)/r.max()*rows)
    pmax = int(float(scanhigh)/r.max()*rows)
    
    # Display the projected image with a Log stretch to highlight the arms.
    ax[0].imshow(newimage,extent=(th.min(),th.max(),r.min(),r.max()),norm=colors.LogNorm(),cmap=plt.get_cmap('gray'),origin='lower',aspect='auto',zorder=0)
    
    # Plot the upper and lower limits on the image.
    ax[0].plot([th.min(),th.max()],[scanlow,scanlow],color='red',linestyle='dashed',linewidth=0.75)
    ax[0].plot([th.min(),th.max()],[scanhigh,scanhigh],color='red',linestyle='dashed',linewidth=0.75)
    ax[0].set_xlim([th.min(),th.max()])
    ax[0].set_ylim([r.min(),r.max()])
    ax[0].set_xticks([])
    ax[0].set_ylabel(r"r (pixels)")
    #plt.axis('auto')

    # Calculate the total intensity along a radial line between the limits given above as a function of phase angle.
    intens_phase = azimuthalScan(newimage,pmin=pmin,pmax=pmax)
    
    # Find all peaks of the intensity more than cols/6 points away from each other and with amplitude greater than 0.
    hipeaks,_ = find_peaks(intens_phase[:,1],height=(0,None),distance=cols/6)
    lopeaks,_ = find_peaks(-intens_phase[:,1],height=(None,0),distance=cols/6)
    
    print("\nPeaks are determined using scipy.signal.find_peaks, and may correspond to star spikes.\nInspect image (Fig. 1) to make sure peak locations and values make sense.")
    
    # Estimate the contrast based on the straight max and min of the intensity data.
    textbox = "$I_{max}$ = %.3f\n$I_{min}$ = %.3f\n$I_{max}/I_{min}$ = %.3f" % (intens_phase[:,1].max(),intens_phase[:,1].min(),intens_phase[:,1].max()/intens_phase[:,1].min())
    
    # Print out the peaks to the terminal, so the user can make more accurate measurements of the contrast if they wish.
    print("\n=========Intensity vs. Phase (Fig. 1)=========")
    print("\nPeaks:")
    print("Phase\tIntensity")
    for peak in hipeaks:
        print("%.3f\t%.3f" % (intens_phase[:,0][peak],intens_phase[:,1][peak]))
    # End loop
    print("\nValleys:")
    print("Phase\tIntensity")
    for peak in lopeaks:
        print("%.3f\t%.3f" % (intens_phase[:,0][peak],intens_phase[:,1][peak]))
    # End loop
    
    # Print out that est. contrast information to the terminal, along with the phase where the max and min intensity are found.
    print("\nImax = %.3f\nImin = %.3f\nImax/Imin = %.3f" % (intens_phase[:,1].max(),intens_phase[:,1].min(),intens_phase[:,1].max()/intens_phase[:,1].min()))
    print("\nPhase of max = %.2f\nPhase of min = %.3f" % (intens_phase[:,0][np.argmax(intens_phase[:,1])],intens_phase[:,0][np.argmin(intens_phase[:,1])]))

    # Plot the intensity between the limits as a function of phase.
    ax[1].scatter(intens_phase[:,0],intens_phase[:,1],marker='.',color='black',s=9)
    ax[1].set_ylim([intens_phase[:,1].min()-10,intens_phase[:,1].max()+10])
    
    # Mark the peaks for reference.
    ax[1].plot(intens_phase[:,0][hipeaks],intens_phase[:,1][hipeaks],marker='^',color='orange',linestyle='None',markersize=6)
    ax[1].plot(intens_phase[:,0][lopeaks],intens_phase[:,1][lopeaks],marker='v',color='cyan',linestyle='None',markersize=6)
    
    ax[1].set_xlim([-np.pi,np.pi])
    ax[1].set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax[1].text(0.05,0.95,textbox,transform=ax[1].transAxes,verticalalignment='top',fontsize=10)
    ax[1].set_xlabel(r"$\theta$ (radians)")
    ax[1].set_ylabel("I (image units)")

    # Compute the (estimated, again using straight max and min) contrast as a function of radius.
    # The number of pixels we sum the flux for is 8% of the distance from pmin to pmax. That choice is
    # pretty arbitrary; it just results in a relatively smooth variation of contrast in r for most galaxies.
    n = (pmax-pmin)/12
    cvr = contrastAsRadius(newimage,r.max(),n,pmin,pmax)
    
    # Find the peaks of the contrast.
    spacing=len(cvr[:,0])/4
    cpeaks,_= find_peaks(cvr[:,1],height=(0,None),distance=spacing)
    cvalleys,_ = find_peaks(-cvr[:,1],height=(None,0),distance=spacing)
    
    # Compute some basic statistics about the contrast.
    textbox = "mean = %.2f\n$\sigma$ = %.2f" % (np.nanmean(cvr[:,1]),np.nanstd(cvr[:,1]))

    # Print that information to the terminal.
    print("\n=========Contrast vs. Radius (Fig. 2)=========")
    print("\nPeaks:")
    print("r\tContrast")
    for peak in cpeaks:
        print("%.3f\t%.3f" % (cvr[:,0][peak],cvr[:,1][peak]))
    # End loop
    print("\nValleys:")
    print("r\tContrast")
    for peak in cvalleys:
        print("%.3f\t%.3f" % (cvr[:,0][peak],cvr[:,1][peak]))
    # End loop

    print("\nmean = %.3f\nstdev = %.3f\n" % (np.nanmean(cvr[:,1]),np.nanstd(cvr[:,1])))

    fig2,ax2 = plt.subplots(figsize=(10,6),dpi=120)
    ax2.scatter(cvr[:,0],cvr[:,1],marker='.',color='black',s=9)
    ax2.plot(cvr[:,0][cpeaks],cvr[:,1][cpeaks],marker='^',color='orange',linestyle='None',markersize=6)
    ax2.plot(cvr[:,0][cvalleys],cvr[:,1][cvalleys],marker='v',color='cyan',linestyle='None',markersize=6)
    
    ax2.set_xlim([cvr[:,0].min(),cvr[:,0].max()])
    ax2.set_ylim([cvr[:,1].min()-0.2,cvr[:,1].max()+0.2])
    ax2.text(0.05,0.95,textbox,transform=ax2.transAxes,verticalalignment='top',fontsize=10)
    ax2.set_xlabel(r"r (pixels)")
    ax2.set_ylabel(r"$I_{max}/I_{min}$")

    if (len(sys.argv) == 3):
        hdu = fits.PrimaryHDU(newimage)
        hdu.writeto(sys.argv[2])
    # End if

    plt.show()
# End definition

if __name__ == "__main__":
    main()
# End if
# End of file
