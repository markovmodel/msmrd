import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Construct asymmetric 2D potential from Gaussian, calculate gradients and plots.
class asym2Dpotential(object):
    def __init__(self, minima=None, sigmas=None, scalefactor=None):
        if minima == None:
            minima = [[0.0,0.0], [1.0,0.0] , [1.1, 1.0], [-0.1,0.9], [-1.3,0.8], \
                       [-1.0,-0.2], [-0.6,-1.0], [0.9,-0.8], [0.2,-1.5]]
            sigmas = [[0.3,0.3],[0.35,0.35],  [0.4,0.3],  [0.4,0.23], [0.25,0.35], \
                       [0.25,0.32],   [0.4,0.28], [0.4,0.3], [0.21,0.45]]
	if scalefactor == None:
		scalefactor = 0.7
        self.minima = minima
        self.sigmas = sigmas
	self.scalefactor = scalefactor

    # Define potential as sum of inverted Gaussians from minimas and variances
    def potential(self,r):
        x, y = r
        output = 0
        for i in range(len(self.minima)):
            mx = self.minima[i][0]
            my = self.minima[i][1]
            sigx = self.sigmas[i][0]
            sigy = self.sigmas[i][1]
            gauss = np.exp(-(x - mx)**2/(2*sigx**2)-(y - my)**2/(2*sigy**2))
            gauss = gauss/(2*np.pi*sigx*sigy)
            output = output - gauss
        return self.scalefactor*output

    # Calculate minus gradient of the potential
    def force(self,r):
        x, y = r
        outx, outy = [0,0]
        for i in range(len(self.minima)):
            mx = self.minima[i][0]
            my = self.minima[i][1]
            sigx = self.sigmas[i][0]
            sigy = self.sigmas[i][1]
            gradx = -(2*(x-mx)/(2*sigx))*np.exp(-(x - mx)**2/(2*sigx**2)-(y - my)**2/(2*sigy**2))
            gradx = gradx/(2*np.pi*sigx*sigy)
            grady = -(2*(y-my)/(2*sigy))*np.exp(-(x - mx)**2/(2*sigx**2)-(y - my)**2/(2*sigy**2))
            grady = grady/(2*np.pi*sigx*sigy)
            outx = outx - gradx
            outy = outy - grady
        return -self.scalefactor*np.array([outx,outy])

    # Calculate norm of gradient of the potential
    def gradnorm(self,r):
        outx, outy = self.force(r)
        out = np.sqrt(outx*outx + outy*outy)
        return out
    
    # Calculate grid and potential values in the grid
    def potential_in_grid(self,xmin,xmax,ymin,ymax):
        x = np.arange(xmin,xmax,(xmax-xmin)/100.0)
        y = np.arange(ymin,ymax,(ymax-ymin)/100.0)
        xx, yy = np.meshgrid(x,y)
        zz = self.potential([xx,yy])
        return xx, yy, zz
    
    # Calculate grid and gradient values in the grid
    def grad_in_grid(self,xmin,xmax,ymin,ymax):
        x = np.arange(xmin,xmax,(xmax-xmin)/100.0)
        y = np.arange(ymin,ymax,(ymax-ymin)/1000.0)
        xx, yy = np.meshgrid(x,y)
        zz = self.gradnorm([xx,yy])
        return xx, yy, zz
        
    # Make contour plot of potential or gradient (numcontour = num contours)
    # grad = boolean to plot gradient or potential
    def plot_contour(self, bounds=None, numcontour=None, grad=None):
        # set default values for bounds and number of countours
        if bounds == None:
            if grad == None:
                bounds = [-3,3,-3,3,-3,0]
            else:
                bounds = [-3,3,-3,3,0,3]
        xmin = bounds[0]; xmax = bounds[1]
        ymin = bounds[2]; ymax = bounds[3]
        zmin = bounds[4]; zmax = bounds[5]
        if numcontour == None:
            numcontour = 25
        # calculate and plot potential or gradient
        if grad == None:
            xx, yy, zz = self.potential_in_grid(xmin,xmax,ymin,ymax)
        elif grad == True:
            xx, yy, zz = self.grad_in_grid(xmin,xmax,ymin,ymax)         
        plt.contour(xx, yy, zz, numcontour)
        plt.axes().set_aspect('equal')
    
    # Make 3d plot of potential, res lower =  better resolution
    # grad = boolean to plot gradient or potential
    def plot_3d(self,bounds=None, res=None, grad=None):
        # set default values for bounds and resolution
        if bounds == None:
            if grad == None:
                bounds = [-3,3,-3,3,-3,0]
            else:
                bounds = [-3,3,-3,3,0,3]
        xmin = bounds[0]; xmax = bounds[1]
        ymin = bounds[2]; ymax = bounds[3]
        zmin = bounds[4]; zmax = bounds[5]
        if res == None:
            res = 2
        # calculate and plot potential or gradient
        if grad == None:
            xx, yy, zz = self.potential_in_grid(xmin,xmax,ymin,ymax)
        elif grad == True:
            xx, yy, zz = self.grad_in_grid(xmin,xmax,ymin,ymax)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(zmin, zmax)
        p = ax.plot_surface(xx, yy, zz, rstride=res, cstride=res, linewidth=0)

        
