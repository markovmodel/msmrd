import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Construct asymmetric 2D potential from Gaussian, calculate gradients and plots.
class asym2Dpotential(object):
    def __init__(self, minima=None, sigmas=None):
        if minima == None:
            minima = [[0.0,0.0], [1.0,0.0] , [1.1, 1.0], [-0.1,0.9], [-1.3,0.8], \
                       [-1.0,-0.2], [-0.6,-1.0], [0.9,-0.8], [0.2,-1.5]]
            sigmas = [[0.26,0.26],[0.35,0.35],  [0.4,0.3],  [0.4,0.23], [0.25,0.35], \
                       [0.25,0.32],   [0.4,0.28], [0.4,0.3], [0.21,0.45]]
        self.minima = minima
        self.sigmas = sigmas

    # Define potential as sum of inverted Gaussians from minimas and variances
    def potential(self,x,y):
        output = 0
        for i in range(len(self.minima)):
            mx = self.minima[i][0]
            my = self.minima[i][1]
            sigx = self.sigmas[i][0]
            sigy = self.sigmas[i][1]
            gauss = np.exp(-(x - mx)**2/(2*sigx**2)-(y - my)**2/(2*sigy**2))
            gauss = gauss/(2*np.pi*sigx*sigy)
            output = output - gauss
        return output

    # Calculate gradient of the potential
    def gradpot(self,x,y):
        outx = 0
        outy = 0
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
        return [outx,outy]

    # Calculate norm of gradient of the potential
    def gradnorm(self,x,y):
        outx, outy = self.gradpot(x,y)
        out = np.sqrt(outx*outx + outy*outy)
        return out
    
    # Calculate grid and potential values in the grid
    def potential_in_grid(self,xmin,xmax,ymin,ymax):
        x = np.arange(xmin,xmax,(xmax-xmin)/100.0)
        y = np.arange(ymin,ymax,(ymax-ymin)/100.0)
        xx, yy = np.meshgrid(x,y)
        zz = self.potential(xx,yy)
        return xx, yy, zz
    
    # Calculate grid and gradient values in the grid
    def grad_in_grid(self,xmin,xmax,ymin,ymax):
        x = np.arange(xmin,xmax,(xmax-xmin)/100.0)
        y = np.arange(ymin,ymax,(ymax-ymin)/1000.0)
        xx, yy = np.meshgrid(x,y)
        zz = self.gradnorm(xx,yy)
        return xx, yy, zz
        
    # Make contour plot of potential or gradient (numcontour = num contours)
    # grad = boolean to plot gradient or potential
    def plot_contour(self,xmin,xmax,ymin,ymax,zmin,zmax,numcontour,grad):
        if grad == True:
            xx, yy, zz = self.grad_in_grid(xmin,xmax,ymin,ymax)
        else:
            xx, yy, zz = self.potential_in_grid(xmin,xmax,ymin,ymax)
        plt.contour(xx, yy, zz, numcontour)
        plt.axes().set_aspect('equal')
    
    # Make 3d plot of potential, res lower =  better resolution
    # grad = boolean to plot gradient or potential
    def plot_3d(self,xmin,xmax,ymin,ymax,zmin,zmax,res,grad):
        if grad == True:
            xx, yy, zz = self.grad_in_grid(xmin,xmax,ymin,ymax)
        else:
            xx, yy, zz = self.potential_in_grid(xmin,xmax,ymin,ymax)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(zmin, zmax)
        p = ax.plot_surface(xx, yy, zz, rstride=res, cstride=res, linewidth=0)

        