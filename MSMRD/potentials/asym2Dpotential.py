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
            gradx = -(2*(x-mx)/(2*sigx**2))*np.exp(-(x - mx)**2/(2*sigx**2)-(y - my)**2/(2*sigy**2))
            gradx = gradx/(2*np.pi*sigx*sigy)
            grady = -(2*(y-my)/(2*sigy**2))*np.exp(-(x - mx)**2/(2*sigx**2)-(y - my)**2/(2*sigy**2))
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

class asym2DpotentialAlt(object):
    def __init__(self, scalefactor=1.0, minima=None, widths=None, depths=None):
        if minima == None:
            minima = np.array([[0.0,0.0], [1.0,0.0] , [1.1, 1.0], [-0.1,0.9], [-1.3,0.8], \
                       [-1.0,-0.2], [-0.6,-1.0], [0.9,-0.8], [0.2,-1.5]])
        if widths == None:
            widths = np.array([[ 0.24782041,  0.1607923 ],[ 0.10879743,  0.12047222], [ 0.22322456,  0.2247435 ], \
                   [ 0.13270383,  0.27585142], [ 0.2114683 ,  0.27406891], [ 0.15899199,  0.22264597], \
                [ 0.20849387,  0.20086717], [ 0.13200173,  0.22780353], [ 0.19862744,  0.25441848]])
        if depths == None:
            depths = np.array([ 1.82629865,  2.12330603,  1.94039645,  1.82848899,  1.82522799, 2.08327185,  2.10279624,  1.87371785,  1.85471121])
        self.minima = minima
        self.widths = widths
        self.depths = depths * scalefactor
        self.scalefactor = scalefactor

    # Define potential as sum of inverted Gaussians from minimas and variances
    def potential(self,r):
        x, y = r
        output = 0
        for i in range(len(self.minima)):
            gauss = np.exp(-(x - self.minima[i,0])**2/(2*self.widths[i,0]**2)-(y - self.minima[i,1])**2/(2*self.widths[i,1]**2))
            gauss = gauss * self.depths[i]
            output = output - gauss
        return output

    # Calculate minus gradient of the potential
    def force(self,r):
        x, y = r
        outx, outy = [0,0]
        for i in range(len(self.minima)):
            gradx = np.exp(-(x - self.minima[i,0])**2/(2*self.widths[i,0]**2)-(y - self.minima[i,1])**2/(2*self.widths[i,1]**2))
            gradx *= self.depths[i]*(x-self.minima[i,0])/(self.widths[i,0]**2)
            grady = np.exp(-(x - self.minima[i,0])**2/(2*self.widths[i,0]**2)-(y - self.minima[i,1])**2/(2*self.widths[i,1]**2))
            grady *= self.depths[i]*(x-self.minima[i,1])/(self.widths[i,1]**2)
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

