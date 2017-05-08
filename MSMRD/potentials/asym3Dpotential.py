import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets



# Construct asymmetric 2D potential from Gaussian, calculate gradients and plots.
class asym3Dpotential(object):
    def __init__(self, minima=None, sigmas=None, scalefactor=None):
        if minima == None:
            minima = [[-0.9,0.7,0.3] ,  [-0.1,0.9,0.7],  [0.8,0.8,-0.8],  \
                      [-1.0,-0.3,-0.4], [0.0,0.0,0.0],   [0.9,-0.1,-0.9], \
                      [-0.7,-1.0,-0.3], [0.0,-0.9,0.1],  [0.8,-0.2,0.8]]
            sigmas = [[0.3,0.3,0.3],    [0.35,0.35,0.35],  [0.4,0.3,0.3], \
                      [0.25,0.32,0.34], [0.4,0.28,0.4],    [0.4,0.3,0.3], \
                      [0.3,0.25,0.4],   [0.3,0.4,0.3],     [0.3,0.4,0.35]]
        if scalefactor == None:
            scalefactor = 0.7
        self.minima = minima
        self.sigmas = sigmas
        self.scalefactor = scalefactor

    # Define potential as sum of inverted Gaussians from minimas and variances
    def potential(self,r):
        x, y, z = r
        output = 0
        for i in range(len(self.minima)):
            mx, my, mz = self.minima[i]
            sigx, sigy, sigz = self.sigmas[i]
            gauss = np.exp(-(x - mx)**2/(2*sigx**2) - (y - my)**2/(2*sigy**2) - (z - mz)**2/(2*sigz**2))
            gauss = gauss/(pow(2*np.pi,3.0/2.0)*sigx*sigy*sigz)
            output = output - gauss
        return self.scalefactor*output

    # Calculate minus gradient of the potential
    def force(self,r):
        x, y, z = r
        outx, outy, outz = [0,0,0]
        for i in range(len(self.minima)):
            mx, my, mz = self.minima[i]
            sigx, sigy, sigz = self.sigmas[i]
            gradx = -(2*(x-mx)/(2*sigx))*np.exp(-(x - mx)**2/(2*sigx**2) - (y - my)**2/(2*sigy**2) - (z - mz)**2/(2*sigz**2))
            gradx = gradx/(pow(2*np.pi,3.0/2.0)*sigx*sigy*sigz)
            grady = -(2*(y-my)/(2*sigy))*np.exp(-(x - mx)**2/(2*sigx**2) - (y - my)**2/(2*sigy**2) - (z - mz)**2/(2*sigz**2))
            grady = grady/(pow(2*np.pi,3.0/2.0)*sigx*sigy*sigz)
            gradz = -(2*(z-mz)/(2*sigz))*np.exp(-(x - mx)**2/(2*sigx**2) - (y - my)**2/(2*sigy**2) - (z - mz)**2/(2*sigz**2))
            gradz = gradz/(pow(2*np.pi,3.0/2.0)*sigx*sigy*sigz)
            outx = outx - gradx
            outy = outy - grady
            outz = outz - gradz
        return -self.scalefactor*np.array([outx,outy,outz])

    # Calculate norm of gradient of the potential
    def gradnorm(self,r):
        outx, outy, outz = self.force(r)
        out = np.sqrt(outx*outx + outy*outy + outz*outz)
        return out
    
    # Calculate grid and potential values in the grid
    def potential_in_grid(self,xmin,xmax,ymin,ymax,zcut):
        x = np.arange(xmin,xmax,(xmax-xmin)/100.0)
        y = np.arange(ymin,ymax,(ymax-ymin)/100.0)
        xx, yy = np.meshgrid(x,y)
        zz = 0*xx + zcut
        val = self.potential([xx,yy,zz])
        return xx, yy, val
    
    # Calculate grid and gradient values in the grid
    def grad_in_grid(self,xmin,xmax,ymin,ymax,zcut):
        x = np.arange(xmin,xmax,(xmax-xmin)/100.0)
        y = np.arange(ymin,ymax,(ymax-ymin)/1000.0)
        xx, yy = np.meshgrid(x,y)
        zz = 0*xx + zcut
        val = self.gradnorm([xx,yy,zz])
        return xx, yy, val
        
    # Make contour plot of potential or gradient (numcontour = num contours)
    # grad = boolean to plot gradient or potential
    def plot_contour(self, zcut, numcontour, grad, fill):
        xmin, xmax, ymin, ymax = [-3,3,-3,3]
        # set default number of countours
        if numcontour == None:
            numcontour = 25
        # calculate and plot potential or gradient
        if grad == False:
            xx, yy, zz = self.potential_in_grid(xmin,xmax,ymin,ymax,zcut)
        elif grad == True:
            xx, yy, zz = self.grad_in_grid(xmin,xmax,ymin,ymax,zcut)
        if fill == True:
            levels1 = np.linspace(-10,0,150)
            if grad == True:
                levels1 = np.linspace(-6,6,150)
            plt.contourf(xx, yy, zz, levels = levels1)
            #plt.colorbar(format='%.2f')
            plt.clim(-3,0)
        if grad == True and fill == True:
            plt.clim(-6,6)
        contours = np.linspace(-10.0,0.5,numcontour)
        contplot = plt.contour(xx, yy, zz, levels=contours, colors='k')
        for c in contplot.collections:
            c.set_linestyle('solid')
            c.set_linewidths(0.7)
        plt.axes().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def plot_interact(self):
        mainplot = self.plot_contour
        interact(mainplot, zcut = widgets.FloatSlider(value=0.0,min=-2.0,max=2.0),
                   numcontour = widgets.IntSlider(value=25,min=5,max=100),
                   grad = widgets.Checkbox(value=False, description='Plot gradient'),
                   fill = widgets.Checkbox(value=False, description='Fill'))
