import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


class Ellipse_fitter():
    def __init__(self):
        print("Ellipse_fitter created")
    
    
    #### FUNCTIONS TO BE CALLED ####
    
    ## function to fit a 2D ellipse to the data with curve_fit() from scipy.optimize
    def fit_ellipse(self, data, pix_size, p0 = None, print_results = True, plot_model = True):
        ## prepare the data for input into the fitting function
        xs, ys, zs = self.__prepare_data_for_fitting(data)
    
        ## make sure an initial guess is made for the fitting
        if(p0 is None):
            p0 = self.__create_init_guess(data.shape, np.nanmax(zs))
            
        ## perform the fitting
        popt, pcov = curve_fit(self.__ellipse_gauss_ring, (ys, xs), zs, p0 = p0)
        
        ## print the results if requested
        if(print_results):
            self.__print_popt_vals(popt, pix_size)
    
        ## plot the elliptical ring model if requested
        if(plot_model):
            self.__plot_ellipse_model(np.indices(data.shape), popt)
            
        return popt, pcov


    ## plot the data with the model of the elliptical ring
    def plot_data_model(self, data, popt, levs, wids):
        ## generate the model
        fit_result = self.__ellipse_gauss_ring(np.indices(data.shape), *popt)
        
        ## create the figure
        fig, ax = plt.subplots()
        ax1 = fig.add_subplot(111)
        
        ## add the image
        im = ax1.imshow(data, origin='lower', vmin=0.)
        
        ## add the contours
        ctrs = ax1.contour(fit_result, levels = levs, linewidths = wids, colors = 'k', origin = 'lower')
        
        ## create and plot the colorbar
        cbar = fig.colorbar(im)
        cbar.set_label('T$_{mb}$ (K)', labelpad=15.,rotation=270.)
    
        ## finalize the plot and show()
        ax.axis('off')
        plt.show()
    
    
        
    #####  SUPPORT FUNCTIONS THAT SHOULD NOT BE CALLED FROM OUTSIDE THIS FILE #####
    
    ## Definition of a 2D elliptical intensity distribution where the elliptical ring has a gaussian intensity distribution
    ## - with a random rotation angle (theta)
    ## - centered on (cx, cy), with major and minor axes (a, b)
    ## - with a width (w) for the ellipse and peak intensity (Tpeak)
    ## returns the predicted intensity for each pixel X (tupple containing (y, x))
    def __ellipse_gauss_ring(self, X, cx, cy, a, b, w, Tpeak, theta):
        ## read pixel coordinates input
        y_pix, x_pix = X
    
        ## perform a translation followed by a rotation of the pixel coordinates to the reference frame of the ellipse
        new_x = (x_pix - cx)*np.cos(theta) - (y_pix - cy)*np.sin(theta)
        new_y = (x_pix - cx)*np.sin(theta) + (y_pix - cy)*np.cos(theta)
    
        ## distance between pixel and center of the ellipse
        d1 = np.sqrt(new_x**2 + new_y**2)
    
        ## Find the intersection with the ellipse
        m = new_y / (new_x + 0.01) ## + 0.01 to avoid division by zero
        cte = a**(-2) + (m/b)**2
    
        ## get x location of the intersection with the ellipse
        ## The -2* includes a conditional statement to handle the situation when new_x < 0
        x_el = np.sqrt(1./cte) - 2.*(new_x < 0)*np.sqrt(1./cte)
    
        ## avoid small effects leading to errors in the square root calculation
        x_el[np.abs(x_el) >= a] = a
    
        ## get y location of the intersection with the ellipse
        ## The -2* includes a conditional statement to handle the situation when new_y < 0
        y_el = b*np.sqrt(1. - (x_el/a)**2) - 2.*(new_y < 0)*b*np.sqrt(1. - (x_el/a)**2)
    
        ## distance to the intersection with the ellipse
        d2 = np.sqrt((x_el)**2 + (y_el)**2)
    
        ## distance between the pixel point and the location on the ellipse
        d = np.abs(d1-d2)
        
        ## return the intensity at the location
        Treturn = Tpeak * np.exp(-(d/w)**2)
        
        return Treturn 


    ## function to prepare the data to be fitted
    def __prepare_data_for_fitting(self, data):
        ## create a numpy 2D array with the indices of the data
        inds = np.indices(data.shape)
        
        ## ravel the indices and intensity data for fitting
        xs_orig = inds[1].ravel() ## because inds: [[ys], [xs]]
        ys_orig = inds[0].ravel()
        zs = data.ravel()
        
        ## remove all nans from the data
        xs = xs_orig[~np.isnan(zs)]
        ys = ys_orig[~np.isnan(zs)]
        zs = zs[~np.isnan(zs)]
        
        return (xs, ys, zs)


    ## creates an initial guess used for curve_fit using the information on the shape of the provided 2D data 
    def __create_init_guess(self, data_shape, max_intensity):
        return [0.5*data_shape[1], 0.5*data_shape[0], 0.35*data_shape[1], 0.35*data_shape[0], 0.05*0.5*(data_shape[1] + data_shape[0]), 0.5*max_intensity, 0.]
    
    
    ## print the values of curve_fit on the 2D elliptical ring
    def __print_popt_vals(self, popt, pix_size):
        ## print the center of the ellipse
        print("The central position of the ellipse is: ({cx},{cy})".format(cx = round(popt[0], 1), cy = round(popt[1], 1)))
        
        ## print the size of the major and minor axis
        print("The major and minor axis of the ellipse are: {a} pc & {b} pc)".format(a = round(popt[2]*pix_size, 1), b = round(popt[3]*pix_size, 1)))
        
        ## print the width of the elliptical ring
        print("The width of the elliptical ring is: {w} pc".format(w = round(popt[4]*pix_size, 1)))
        
        ## print the peak intenstiy at the crest of the elliptical ring
        print("The peak intensity at the crest of the elliptical ring is: {T} K".format(T = round(popt[5], 1)))
        
        ## print the rotation angle of the ellipse
        print(u"The tilt angle of the ellipse is: {th}\N{DEGREE SIGN}".format(th = round(popt[6]*180./np.pi, 1)))
        
    
    ## plot the model of the elliptical ring
    def __plot_ellipse_model(self, inds, popt):
        ## generate the model
        fit_result = self.__ellipse_gauss_ring(inds, *popt)
        
        ## create figure
        fig, ax = plt.subplots()
        ax1 = fig.add_subplot(111)
        
        ## add the image
        im = ax1.imshow(fit_result, origin='lower', vmin=0.)
        
        ## create and plot the colorbar
        cbar = fig.colorbar(im)
        cbar.set_label('T$_{mb}$ (K)', labelpad=15.,rotation=270.)
    
        ## finalize the plot and show()
        ax.axis('off')
        plt.show()