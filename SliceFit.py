'''
This code contains the fitting and visualization tools used in the Master Thesis: "Characterization of nuclear activity for galaxies in different large-scale environments in terms of morphology, colour, and specific star formation rate" by Daniel Ariza Quintana.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import imageio

from scipy import stats
from glob import glob
from scipy.optimize import curve_fit, leastsq
from tqdm import tqdm


def Parametros(y,z, r=5, sig: list=[0.25,0.25]):
    '''
    Finds initial parameters to gide the fit.
    It takes the highest point of the histogram. Then centered in this max.
    looks for the next maximum in a neighbourhood of distance r. If it is not
    found then it is labeled as the other peak.
    (y)   Quantity of the histogram
    (z)   Number of counts
    (r)   Radious of the environment around the maximum
    (sig) Initial value of sigma

    RETURN: Parameters for a double gaussian.
    '''
    # Initial values
    c1 =  0 # Hight of peak 1
    c2 =  0 # Hight of peak 2
    mu2 = 0 # Median for the second peak
    
    # Getting hights from highest to lower
    max = np.sort(z)[::-1]
    c1  = max[0] # Assigning the highest point to the first peak
    
    # We run through z looking for the other peak, creating an interval
    # around c1 that grows every 2 iterations.
    ind1 = np.where(z==c1)[0][0] # c1 index
    mu1  = y[ind1]  # mu1 is assigned as the quantity of c1
    for i,hight in enumerate(max):
        if i%2==0: r+=1                 # Update environment
        loc = np.where(z==hight)[0][0]  # Find the position  
        if (loc not in range(ind1-r+2,ind1+r+1+2)): # Checking if peak in env.
            mu2 = y[loc]  # Peak value in y
            c2  = z[loc]  # Peak value in z
            break         # Exit loop when found
    return [c1, mu1, sig[0], c2, mu2, sig[1]]

def double_gaussian(x, params):
    '''
    Double Gaussian Function
    (x)      Entry array values
    (params) Double gaussian parameters
    '''
    (c1, mu1, sigma1, c2, mu2, sigma2) = params
    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res

def double_gaussian_fit(params, x, y):
    '''
    Fitting function for double gaussian
    (params) Double gaussian parameters
    (x)      Entry array values
    (y)      Data to fit

    RETURN: Difference between fit and data, to be feeded to least squares.
    '''
    fit = double_gaussian(x, params)
    return (fit - y)

def DG_fit(y: np.array, z: np.array) -> list:
    '''
    Double Gaussian Fit of 2 given arrays
    (Y)   Y-axis data
    (Z)   Z-axis data
    (Sg)  Sigma values for the fit

    RETURN: Returns the fit parameters for the double gaussian
    '''
    params = Parametros(y,z)  #[c1, mu1, sig1, c2, mu2, sig2]
    
    # Least squares fit. Starting values found by inspection.
    fit = leastsq(double_gaussian_fit, params, args=(y, z))

    return fit

def DG_fit_sig(y: np.array, z: np.array, Sg: list, r: int=5) -> list:
    '''
    Double Gaussian Fit of 2 given arrays
    (Y)   Y-axis data
    (Z)   Z-axis data
    (Sg)  Sigma values for the fit

    RETURN: Returns the fit parameters for the double gaussian
    '''
    params = Parametros(y,z, sig=Sg, r=r)  #[c1, mu1, sig1, c2, mu2, sig2]
    
    # Least squares fit. Starting values found by inspection.
    fit = leastsq(double_gaussian_fit, params, args=(y, z))

    return fit

def contour_params_array(a, b, Nbins = 50) -> list:
    '''
    Creates the proper data set-up for a contour plot from 2 arrays.
    (a)     First  array to use
    (b)     Second array to use
    (Nbins) Number of bins

    RETURN: X, Y, Z arrays for a contour plot.
    '''
    # Get rid of nans for both
    Nancond = np.isnan(a) | np.isnan(b)
    # Define the arrays
    x = a[~Nancond]
    y = b[~Nancond]
    
    # Create the contour
    x_c = np.linspace(np.min(x), np.max(x), Nbins)
    y_c = np.linspace(np.min(y), np.max(y), Nbins)
    X, Y    = np.meshgrid(x_c,y_c)
    Z, xbin, ybin = np.histogram2d(x, y, bins=Nbins)

    return X, Y, Z.transpose()

# Se podría acelerar sacando la parte de GRID y pasándole cada slice
def SliceFit(Dx, Dy, pos, plot=True,  hw : float=0.1, cmin: int=10, nc: int=10, Nbins: int=50, save=None, exp=None) -> list:
    '''
    - Funtion that takes a slice of the contour a makes a gaussian fit of it and a plot for M*-sSFR.
    (Dx)     X axis data of contour
    (Dy)     Y axis data of contour
    (pos)    Position on the X axis where to perform the cut. Must be the index of mashgrid (Arreglar)
    (plot)   Weather to plot the fit or not
    (hw)     Histogram width
    (cmin)   Minimum contour to show
    (nc)     Number of contour lines
    (Nbins)  Number of bins for the histogram
    (exp)    Exponent for the exponential of the counts, if specified

    - RETURN: An image of the fit and the parámeters of the fit (c1, mu1, sigma1, c2, mu2, sigma2).
    '''

    ## GRID ##
    X, Y, Z = contour_params_array(Dx,Dy, Nbins)   # Creating 2D histogram

    if exp: Z = np.exp(exp*Z)  # Redefinimos Z con la exponencial

    ## AJUSTE ## 
    # Datos para el ajuste
    y = Y[:,pos].copy()
    z = Z[:,pos].copy()
    fit = DG_fit(y,z)   # ajuste
    
    ### PLOT ###
    if plot or save:
        fig, [ax, axs] = plt.subplots(1,2, figsize=(13,5))
        col = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        ## Contour ##
        CL = np.linspace(cmin,np.max(Z.flatten()),nc) # Frecuancias donde pintar los contornos
        im = ax.contour(X, Y, Z, CL)      # Contour        
        ax.axvline(X[0,pos], c='b')  # Línea donde se secciona
        ax.set_xlabel('logM*') 
        ax.set_ylabel('log(sSFR)')
        ax.set_title('M*-sSFR')
    
        # Ajuste de los mínimos de la distribución
        '''
        ax.scatter(Valley[:,0],Valley[:,1], c='r', marker='s', s=8)
        x = np.linspace(8, 12, 50)
        ax.plot(x, f(x, a,b) , label=f'{a:.3f}x + {b:.3f}', c='darkred', lw=2)
        '''
        
        ## Histograma ##
        axs.bar(Y[:,pos], Z[:,pos], hw,color='darkblue', edgecolor='k', alpha=0.6)  # Histograma de la posición seleccionada
        axs.scatter(y, z, edgecolors='b' , s=30, facecolors='w')
        axs.plot(y, double_gaussian(y, fit[0]), c='r' )
        axs.set_xlabel('log(sSFR)')
        axs.set_ylabel('frec')
        axs.set_title(f'Distribución M*= {X[0,pos]:.3f} | sig1={fit[0][2]:.3f} sig2={fit[0][5]:.3f}')
        axs.set_ylim(-1,None)
        
        if plot: plt.show()
        
        if save: 
            try:
                plt.savefig(f'../Graph/{save}/ConFit_{pos}_{X[0,pos]:.3f}.png')
            except:
                os.mkdir(f'../Graph/{save}')
                plt.savefig(f'../Graph/{save}/ConFit_{pos}_{X[0,pos]:.3f}.png')
        plt.close()   

    return fit[0]

def plot_cont(x, y, Nbins: int=50, save=None, a: float=1, nc: int=10, cmin: int=100, scat=np.array(None), linear=np.array(None), err=None) -> None:
    '''
    Function that does contour plot for two given series.
    (x)     First column to plot
    (y)     Second column to plot
    (save)  ['<file_name>.png'] Wheather to save the image or not
    (a)     Opacity
    (nc)    Number of contour lines
    (cmin)  Number to star the contour lines from
    (scat)  Tuple of arrays to scatter plot like scat = (x,y)
    (linear)Tuple of parameters of linear regresion; linear=(m,n)
    (err)   Error of the y coord for the linear regresion
    
    RETURN: None. Shows image, saves it as png.
    '''

    ## PLOT ##
    fig, ax = plt.subplots( figsize=(10,6), layout='tight')

    # Creamos el grid y el histograma
    X, Y, Z = contour_params_array(x,y, Nbins)
    CL = np.linspace(cmin,np.max(Z.flatten()),nc)
    CL.sort()
    im = ax.contour(X, Y, Z, CL)
    ax.set_xlabel('$log(M_s)$ $[M\odot]$')
    ax.set_ylabel('$log(sSFR)$ $[yr^{-1}]$')
    ax.plot
    
    cb = plt.colorbar(im, ax=ax)
    #cb.set_label('Contour lines')

    # Scatter de puntos
    if scat.any():
        plt.scatter(scat[:,0],scat[:,1], s=8, marker='s', c='r')
    if linear.any():
        x = X[0]
        plt.plot(x, linear[0]*x+linear[1], color='brown', label=f'{linear[0]:.3f}x {linear[1]:.3f}')
        if err: ax.fill_between(x, linear[0]*x+linear[1] + err, linear[0]*x+linear[1] - err,  color='r', alpha=0.2)
        plt.legend()
    
        
    
    # Lines
    #l = np.linspace(8,13)
    #ax.plot(l, Lac(l), 'k--', label='Lacerna')
    #ax.legend()
    if save: plt.savefig(f'../Graph/Cont_{nc}_{save}')
    plt.show()
    plt.close()

def Contour_Fit_sig(Dx, Dy, Nbins=50, r: int=3, pass_it=False, exp=None) -> list:
    '''
    Funtion that takes a contour and makes a daouble gaussian fit for every slice.
    (Dx)      X axis data of contour
    (Dy)      Y axis data of contour
    (Nbins)   Number of bins for the histogram
    (r)       Radious of search for the initial parameters
    (pass_it) If True, passes previous sigma calculated as guide for the next fit
    (exp)     Exponent for the exponential of the counts, if specified

    RETURN: List of fitted parameters of double gaussian (c1, mu1, sigma1, c2, mu2, sigma2).
    '''
    Fits = np.empty((Nbins,6)) # Array to store solutions
    
    # Generating contour
    X, Y, Z = contour_params_array(Dx,Dy, Nbins) # Creating 2D histogram

    if exp: Z = np.exp(exp*Z)                    # Redefinition of Z with exp
    
    # Fit for the first slice
    a   = Y[:,0].copy()
    b   = Z[:,0].copy()
    Sig = [0.03,0.03]
    Fits[0], res = DG_fit_sig(a,b, Sg=Sig, r=r)

    
    # Defining the conditions to pass on to the next sigma
    if pass_it: Sig = pass_sig(Fits[0], Sgg, (0,2), (0,1))
    
    # Loop over the rest of the slices
    for i in range(1,Nbins):
        a   = Y[:,i].copy()
        b   = Z[:,i].copy()
        Fits[i], res    = DG_fit_sig(a,b, Sg=Sig, r=r)
        if pass_it: Sig = pass_sig(Fits[i], Sig, (0,2), (0,1))
            
    return Fits

def plot_Contour_Fit(Dx, Dy, Fits: list, save: str, plot: bool=False, hw: float=0.12, cmin: int=50, nc: int=10, Nbins: int=50, exp=None) -> None:
    '''
    Funtion that takes a slice of the contour a makes a gaussian fit of it and a plot for M*-sSFR.
    (Dx)     X axis data of contour
    (Dy)     Y axis data of contour
    (Fits)   Parameters of the fit to plot with the histogram
    (save)   Name of the folder to save the images in
    (plot)   Weather to show the fit or not
    (hw)     Histogram width
    (cmin)   Minimum contour to show
    (nc)     Number of contour lines
    (Nbins)  Number of bins for the histogram
    (exp)    Exponent for the exponential of the counts, if specified

    RETURN: A folder with all the images of the fitted slices.
    '''

    # Generating contour
    X, Y, Z = contour_params_array(Dx,Dy, Nbins) # Creating 2D histogram

    if exp: Z = np.exp(exp*Z)                    # Redefinition of Z with exp

    for pos, fit in tqdm(enumerate(Fits)):
        fig, [ax, axs] = plt.subplots(1,2, figsize=(13,6), layout='tight')
        
        ## Contour ##
        CL = np.linspace(cmin,np.max(Z.flatten()),nc) # Contour lines
        im = ax.contour(X, Y, Z, CL)                      
        ax.axvline(X[0,pos], c='b')                   # Fitted slice line
        
        ax.set_xlabel(r'$ \log(M_\star/M_\odot)$', fontsize=15)
        ax.set_ylabel(r'$\log(sSFR/yr^{-1})$', fontsize=15)
        ax.set_title( r'sSFR-$M_\star$', fontsize=15)
        
        ## Histogram ##
        a   = Y[:,pos].copy()
        b   = Z[:,pos].copy()
        axs.bar(Y[:,pos], Z[:,pos], hw,color='darkblue', 
                edgecolor='k', alpha=0.6)         # Selected position histogram
        axs.scatter(a, b, edgecolors='b' , s=30, facecolors='w')
        axs.plot(a, double_gaussian(a, fit), c='r' )
        axs.set_xlabel(r'$\log(sSFR/yr^{-1})$', fontsize=15)
        axs.set_ylabel('Counts', fontsize=15)
        axs.set_title(r'Distribution $M_\star=$'\
                      + f'{X[0,pos]:.3f} | sig1={fit[2]:.3f} sig2={fit[5]:.3f}'
                      , fontsize=15)
        axs.set_ylim(-1,None)
        if plot: plt.show()
        
        if save: 
            try:
                plt.savefig(f'./{save}/ConFit_{pos:2}_{Nbins}_{X[0,pos]:.3f}.png')
            except:
                os.mkdir(f'./{save}')
                plt.savefig(f'./{save}/ConFit_{pos:2}_{Nbins}_{X[0,pos]:.3f}.png')
        plt.close()   


def Valley(Fits, X: np.array, Y: np.array, xmin: float, xmax: float, Nbins: int=50, dmu: float=4, z_val: float=1500, save=None, fit=False) -> np.array:
    '''
    Funciton that calculates the lowest point between the peaks of a contour plot.
    It only compute the mid point if the fit has two peaks and the difference is
    in sigma is less than a given dmu.
    It shows a plot with the found middle points to check.
    (X)     X axis data in the contour
    (Y)     Y axis data in the contour
    (xmin)  x minimum in the Y domain
    (xmax)  x maximum in the Y domain
    (Nbins)   Number of bins for the histogram
    (dmu)   Limit difference between mu1 and mu2: dmu = abs(mu1-mu2)
    (z_val) Z value to give to the minimums
    (save)  File name name to save the points with, .npy format
    (fit)   If true, performs a linear regresion on the points.

    RETURN: Array with the found minimums. Returns 3 coordinates for every point.
            If fit==True, then also returns the liear fit parameters and the correlation matrix.
    '''
    A, B, C = contour_params_array(X, Y, Nbins)
    
    m_s = A[0]                        # X values to use
    x   = np.linspace(xmin,xmax,100)  # Defining the values for the gaussian

    fig, [ax, axs] = plt.subplots(1,2, figsize=(12,5), layout='tight')
    ax.set_xlabel('Y')
    ax.set_ylabel('Counts')
    ax.set_title('Found minimums')
    
    # Iterative search algorithm
    valley = []
    for i in (range(Nbins)): # Loop over all the X values
        p  = Fits[i]               # Parameters
        fx = double_gaussian(x,p)  # Update function
        [c1, mu1, sigma1, c2, mu2, sigma2] = p
        
        if (x[0]<mu1<x[-1])&(x[0]<mu2<x[-1])&(abs(mu1-mu2)<dmu): # Double peak condition
            if mu1 < mu2: 
                cond = (mu1<x)&(x<mu2)
            else: 
                cond = (mu2<x)&(x<mu1)
            try: 
                fmin = np.min(fx[cond])           # Minimum between peaks
                locmin = np.where(fx==fmin)[0][0] # Min. point position
                ax.plot(x, fx)
                ax.scatter(x[locmin], fmin)
                valley.append([m_s[i],x[locmin],z_val]) # Saving point
            except: pass
    valley = np.array(valley)

    CL = np.linspace(50, np.max(C.flatten()), 15) # Contour lines
    axs.contour(A, B, C, CL)
    axs.scatter(valley[:,0], valley[:,1], s=8, marker='s', c='r')
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.set_title('Display in contour')
    plt.show()

    if save: np.save(f'Min_Fit_{save}.npy', valley)

    if fit:
        def f(x,a,b):
            return a*x+b
        
        x = valley[:,0] ; y = valley[:,1]
        line, corr = curve_fit(f,x,y)
        a, b = line
        line = np.array(line)

        plt.figure(figsize=(12,4))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Fit')
        l = np.linspace(np.min(x), np.max(x))
        plt.plot(l, a*l+b, label=f'{a:.3f}x + {b:.3f}')
        plt.scatter(x,y)
        plt.legend()
        
        return valley, line, corr

    return valley