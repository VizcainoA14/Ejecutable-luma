import sys
import requests
from bs4 import BeautifulSoup
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread, imshow
from skimage import data
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
import pywt
import cv2
from scipy.stats import entropy
from scipy.signal import convolve2d, find_peaks
from numba import njit, prange
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from SubiraBD import SubirDB
import subprocess




def load_image(url) -> np.ndarray:

    """
    Load an image.
    """
 

    try:
        image = io.imread(url)
    except Exception as e:
        logging.error(f"Error loading image from url: {url}. Error: {e}")
        image = None

    return image
    
    
def converttogray(image,ind):
	if ind==1:
		image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if ind==2:
		image_gray=rgb2gray(image)
	return image_gray
# Here use ind==1
ind=1
    
    
    


def compute_entropy_statistics(image: np.ndarray) -> float:
    """
    1. Compute the entropy of an image.
        The entropy is a statistical measure of randomness that can be used to characterize the texture of an input image.
    2. Compute the mean intensity of an image.
        The mean intensity is a measure that represents the average level of brightness of the image.
    3. Compute the standard deviation of the pixel intensity of an image.    
        The standard deviation is a measure that represents the dispersion of the pixel intensities in the image.
    4. Compute the skewness of an image.
        The skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. Skewness can be 
        positive or negative, or undefined, and it quantifies the extent and direction of skew (horizontal symmetry).
    5. Compute the kurtosis of an image.
        The kurtosis is a measure of the shape of the probability distribution of a real-valued random variable about its mean. Kurtosis can be 
        positive or negative, or undefined, and it quantifies how data is near from mean or not.
    6. Compute the uniformity of an image.
        The uniformity is a measure of the texture of an image. It measures the sum of the squares of the pixel intensities, normalized to be in the 
        range [0, 1]. A value close to 1 indicates an image with low variation in pixel intensities (like a completely black or white image), while 
        a value close to 0 indicates high variation in pixel intensities.
    7. Compute the relative smoothness of an image.
        The relative smoothness is a measure of variation in the pixel intensity levels.
    8. Compute the Taruma contrast of an image.
        The Taruma contrast is a measure derived from the image's standard deviation and kurtosis, giving insight into its contrast characteristics.
        
    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.
    Returns
    -------
    entropy : float
        Calculated entropy.
    mean-intensity : float
        Calculated mean intensity.
    standard_deviation : float
        Calculated standard deviation.
    skewness : float
        Calculated skewness of the image pixel intensity distribution.
    kurtosis : float
        Calculated kurtosis of the image pixel intensity distribution.
    uniformity : float
        Calculated uniformity of the image pixel intensity distribution.
    relative_smoothness : float
        Calculated relative smoothness.
    taruma_contrast : float
        Calculated Taruma contrast.
    """
    try:
    # Try to read the image
        #image = io.imread(url)
        #image_gray = rgb2gray(image)
        image_gray=converttogray(image,ind)
        histogram, _ = np.histogram(image_gray.ravel(), bins=256, range=[0,256])
        histogram_length = sum(histogram)
        samples_probability = [float(h) / histogram_length for h in histogram]
        # Compute the entropy of an image.    
        entropy=-sum([p * math.log(p, 2) for p in samples_probability if p != 0])
        # Compute the pixel intensities
        intensities = np.arange(256)
        # Compute mean intensity
        mean_intensity = np.sum(intensities * samples_probability)
        # Compute standard deviation
        standard_deviation = np.sqrt(np.sum(samples_probability * (intensities - mean_intensity)**2))
        # Compute skewness
        skewness = (1/standard_deviation**3)*(np.sum(samples_probability * (intensities - mean_intensity)**3))
        # Compute kurtosis
        kurtosis = (1/standard_deviation**4)*(np.sum(samples_probability * (intensities - mean_intensity)**4))
        # Compute uniformity
        uniformity =  sum([p ** 2 for p in samples_probability if p != 0])
        # Compute relative smoothness
        relative_smoothness = 1 - (1 / (1 + standard_deviation**2))
    except:
        entropy=np.nan
        mean_intensity=np.nan
        standard_deviation=np.nan
        skewness=np.nan
        kurtosis=np.nan
        relative_smoothness=np.nan
        uniformity=np.nan
    return entropy, mean_intensity, standard_deviation, skewness, kurtosis, relative_smoothness, uniformity

def compute_fractal_dimension(image: np.ndarray) -> float:
    """
    Computes the fractal dimension of a image.
    Computes the fractal dimension of a 2D grayscale image using the box-counting method.
    
    Parameters
    ----------
    image : np.ndarray
        2D input image data. Will be converted to grayscale if not already.
    threshold : float, optional
        Threshold to convert the grayscale image to binary, by default 0.5.
    Returns
    -------
    fractal_dimension : float
        Calculated fractal dimension of the image.
    """
    threshold=0.5
    try:
    # Try to read the image
        #image = io.imread(url)
        #image_gray = rgb2gray(image)
        image_gray=converttogray(image,ind)
        # Binarize the image using the given threshold
        image_gray = image_gray < threshold    
        # Define the boxcount function
        def boxcount(Z, k):
            S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0), np.arange(0, Z.shape[1], k), axis=1)
            return np.count_nonzero((0 < S) & (S < k*k))
        # Define the box sizes
        p = min(image_gray.shape)
        n = int(np.floor(np.log2(p)))
        sizes = 2**np.arange(n, 1, -1)
        # Count the number of boxes for each size
        counts = [boxcount(image_gray, size) for size in sizes]
        # Perform a linear fit (y = ax + b) to the sizes and counts
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        # Return the fractal dimension (-a)        
        fractal_dimension=-coeffs[0]
    except:
        fractal_dimension=np.nan
    return fractal_dimension

def compute_taruma_contrast(image: np.ndarray) -> float:
    """
    Compute the Taruma contrast of an image.
    The Taruma contrast is a measure derived from the image's standard deviation and kurtosis, giving insight into its contrast characteristics.

    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.
    Returns
    -------
    taruma_contrast : float
        Calculated Taruma contrast.
    """
    try:
        # Compute standard deviation
        sigma,kurtosis = compute_entropy_statistics(url)[2], compute_entropy_statistics(url)[4]
        # Compute Taruma contrast
        taruma_contrast = (sigma**2) / (kurtosis**(0.25))    
    except:
        taruma_contrast=np.nan
    return taruma_contrast

def compute_taruma_directionality(image: np.ndarray)-> float:
    """
    Compute the Taruma directionality of a given image.
    This function employs gradient filters to compute the horizontal and vertical gradient components of an image. Subsequently, it uses the 
    magnitude and direction of these gradients to construct a directionality histogram. From this histogram, dominant direction peaks are identified. 
    Finally, a Taruma directionality value is computed based on these peaks.

    Parameters:
    -----------
    image : np.ndarray
        An image in ndarray format. Can be colored or grayscale. If colored, it will automatically be converted to grayscale.
    plot : bool, optional
        If set to True, a directionality histogram with highlighted dominant peaks will be displayed. Default is False.
    Returns:
    --------
    float
        A Taruma directionality value ranging between 0 and 1, where values closer to 1 indicate high directionality and values closer to 0 indicate 
        low directionality.
    Example:
    --------
    >>> image = np.array(io.imread('path_to_image.jpg'))
    >>> directionality_value = compute_taruma_directionality(image, plot=True)
    Notes:
    ------
    - The function uses convolutions to compute gradients, so performance may vary based on the image size.
    - Ensure that the input image has an appropriate value range (e.g., between 0 and 255 for 8-bit images).
    """
    try:
    # Try to read the image
        #image = io.imread(url)
        #image_gray = rgb2gray(image)
        image = np.array(image, dtype='int64')
        image = np.mean(image, axis=-1) 

        h = image.shape[0]
        w = image.shape[1]

        # Kernels de convolución
        convH = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        convV = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])    
    
        # Calcula componentes horizontales y verticales usando convolución
        deltaH = convolve2d(image, convH, mode='same', boundary='symm')
        deltaV = convolve2d(image, convV, mode='same', boundary='symm')

        # Calcula la magnitud de gradiente
        deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0

        # Calcula el ángulo de dirección
        theta = np.arctan2(deltaV, deltaH) + np.pi / 2.0

        # Cuantización y histograma de dirección
        n = 90
        hist, edges = np.histogram(theta, bins=n, range=(0, np.pi), density=True)
    
        # Normalizar el histograma
        hist = hist / np.max(hist)

        # Calcular el umbral usando la media
        threshold = np.mean(hist)

        # Encuentra todos los picos que están por encima de la media
        all_peaks, properties = find_peaks(hist, height=threshold)

        # De esos picos, solo nos quedamos con los 5 más altos
        if len(all_peaks) > 5:
            sorted_peak_indices = np.argsort(properties['peak_heights'])[-5:]
            peaks = all_peaks[sorted_peak_indices]
            peak_properties = properties['peak_heights'][sorted_peak_indices]
        else:
            peaks = all_peaks
            peak_properties = properties['peak_heights']

        np_ = len(peaks)  # número de picos

        # Calcula F_dir según la formulación dada
        r = 1.0 / n  # factor de normalización
        phi = np.linspace(0, np.pi, n, endpoint=False) + np.pi / (2 * n)
        F_dir = 0
        for p in peaks:
            phi_p = phi[p]
            F_dir +=  np.sum((phi - phi_p) ** 2 * hist)
        taruma_directionality=1-(r*np_*F_dir)
    except:
        taruma_directionality=np.nan    
    return taruma_directionality

def compute_taruma_coarseness(image: np.ndarray) -> float:
    """
    Compute the Taruma coarseness of an image.
    Coarseness is a metric that captures the roughness or coarse texture in an image. Although it is not directly related to the Taruma direction, 
    you can use similar techniques to calculate it. Below, I provide you with an example of how to calculate the coarseness metric using the Wavelet 
    Transform in Python with the PyWavelets library.

    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.
    Returns
    -------
    taruma_contrast : float
        Calculated Taruma coarseness.
    """
    try:
    # Try to read the image
        #image = io.imread(url)
        wavelet_level=1
        # Calcular la transformada de wavelet
        coeffs = pywt.wavedec2(image, 'haar', level=wavelet_level)
    
        # Calcular el coarseness basado en los coeficientes de la transformada de wavelet
        taruma_coarseness = np.sum(np.abs(coeffs[wavelet_level])) / (image.shape[0] * image.shape[1])
    except:
        taruma_coarseness=np.nan
    return taruma_coarseness

def compute_taruma_linelikeness(image: np.ndarray) -> float:
    """
    Compute the Taruma linelikeness of an image.
    Linelikeness is a metric that seeks to quantify the presence and predominance of lines in an image. Although it is not a standard metric related 
    to Taruma direction, you can adapt the approach to calculate it using edge detection and directionality techniques. Below, I provide you with an 
    example of how to calculate a linelikeness metric in Python using edge detection and predominant direction calculation.
    
    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.
    Returns
    -------
    taruma_contrast : float
        Calculated Taruma linelikeness.
    """
    try:
        #image = io.imread(url)
        sobel_kernel=3
        # Calcular gradientes horizontales y verticales con el operador de Sobel
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
        # Calcular la magnitud y la dirección de los gradientes
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
    
        # Calcular una métrica de linelikeness basada en la dirección de los gradientes
        taruma_linelikeness = np.sum(np.abs(np.sin(2 * gradient_direction))) / (image.shape[0] * image.shape[1])
    except:
        taruma_linelikeness=np.nan
    return taruma_linelikeness
    
def compute_taruma_regularities(image: np.ndarray) -> float:
    """
    Compute the Taruma regularities of an image.
    Regularity" in an image refers to the uniformity and repeatability of patterns present in the image. There is no standard formula for 
    "regularity" as in the case of Taruma's direction. However, you can adapt texture analysis techniques to quantify regularity in an image. 
    Here is an example that uses the cooccurrence matrix to calculate a measure of regularity.

    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.
    Returns
    -------
    taruma_contrast : float
        Calculated Taruma regularities.
    """
    try:
        #image = io.imread(url)
        distance=1
        angles=[0]
        levels=256
        #image = rgb2gray(image)
        image_gray=converttogray(image,ind)
        image = np.array(image_gray, dtype='int64')
        # Calcular la matriz de coocurrencia
        cooccurrence_matrix = graycomatrix(image, [distance], angles, symmetric=True, normed=True, levels=levels)
        # Calcular medidas de regularidad
        contrast = graycoprops(cooccurrence_matrix, prop='contrast')
        correlation = graycoprops(cooccurrence_matrix, prop='correlation')
        # Calcular una métrica de regularity basada en las medidas de coocurrencia
        taruma_regularity = (contrast + (1 - correlation)) / 2
        taruma_regularity = taruma_regularity[0][0]
    except:
        taruma_regularity=np.nan
    return taruma_regularity

def compute_taruma_roughness(image: np.ndarray) -> float:
    """
    Compute the Taruma roughness of an image.
    Roughness refers to the texture or irregular patterns present in an image. There is  no specific roughness metric in the context of Taruma, 
    but you can adapt texture analysis techniques to quantify roughness in an image. Below, I provide an example that uses the co-occurrence 
    matrix to calculate a roughness measure.

    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.
    Returns
    -------
    taruma_contrast : float
        Calculated Taruma roughness.
    """
    try:
        #image = io.imread(url)
        distance=1
        angles=[0]
        levels=256
        #image = rgb2gray(image)
        image_gray=converttogray(image,1)
        #image = np.array(image, dtype='int64')
        # Calcular la matriz de coocurrencia
        cooccurrence_matrix = graycomatrix(image_gray, [distance], angles, symmetric=True, normed=True, levels=levels)
        # Calcular medidas de textura
        energy = graycoprops(cooccurrence_matrix, prop='energy')
        homogeneity = graycoprops(cooccurrence_matrix, prop='homogeneity')
        # Calcular una métrica de roughness basada en las medidas de coocurrencia
        taruma_roughness = (1 - energy) * homogeneity
        taruma_roughness = taruma_roughness[0][0]
    except:
        taruma_roughness=np.nan
    return taruma_roughness

def get_output_data(data):
    out = pd.DataFrame(columns=["date", "url", "entropy", "mean_intensity", "standard_deviation", 
                                "skewness", "kurtosis", "relative_smoothness", "uniformity",
                                "fractal_dimension", "taruma_contrast", "taruma_directionality", 
                                "taruma_coarseness", "taruma_linelikeness", "taruma_regularity",
                                "taruma_roughness"])
    for i in range(len(data)):
        date=data["date"][i]
        url=data["url"][i]
        try:
            image = io.imread(url)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(compute_entropy_statistics, image): "entropy_statistics",
                           executor.submit(compute_fractal_dimension, image): "fractal_dimension",
                           executor.submit(compute_taruma_directionality, image): "taruma_directionality",
                           executor.submit(compute_taruma_coarseness, image): "taruma_coarseness",
                           executor.submit(compute_taruma_linelikeness, image): "taruma_linelikeness",
                           executor.submit(compute_taruma_regularities, image): "taruma_regularity",
                           executor.submit(compute_taruma_roughness, image): "taruma_roughness"}

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if futures[future] == "entropy_statistics":
                        vec1 = result
                        entropy=vec1[0]
                        mean_intensity=vec1[1]
                        standard_deviation=vec1[2]
                        skewness=vec1[3]
                        kurtosis=vec1[4]
                        relative_smoothness=vec1[5]
                        uniformity=vec1[6]
                        taruma_contrast=(standard_deviation**2)/(kurtosis**(0.25))
                    elif futures[future] == "fractal_dimension":
                        fractal_dimension = result
                    elif futures[future] == "taruma_directionality":
                        taruma_directionality = result
                    elif futures[future] == "taruma_coarseness":
                        taruma_coarseness = result
                    elif futures[future] == "taruma_linelikeness":
                        taruma_linelikeness = result
                    elif futures[future] == "taruma_regularity":
                        taruma_regularity = result
                    elif futures[future] == "taruma_roughness":
                        taruma_roughness = result

            out.loc[i] = [date, url, entropy, mean_intensity, standard_deviation, skewness,
                          kurtosis, relative_smoothness, uniformity, fractal_dimension, taruma_contrast,
                          taruma_directionality, taruma_coarseness, taruma_linelikeness, taruma_regularity,
                          taruma_roughness]
        except:
            continue
    return out
    

def process_type(args):
    type, year = args
    data=pd.read_csv("URLS/urls_{}_{}.csv".format(str(year),type))

    start_time = time.time()
    out=get_output_data(data)
    output_name="output_"+str(year)+"_"+type+".csv"
    out.to_csv("DATA/"+output_name,index=False)
    print("Save data type {} from year {}:".format(type,str(year)),"as:",output_name)
    end_time = time.time()
    execution_time1 = -(start_time - end_time) / 60
    print("Execution time to analize images from urls:",execution_time1,"minutes")
    print("DOne...")

if __name__ == '__main__':

    types=["eit171","eit195","eit284","eit304","hmiigr","hmimag"]
    
    year=sys.argv[1]

    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(process_type, [(type, year) for type in types])
    except Exception as e:
        print(f"Se produjo un error: {e}")
        sys.exit(1)

    try:
        x = SubirDB()
        x.create_tables()
    except Exception as e:
        print(f"Se produjo un error al crear las tablas: {e}")
        sys.exit(1)


    subprocess.run(["cmd", "/c", "commit.bat"])

