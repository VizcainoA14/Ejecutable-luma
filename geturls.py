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
import os
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from SubiraBD import SubirDB
import subprocess
import pandas as pd
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

# Base url from server in witch located the images...
path="https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/"


def get_files_for_month(year, type, m):
    soup2 = BeautifulSoup(requests.get(path+"{}/{}/{}".format(year,type,m)).text,features="lxml")
    urls2, month, day, Hour, Minute = [], [], [], [], []
    for link in soup2.find_all('a')[5:]:
        if "_1024." in link.get('href'):
            urls2.append(path+"{}/{}/{}".format(year,type,m)+link.get('href'))
            month.append(link.get('href')[4:6])
            day.append(link.get('href')[6:8])
            Hour.append(link.get('href')[9:11])
            Minute.append(link.get('href')[11:13])
    return urls2, month, day, Hour, Minute

def get_all_files(year,type):
    soup = BeautifulSoup(requests.get(path+"{}/{}".format(year,type)).text,features="lxml")
    urls = [link.get('href') for link in soup.find_all('a')[5:]]

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda args: get_files_for_month(*args), [(year, type, m) for m in urls])

    urls2, month, day, Hour, Minute = [], [], [], [], []
    for result in results:
        urls2.extend(result[0])
        month.extend(result[1])
        day.extend(result[2])
        Hour.extend(result[3])
        Minute.extend(result[4])

    y=[year]*len(urls2)
    urls_data=pd.DataFrame()
    urls_data["Year"]=y
    urls_data["Month"]=month
    urls_data["Day"]=day
    urls_data["Hour"]=Hour
    urls_data["Minute"]=Minute
    urls_data["url"]=urls2
    date=pd.to_datetime(urls_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    urls_data["date"]=date
    urls_data=urls_data.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])
    urls_data=urls_data[["date","url"]]
    return urls_data

def process(year, type):
    start_time = time.time()
    # Precomputing urls from images availables in server...
    url_files=get_all_files(year,type)
    output_name="urls_"+str(year)+"_"+type+".csv"
    if not os.path.exists('URLS'):
        os.makedirs('URLS')
    url_files.to_csv("URLS/"+output_name,index=False)
    print("Save data type {} from year {}:".format(type,str(year)),"as:",output_name)
    end_time = time.time()
    execution_time1 = -(start_time - end_time) / 60
    print("Execution time to get urls:",execution_time1,"minutes")
    print("DOne...")

if __name__ == '__main__':

    types=["eit171","eit195","eit284","eit304","hmiigr","hmimag"]
    Max_date = []

    for type in types:
        #get max years
        x =  SubirDB()
        max_date = datetime.strptime(x.max_date(type), '%Y-%m-%d %H:%M:%S')
        Max_date.append(max_date)

    year = "2024"

    with ProcessPoolExecutor() as executor:
        executor.map(process, [year]*len(types), types)

    print("Todas las tareas se han completado.")


    archivos_csv = ["URLS/urls_" + str(year) + "_" + type + ".csv" for type in types]

    for archivo, max_date in zip(archivos_csv, Max_date):
        # Leer el archivo CSV en un DataFrame
        df = pd.read_csv(archivo)

        # Convertir la columna 'date' a datetime
        df['date'] = pd.to_datetime(df['date'])

        # Filtrar el DataFrame para que solo contenga las filas con fechas mayores a la fecha máxima
        df = df[df['date'] > max_date]

        # Guardar el DataFrame filtrado de nuevo en el archivo CSV
        df.to_csv(archivo, index=False)

    subprocess.run([sys.executable, "./saiyajin_v3.py", year])
