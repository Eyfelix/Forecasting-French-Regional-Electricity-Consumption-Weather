import pandas as pd
import os
from typing import *



DEPS = list(range(1, 96))
N_NEIGHBORS = 5  # nombre de stations voisines pour la moyenne
METEO_COLS_HOUR = ["RR1","FF","T","TD","TN","TX","DG","U","UN","UX","DHUMI40","DHUMI80"]


def getFileNameMet(step:Literal["raw","cleaned"],delta:Literal["ByDay","ByHour"]=None,departementNumber:int=None,filled:bool=False):
    metoBase = "meteo_"
    number = str(departementNumber).zfill(2)

    return os.path.join(step+"Data","meteo","fill" if filled else "unfilled",delta,metoBase+number+".csv")

def getFileNameEnr(step:Literal["raw","cleaned"],region:str=None,year:int=None):
    if step=="raw":
        # print(os.path.join(step+"Data",dataSource,f"eCO2mix_RTE_{region}_Annuel-Definitif_{year}.zip"))
        return os.path.join(step+"Data","energy",f"eCO2mix_RTE_{region.replace(" ","-")}_Annuel-Definitif_{year}.xls")
    else:
        return os.path.join(step+"Data","energy",f"Consommation_{region.replace(" ","-")}.csv")
    
def getEnrData(region:str):
    return pd.read_csv(getFileNameEnr("cleaned",region=region),parse_dates=["dateTime"])

def getMetData(delta:Literal["ByDay","ByHour"],dep:int,filled:bool=False,detailled:bool=False):
    if delta=="ByDay":
        if detailled:
            return pd.read_csv(getFileNameMet("cleaned",delta,dep,filled),parse_dates=["DATE"])
        usedColumns = ["RR","TN","TX","TM","TNSOL","TN50","DG","FFM","FF2M","DRR","DATE"]
        return pd.read_csv(getFileNameMet("cleaned",delta,dep,filled),parse_dates=["DATE"],usecols=usedColumns)
    
    #* By Hour
    if detailled: 
            return pd.read_csv(getFileNameMet("cleaned",delta,dep,filled),parse_dates=["dateTime"])
    
    usedColumns = ["RR1","DRR1","FF","T","TD","TN","TX","DG","TNSOL","TN50","U","UN","UX","DHUMI40","DHUMI80","dateTime"]
    return pd.read_csv(getFileNameMet("cleaned",delta,dep,filled),parse_dates=["dateTime"],usecols=usedColumns)



