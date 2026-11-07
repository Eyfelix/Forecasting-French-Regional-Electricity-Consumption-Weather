import pandas as pd
import numpy as np
import time
import os
from typing import *
import datetime
import argparse
import gzip
import shutil
import urllib.parse
import urllib.request

from sklearn.neighbors import BallTree


#* typing definitions
typingModelNames = Literal["polyReg","ridgeReg","decTree","adaBoostDecTree"]
typingDelta = Literal["ByHour","ByDay"]
typingScale = Literal["Region","Dep"]

#* Data directory : <repo>/data by default, can be changed with --data-dir
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_DIR = DEFAULT_DATA_DIR

#* Stable download URL of a data.gouv.fr resource (redirects to the actual file)
DATAGOUV_RESOURCE_URL = "https://www.data.gouv.fr/fr/datasets/r/{}"


DEPS = list(range(1, 96))
DEPS.pop(19) #* no corsica
N_NEIGHBORS = 5  # nombre de stations voisines pour la moyenne
METEO_COLS_HOUR = ["RR1","FF","T","TD","TN","TX","DG","U","UN","UX","DHUMI40","DHUMI80"]

regions_departements = {
    "Auvergne-Rhône-Alpes": [
        1, 3, 7, 15, 26, 38, 42, 43, 63, 69, 73, 74
    ],
    "Bourgogne-Franche-Comté": [
        21, 25, 39, 58, 70, 71, 89, 90
    ],
    "Bretagne": [
        22, 29, 35, 56
    ],
    "Centre-Val de Loire": [
        18, 28, 36, 37, 41, 45
    ],
    # "Corse": [ #* Corsica not in the RTE data scope
    #     20
    # ],
    "Grand Est": [
        8, 10, 51, 52, 54, 55, 57, 67, 68, 88
    ],
    "Hauts-de-France": [
        2, 59, 60, 62, 80
    ],
    "Ile-de-France": [
        75, 77, 78, 91, 92, 93, 94, 95
    ],
    "Nouvelle-Aquitaine": [
        16, 17, 19, 23, 24, 33, 40, 47, 64, 79, 86, 87
    ],
    "Normandie": [
        14, 27, 50, 61, 76
    ],
    "Occitanie": [
        9, 11, 12, 30, 31, 32, 34, 46, 48, 65, 66, 81, 82
    ],
    "Pays de la Loire": [
        44, 49, 53, 72, 85
    ],
    "PACA": [
        4, 5, 6, 13, 83, 84
    ]
}

#* features and important columns
relevantColsDay = ["NUM_POSTE",#* Numéro de poste,
                    "LAT", #* L'attitude
                    "LON", #* Longitude
                    "DATE", #* Date
                    "RR", #* Quantité de précipipation
                    "TN", #* Température minimal à l'abri
                    "TX", #* Température maximal à l'abri
                    "TM", #* Température moyenne à l'abri
                    "DG", #* Durée du gel
                    "FFM", #* Force moyenne des vents à 10m
                ]

relevantColsHOR = [ "NUM_POSTE",#* Numéro de poste,
                    "dateTime", #* Date et heure
                    "LAT", #* L'attitude
                    "LON", #* Longitude
                    "RR1", #* Quantitée des précipitations sur l'heure
                    "FF", #* Force moyenne du vent à 10m
                    "T", #* Température à l'abri
                    "TD", #* Température au point de rosé
                    "TN", #* Température minimal à l'abri
                    "TX", #* Température maximal à l'abri
                    "DG", #* Durée du gel en mn
                    "U", #* Humidité (%)
                    "UN",#* Humidité min (%)
                    "UX", #* Humidité max (%)
                    "DHUMI40", #* durée humidité <=40%
                    "DHUMI80", #* durée humidité <=80%
                    ]


#* BASIC FUNCTIONS 

def getFileNameMet(step:Literal["raw","cleaned"],delta:Literal["ByDay","ByHour"]=None,departementNumber:int=None,filled:bool=False):
    metoBase = "meteo_"
    number = str(departementNumber).zfill(2)
    if step=="raw":
        return os.path.join(DATA_DIR,step+"Data","meteo",delta,metoBase+number+".csv")

    return os.path.join(DATA_DIR,step+"Data","meteo","fill" if filled else "unfilled",delta,metoBase+number+".csv")

def getFileNameEnr(step:Literal["raw","cleaned"],region:str=None,year:int=None):
    if step=="raw":
        # print(os.path.join(step+"Data",dataSource,f"eCO2mix_RTE_{region}_Annuel-Definitif_{year}.zip"))
        return os.path.join(DATA_DIR,step+"Data","energy",f"eCO2mix_RTE_{region.replace(' ','-')}_Annuel-Definitif_{year}.xls")
    else:
        #* The RTE data is already preprocessed and versioned in data/energy
        return os.path.join(DATA_DIR,"energy",f"Consommation_{region.replace(' ','-')}.csv")



def getEnrData(region:str) -> pd.DataFrame:
    return pd.read_csv(getFileNameEnr("cleaned",region=region),parse_dates=["dateTime"])

def getMetData(delta:typingDelta,dep:int,filled:bool=False,detailled:bool=False) -> pd.DataFrame:
    if delta=="ByDay":
        if detailled:
            return pd.read_csv(getFileNameMet("cleaned",delta,dep,filled),parse_dates=["DATE"])
        usedColumns = ["DATE","RR","TN","TX","TM","DG","FFM"]
        return pd.read_csv(getFileNameMet("cleaned",delta,dep,filled),parse_dates=["DATE"],usecols=usedColumns)
    
    #* By Hour
    if detailled: 
            return pd.read_csv(getFileNameMet("cleaned",delta,dep,filled),parse_dates=["dateTime"])
    
    usedColumns = ["RR1","FF","T","TD","TN","TX","DG","U","UN","UX","DHUMI40","DHUMI80","dateTime"]
    return pd.read_csv(getFileNameMet("cleaned",delta,dep,filled),parse_dates=["dateTime"],usecols=usedColumns)



#* ============ DOWNLOADING WEATHER DATA ===================

def resourceIdFromUrl(url:str) -> Optional[str]:
    #* Extract the resource_id from a data.gouv.fr dataset page URL
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if "resource_id" in query:
        return query["resource_id"][0]
    return None


def downloadWeatherFiles(linksFile:str,delta:typingDelta,verbose:bool=True) -> None:
    #* Download (and unzip if needed) every file listed in linksFile into rawData/meteo/<delta>
    outDir = os.path.join(DATA_DIR,"rawData","meteo",delta)
    os.makedirs(outDir, exist_ok=True)

    with open(linksFile, encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    failed = []
    for i, url in enumerate(urls, 1):
        resourceId = resourceIdFromUrl(url)
        downloadUrl = DATAGOUV_RESOURCE_URL.format(resourceId) if resourceId else url
        try:
            request = urllib.request.Request(downloadUrl, headers={"User-Agent": "Mozilla/5.0 (dataset builder)"})
            with urllib.request.urlopen(request, timeout=120) as response:
                fileName = os.path.basename(urllib.parse.urlparse(response.url).path)
                csvName = fileName[:-3] if fileName.endswith(".gz") else fileName
                csvPath = os.path.join(outDir, csvName)

                if os.path.exists(csvPath):
                    if verbose:
                        print(f"[{i}/{len(urls)}] {csvName} already downloaded, skip")
                    continue

                if verbose:
                    print(f"[{i}/{len(urls)}] downloading {fileName}")
                tmpPath = os.path.join(outDir, fileName + ".part")
                with open(tmpPath, "wb") as out:
                    shutil.copyfileobj(response, out)

            if fileName.endswith(".gz"):
                with gzip.open(tmpPath, "rb") as src, open(csvPath + ".part", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.remove(tmpPath)
                os.replace(csvPath + ".part", csvPath)
            else:
                os.replace(tmpPath, csvPath)

        except Exception as e:
            print(f"[{i}/{len(urls)}] FAILED {url} : {e}")
            failed.append(url)

    print(f"Download {delta} finished, {len(urls)-len(failed)}/{len(urls)} files ok")
    if failed:
        print("Failed urls (re-run the download step to retry):")
        for url in failed:
            print("   ", url)



#* ============ PROCESSING WEATHER DATA ===================

def basicFilterWeatherDay(dep:int) -> None:
    print(dep,datetime.datetime.now())
    minDate = 20130101
    depStr = str(dep).zfill(2)
    rawPath = os.path.join(DATA_DIR,"rawData","meteo","ByDay",f"Q_{depStr}_previous-1950-2024_RR-T-Vent.csv")
    if not os.path.exists(rawPath):
        print(f"    missing raw file {rawPath}, skip")
        return
    dataMeteoRaw = pd.read_csv(
        rawPath,
        sep=";",
        usecols=lambda c: not c.startswith("STATUS_") and not c.startswith("Q") and not c=="NOM_USUEL" and not c=="ALTI"
    )

    dataDay = dataMeteoRaw[dataMeteoRaw["AAAAMMJJ"] >= minDate].copy()
    dataDay.insert(0, "DATE", pd.to_datetime(dataDay["AAAAMMJJ"], format="%Y%m%d").dt.date)
    dataDay.drop(columns=["AAAAMMJJ"], inplace=True)

    dataDay = dataDay[relevantColsDay]

    outPath = getFileNameMet("cleaned","ByDay",dep)
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    dataDay.to_csv(outPath, index=False)



def basicFilterWeatherHOR(dep:int) -> None:
    print(dep,datetime.datetime.now())

    unusedCols = ['CM', 'CH', 'N1', 'C1', 'B1', 'N2', 'C2', 'B2', 'N3', 'C3', 'B3',
       'GEOP', 'CL', 'N4', 'B4', 'C4', 'N', 'NBAS', 'DHUMEC', 'DIR', 'HNEIGEF',
       'HNEIGEFI3', 'TUBENEIGE', 'TSNEIGE', 'DIRHOULE', 'HVAGUE', 'PVAGUE',
       'ETATMER', 'TMER', 'VVMER', 'W2', 'W1', 'HNEIGEFI1', 'ESNEIGE',
       'CHARGENEIGE', 'DIF', 'UV_INDICE', 'TLAGON', 'ECOULEMENT', 'TVEGETAUX',
       'DIF2', 'UV', 'DIR2', 'INFRAR', 'INFRAR2', 'UV2', 'SOLNG']

    depStr = str(dep).zfill(2)
    rawPath = os.path.join(DATA_DIR,"rawData","meteo","ByHour",f"H_{depStr}_previous-2020-2024.csv")
    if not os.path.exists(rawPath):
        print(f"    missing raw file {rawPath}, skip")
        return
    dataWeatherRaw = pd.read_csv(rawPath,sep=";",
                               usecols=lambda c: not c.startswith("STATUS_") and not c.startswith("Q") and not c=="NOM_USUEL" and not c=="ALTI" and not c in unusedCols
                               )

    dataWeatherRaw.insert(0, "dateTime", pd.to_datetime(dataWeatherRaw["AAAAMMJJHH"], format="%Y%m%d%H"))
    dataWeatherRaw.drop(columns=["AAAAMMJJHH"], inplace=True)

    dataWeather = dataWeatherRaw[relevantColsHOR]

    outPath = getFileNameMet("cleaned","ByHour",dep)
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    dataWeather.to_csv(outPath, index=False)

#* ============== ENERGY DATASETS ================ 
#* Not run by default : the preprocessed RTE files are already in data/energy
def cleanEnergy(region:str) -> None:
    yearsData = []
    print(region)
    for year in range(2013,2025):
        # try : 
            yearConsommationDf = pd.read_csv(
                    getFileNameEnr("raw", region, year),
                    sep="\t",
                    encoding="latin-1",  # À cause des accents (é, è...)
                    # usecols=["Date", "Heures", "Consommation"],
                    usecols= [2, 3, 4],
                    na_values=["ND"]
                ).dropna()

            x_energy = pd.to_datetime(yearConsommationDf["Date"].astype(str) + " " + yearConsommationDf["Heures"])
            
            yearConsommationDf = yearConsommationDf.set_index(x_energy).resample("1h").sum(numeric_only=True).reset_index(names=["dateTime"]) #* somme les demi-heure en heure

            yearsData.append(yearConsommationDf)
    
    consommationDf = pd.concat(yearsData)
    consommationDf.sort_values(["dateTime"],inplace=True)
    consommationDf["Consommation"] = consommationDf["Consommation"] / 10_000

    consommationDf.to_csv(getFileNameEnr("cleaned",region,year),index=False)

#* ============== MERGING FOR DATASET CREATION ==============

def combineEnrAndMet(energyData:pd.DataFrame,weatherData:pd.DataFrame,delta:typingDelta) -> pd.DataFrame:
    if delta=="ByDay":
        return energyData.merge(weatherData,"inner",left_on="Date",right_on="DATE").drop(['DATE'],axis=1)
    
    return energyData.merge(weatherData,"inner",left_on="dateTime",right_on="dateTime")

def fillWithRegMean(combinedData:pd.DataFrame)->pd.DataFrame:
    #* Interpolates holls with the region mean

    base_cols = set(col.rsplit("_", 1)[0] for col in combinedData.columns if "_" in col)

    for base_col in base_cols:
        matching_cols = [col for col in combinedData.columns if col.rsplit("_", 1)[0] == base_col]
        
        
        col_mean = combinedData[matching_cols].mean(axis=1)
        
        combinedData[matching_cols] = combinedData[matching_cols].apply(lambda col: col.fillna(col_mean))
    
    return combinedData

def createDataSet(delta:typingDelta,scale:typingScale,region:str,fill:bool=False,verbose:bool=True) -> pd.DataFrame:
    if verbose:
        print(f"{region} stating at {datetime.datetime.now()}")

    if delta=="ByDay":
        dateMin = "2013-01-01"
        dateMax = "2024-12-31"
        dateCol = "DATE"
    else:
        dateMin = "2020-01-01 00:00:00"
        dateMax = "2024-12-31 23:00:00"
        dateCol = "dateTime"

    energyData = getEnrData(region)
    if delta=="ByDay": #* sum the energy data to have daily consumption
        energyData.insert(0,"Date",pd.to_datetime(energyData["dateTime"].dt.date))
        energyData.drop(["dateTime"],axis=1,inplace=True)
        energyData =  energyData.groupby("Date",as_index=False).sum(numeric_only=True)

    departementsData = []
    deps = regions_departements[region]
    for dep in deps: 
        if verbose:
            print("dep ", dep)
        depDf = getMetData(delta,dep,filled=fill)

        depDf = depDf[(depDf[dateCol] >= dateMin) & (depDf[dateCol] <= dateMax)]
        if verbose:
            print("shape df : ", depDf.shape)
        if scale=="Region":
   
            departementsData.append(depDf)
            
        if scale=="Dep":
            depAverage = depDf.groupby([dateCol]).mean() #* moyenne sur les heures
            depAverage.columns = [f"{col}_{dep}" for col in depAverage.columns]
            if verbose:
                print("shape Dep average : ", depAverage.shape)

            departementsData.append(depAverage)

    if scale=="Region":
        departementUnited = pd.concat(departementsData)
        if verbose:
            print("shape united : ", departementUnited.shape)
        regionAverage = departementUnited.groupby([dateCol],as_index=False).mean() #* Mean of all departements
        if verbose:
            print("shape regionAverage : ", regionAverage.shape)
        combinedData = combineEnrAndMet(energyData,regionAverage,delta)
        if verbose:
            print("shape combinedData : ", combinedData.shape)
    if scale=="Dep":

        departementUnited = pd.concat(departementsData, axis=1, join="outer").reset_index(names=[dateCol])
        if verbose:
            print("shape united : ", departementUnited.shape)

        if not fill: #* the weather data has not been filled so we fill it here
            departementUnited = fillWithRegMean(departementUnited)
            if verbose:
                print("shape united Filled: ", departementUnited.shape)

        combinedData = combineEnrAndMet(energyData,departementUnited,delta)
        if verbose:
                print("shape combinedData: ", combinedData.shape)

    if fill:
        outPath = os.path.join(DATA_DIR,"datasets","filled",f"{scale}_{delta}",f"{region}-data.csv")
    else:
        outPath = os.path.join(DATA_DIR,"datasets",f"{scale}_{delta}",f"{region}-data.csv")
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    combinedData.to_csv(outPath,index=False)

    if verbose:
        print(f"{region} Finish at {datetime.datetime.now()}, shape : {combinedData.shape}")


    return combinedData


#* =========== FILLING MISSING VALUES =================

adjacentDepts = {
    1 : [38, 73, 74, 39, 71, 69],
    2 : [59, 80, 60, 77, 51, 8],
    3 : [42, 71, 58, 18, 23, 63],
    4 : [83, 6, 5, 26, 84, 13],
    5 : [4, 26, 38, 73],
    6 : [83, 4],
    7 : [48, 30, 84, 26, 38, 69, 43],
    8 : [55, 51, 2, 59],
    9 : [31, 11, 66],
    10 :[89, 21, 52, 51, 77],
    11 :[9, 31, 81, 34, 66],
    12 :[34, 30, 48, 15, 46, 82, 81],
    13 :[30, 84, 83],
    14 :[27, 61, 50],
    15 :[48, 43, 63, 19, 46, 12],
    16 :[17, 79, 86, 87, 24],
    17 :[85, 79, 16, 33],
    18 :[36, 41, 45, 89, 58, 3, 23],
    19 :[23, 87, 24, 46, 15, 63],
    # 20 :
    21 :[10, 89, 58, 71, 39, 70, 52],
    22 :[29, 56, 35],
    23 :[87, 19, 63, 3, 18, 36],
    24 :[17, 16, 87, 19, 46, 47, 33],
    25 :[39, 70, 90],
    26 :[84, 4, 5, 38, 69, 7],
    27 :[76, 60, 95, 78, 28, 61, 14],
    28 :[27, 78, 91, 45, 41, 72, 61],
    29 :[22, 56],
    30 :[34, 12, 48, 7, 26, 84, 13],
    31 :[65, 32, 82, 81, 11, 9],
    32 :[31, 65, 64, 40, 47, 82],
    33 :[17, 24, 47, 40],
    34 :[11, 81, 12, 30],
    35 :[50, 53, 49, 44, 56, 22],
    36 :[37, 41, 18, 23, 87, 86],
    37 :[41, 36, 86, 49, 72],
    38 :[5, 73, 1, 69, 7, 26],
    39 :[1, 71, 21, 70, 25],
    40 :[33, 47, 32, 64],
    41 :[72, 37, 36, 18, 45, 28],
    42 :[43, 69, 71, 3, 63],
    43 :[48, 7, 69, 42, 63, 15],
    44 :[56, 35, 53, 49, 85],
    45 :[91, 77, 28, 41, 18, 58, 89],
    46 :[15, 19, 24, 47, 82, 12],
    47 :[33, 40, 32, 82, 46, 24],
    48 :[12, 30, 7, 43, 15],
    49 :[44, 53, 72, 37, 86, 79, 85, 35],
    50 :[14, 61, 53, 35],
    51 :[2, 77, 10, 52, 55, 8],
    52 :[21, 70, 88, 55, 51, 10],
    53 :[49, 72, 61, 50, 35, 44],
    54 :[57, 67, 88, 55],
    55 :[54, 88, 52, 51, 8],
    56 :[29, 22, 35, 44],
    57 :[67, 88, 54],
    58 :[3, 71, 21, 89, 45, 18],
    59 :[2, 80, 62, 8],
    60 :[2, 80, 76, 27, 95, 77],
    61 :[14, 27, 28, 72, 53, 50],
    62 :[80, 59, 2],
    63 :[15, 43, 42, 3, 23, 19],
    64 :[40, 32, 65],
    65 :[64, 32, 31],
    66 :[9, 11],
    67 :[68, 88, 54, 57],
    68 :[90, 70, 88, 67],
    69 :[1, 71, 42, 43, 7, 26, 38],
    70 :[21, 39, 25, 90, 88, 52],
    71 :[3, 42, 69, 1, 39, 21, 58],
    72 :[28, 61, 53, 49, 37, 41],
    73 :[74, 1, 38, 5],
    74 :[73, 1],
    75 :[92, 93, 94],
    76 :[80, 60, 27],
    77 :[89, 10, 51, 2, 60, 93, 91, 45, 95, 94],
    78 :[95, 27, 28, 91, 92],
    79 :[17, 85, 49, 86, 16],
    80 :[62, 59, 2, 60, 76],
    81 :[11, 34, 12, 82, 31],
    82 :[32, 31, 81, 80, 46, 47],
    83 :[13, 84, 4, 6],
    84 :[13, 4, 26, 7, 30],
    85 :[44, 49, 79, 17],
    86 :[36, 37, 49, 79, 16, 87],
    87 :[23, 19, 24, 16, 86],
    88 :[70, 90, 68, 67, 57, 54, 55, 52],
    89 :[58, 21, 10, 77, 45],
    90 :[25, 70, 88, 68],
    91 :[45, 28, 77, 78, 92, 94],
    92 :[75, 95, 78, 93],
    93 :[92, 95, 94, 77],
    94 :[77, 75, 92, 93, 91],
    95 :[78, 27, 60, 93, 92, 77],
}
N_NEIGHBORS_STATIONS = 5



postesNoNeighbors = []


def buildStationsIndex(delta: typingDelta) -> pd.DataFrame:
    
    records = []
    for dep in DEPS:
        try:
            df = pd.read_csv(
                getFileNameMet("cleaned", delta, dep, filled=False),
                usecols=["NUM_POSTE", "LAT", "LON"]
            )
            df["dep"] = dep
            records.append(df.drop_duplicates("NUM_POSTE"))
        except FileNotFoundError:
            pass
    return pd.concat(records, ignore_index=True).drop_duplicates("NUM_POSTE").reset_index(drop=True)



def buildBallTree(stations: pd.DataFrame):

    coords_rad = np.radians(stations[["LAT","LON"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    return tree

depDataCache = {} 

def loadDepData(dep: int, delta: typingDelta, timeCol: str) -> pd.DataFrame:

    if dep in depDataCache:
        return depDataCache[dep]
    try:
        df = pd.read_csv(
            getFileNameMet("cleaned", delta, dep, filled=False),
            parse_dates=[timeCol]
        ).set_index(["NUM_POSTE", timeCol]).sort_index()
        depDataCache[dep] = df
    except FileNotFoundError:
        depDataCache[dep] = pd.DataFrame()  #* dep missing
    return depDataCache[dep]


def loadNearestDepsData(dep: int, delta: typingDelta,timeCol: str) -> pd.DataFrame:

    timer = time.time()

    nearestDeps = adjacentDepts[dep] + [dep]
    dfs = [loadDepData(d, delta, timeCol) for d in nearestDeps]
    print(f"    nearest load in {time.time()-timer} s ")

    dfs = [df for df in dfs if not df.empty]

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs).sort_index()


def fillMissingValues(depDf:pd.DataFrame, stations:pd.DataFrame, stationsTree:BallTree, nearestDepsData:pd.DataFrame, metCols, timeCol, nNeighbors=N_NEIGHBORS_STATIONS):
    timer = time.time()
    depDf = depDf.copy()
    postes = depDf["NUM_POSTE"].unique()
    print(f"{postes.shape[0]} postes to fill")
    posteN = 0
    for poste in postes:
        posteN +=1 
        print(f"        fill value for {posteN}/{postes.shape[0]}, function timer : {(time.time()-timer)/60} min")
        stationMask = depDf["NUM_POSTE"] == poste
        stationInfo = stations[stations["NUM_POSTE"] == poste]
        if stationInfo.empty:
            print("         Stations not existing ")
            continue
        if depDf[stationMask].isna().any().sum() ==0:
            print("         Stations filled, skip")
            continue
        
        coords_rad = np.radians([[stationInfo["LAT"].values[0], stationInfo["LON"].values[0]]])
        k = 50 * min(nNeighbors + 1, len(stations)) #* we take more stations to be sure we have at least one value
        _, indices = stationsTree.query(coords_rad, k=k)
        neighborPostes = [p for p in stations.iloc[indices[0]]["NUM_POSTE"].values if p != poste]

        stationRows = depDf[stationMask]
        colsWithNan = [c for c in metCols if stationRows[c].isna().any()]
        print(f"        found nearest stations, in {(time.time()-timer)/60} min")
        print(f"        filling {len(colsWithNan)} cols")
        timer2 = time.time()
        for col in colsWithNan:
            nanMask = stationMask & depDf[col].isna()
            nanTimestamps = depDf.loc[nanMask, timeCol].values
            print(f"            col : {col} nan : {nanTimestamps.shape} | intern timer {(time.time()-timer2)} s")
            for ts in nanTimestamps:

                neighborValues = []
                for i,neighborPoste in enumerate(neighborPostes): 
                    try:
                        val = nearestDepsData.loc[(neighborPoste, ts), col]
                        if pd.notna(val):
                            neighborValues.append(val)
                            if len(neighborValues)==nNeighbors: #* Escape early the loop
                                break
                        if i>50 and len(neighborValues)>0:
                            break
                    except KeyError:
                        pass
                if neighborValues:
                    depDf.loc[nanMask & (depDf[timeCol] == ts), col] = np.mean(neighborValues[:nNeighbors])
                else:
                    print("     NO NEIGHBORS FOUND")
                    postesNoNeighbors.append(poste)
    return depDf


def fillAllDepartements(delta: typingDelta, nNeighbors: int = N_NEIGHBORS_STATIONS):
    global depDataCache
    depDataCache = {}  

    timeCol = "dateTime" if delta == "ByHour" else "DATE"
    metCols = METEO_COLS_HOUR if delta == "ByHour" else ["RR","TN","TX","TM","DG","FFM"]
    timer = time.time()

    stationsPath = os.path.join(DATA_DIR,f"stationsCoord{delta}.csv")
    try:
        stations = pd.read_csv(stationsPath)
    except:
        stations     = buildStationsIndex(delta) #* If not
        stations.to_csv(stationsPath,index=False)


    print(f"Stations load in {(time.time()-timer)/60} min")
    
    stationsTree = buildBallTree(stations)
    print(f"Stations ball tree in {(time.time()-timer)/60} min")

    print(f"  {len(stations)} stations | {(time.time()-timer)/60} min")

    for dep in DEPS:
        print(f"Running for : {(time.time()-timer)/60} min")
        print(f"=========== Département {dep} ============")
        try:
            #* resume from the filled file if it exists, otherwise start from the unfilled one
            filledPath = getFileNameMet("cleaned", delta, dep, filled=True)
            sourcePath = filledPath if os.path.exists(filledPath) else getFileNameMet("cleaned", delta, dep, filled=False)
            depDf = pd.read_csv(sourcePath, parse_dates=[timeCol])
        except FileNotFoundError:
            print(f"  File {delta} {dep} cleaned not found")
            continue
        print(f"    file load in {(time.time()-timer)/60} min")
        if depDf.isna().any().sum() ==0:
            print("    file already filled, skip")
            continue
        
        nearestDepsData = loadNearestDepsData(dep, delta, timeCol)
        print(f"    nearest Load in {(time.time()-timer)/60} min ")

        depDf = fillMissingValues(depDf, stations, stationsTree, nearestDepsData, metCols, timeCol, nNeighbors)
        print(f"    filled missed value in {(time.time()-timer)/60} min")

        outPath = getFileNameMet("cleaned", delta, dep, filled=True)
        print(f"    file saved in {(time.time()-timer)/60} min")
        os.makedirs(os.path.dirname(outPath), exist_ok=True)
        depDf.to_csv(outPath, index=False)
        print(f"  Sauvegardé : {outPath}")

    print("Finish.")
    print(f"Stations without enough neighbors : {len(postesNoNeighbors)} \n {postesNoNeighbors}")



#* =========== COMMAND LINE =================

STEPS = ["download","filter","build","fill"]

def parseArgs():
    parser = argparse.ArgumentParser(
        description="Download the Météo-France data from data.gouv.fr and build the weather / consumption datasets. "
                    "The RTE consumption data is already preprocessed in data/energy."
    )
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS,
                        help="Steps to run, in this order : download raw weather files, filter them, "
                             "build the datasets, fill missing values and build the filled datasets (ByDay only). Default : all")
    parser.add_argument("--delta", nargs="+", choices=["ByDay","ByHour"], default=["ByDay","ByHour"],
                        help="Temporal resolution(s) to process. Default : both")
    parser.add_argument("--scale", nargs="+", choices=["Region","Dep"], default=["Region","Dep"],
                        help="Geographic scale(s) of the datasets to build. Default : both")
    parser.add_argument("--regions", nargs="+", choices=list(regions_departements.keys()), default=list(regions_departements.keys()),
                        metavar="REGION", help="Regions to build (quote names with spaces). Default : all")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"Data directory. Default : {DEFAULT_DATA_DIR}")
    parser.add_argument("--day-links", default=None,
                        help="Text file with the daily data urls. Default : <data-dir>/ressoourcesLink.txt")
    parser.add_argument("--hour-links", default=None,
                        help="Text file with the hourly data urls. Default : <data-dir>/HORLinks.txt")
    parser.add_argument("--neighbors", type=int, default=N_NEIGHBORS_STATIONS,
                        help=f"Number of neighbouring stations used to fill missing values. Default : {N_NEIGHBORS_STATIONS}")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    return parser.parse_args()


if __name__ =="__main__":
    args = parseArgs()
    DATA_DIR = os.path.abspath(args.data_dir)
    verbose = not args.quiet
    linksFiles = {
        "ByDay": args.day_links or os.path.join(DATA_DIR,"ressoourcesLink.txt"),
        "ByHour": args.hour_links or os.path.join(DATA_DIR,"HORLinks.txt"),
    }

    #* DOWNLOAD THE RAW WEATHER DATA
    if "download" in args.steps:
        for delta in args.delta:
            downloadWeatherFiles(linksFiles[delta], delta, verbose)

    #* CLEANED THE DATA
    #* Energy : already preprocessed in data/energy (see cleanEnergy)
    #* Weather
    if "filter" in args.steps:
        for dep in DEPS:
            if "ByDay" in args.delta:
                basicFilterWeatherDay(dep)
            if "ByHour" in args.delta:
                basicFilterWeatherHOR(dep)

    #* CREATE THE REGION AND DEPARTEMENT DATASETS
    if "build" in args.steps:
        for scale in args.scale:
            for region in args.regions:
                for delta in args.delta:
                    createDataSet(delta,scale,region,fill=False,verbose=verbose)

    #* CREATE THE FILLED DATASET (ByDay only)
    if "fill" in args.steps:
        if "ByDay" in args.delta:
            #* Fill the weather datasets
            fillAllDepartements("ByDay",args.neighbors)

            for region in args.regions:
                for scale in args.scale:
                    createDataSet("ByDay",scale,region,fill=True,verbose=verbose)
        else:
            print("The fill step is only available for ByDay, skipped")