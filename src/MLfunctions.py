from src.createDataSet import *

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer,StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.base import BaseEstimator, TransformerMixin


defaultsSearchParamsGrid = {
            "polyReg":{
                "use_hour": [True],
                "use_weekDay": [True],
                "use_yearDay": [True],
                "pca__precision": ["0", "0.01", "0.005"],
                "poly__degree": [2, 3],
            },
            "ridgeReg":{
                "use_hour": [True],
                "use_weekDay": [True],
                "use_yearDay": [True],
                "pca__precision": ["0", "0.01", "0.005"],
                "poly__degree": [2],
                "ridge__alpha": [0.005, 0.01, 0.1, 1, 10]
            },
            "decTree":{
                "use_hour": [True],
                "use_weekDay": [True],
                "use_yearDay": [True],
                "pca__precision": ["0", "0.01", "0.005"],
                "decTree__max_depth": [4, 6, 8],
                "decTree__max_features": [None, "sqrt", "log2"]
            },
            "adaBoostTree":{
                "use_hour": [True],
                "use_weekDay": [True],
                "use_yearDay": [True],
                "pca__precision": ["0", "0.01", "0.005"],
                "adaBoostTree__n_estimators": [100, 300, 500],
                "adaBoostTree__estimator__max_depth": [6, 8, 10]
            }
}


#* Import the datasets
def getCombinedDatas(delta:typingDelta,scale:typingScale,fill:bool=False) -> List[pd.DataFrame]:
    combinedDatas = []
    for region in regions_departements.keys():
        if fill:
            df = pd.read_csv(os.path.join("datasets","filled",f"{scale}_{delta}",f"{region}-data.csv"),parse_dates=["Date"] if delta=="ByDay" else ["dateTime"])
        else:
            df = pd.read_csv(os.path.join("datasets","unfilled",f"{scale}_{delta}",f"{region}-data.csv"),parse_dates=["Date"] if delta=="ByDay" else ["dateTime"])

        combinedDatas.append(df)
    return combinedDatas

def getTestTrain(combinedData,separatingDate):
    if "dateTime" in combinedData.columns: #* ByHour
            
        trainData = combinedData[combinedData['dateTime'].dt.date<pd.to_datetime(separatingDate).date()]
        testData = combinedData[combinedData['dateTime'].dt.date>=pd.to_datetime(separatingDate).date()]

        Xtrain = trainData.drop(["Consommation"],axis=1)
        yTrain = trainData[["Consommation"]]
        Xtest  = testData.drop(["Consommation"],axis=1)
        yTest  = testData[["Consommation"]]
        
    else:
        trainData = combinedData[combinedData['Date']<separatingDate]
        testData = combinedData[combinedData['Date']>=separatingDate]

        Xtrain = trainData.drop(["Consommation"], axis=1)
        yTrain = trainData[["Consommation"]]
        Xtest  = testData.drop(["Consommation"], axis=1)
        yTest  = testData[["Consommation"]]

    return Xtrain,yTrain.to_numpy().ravel(),Xtest,yTest.to_numpy().ravel()


#* ======== PREPROCESSING ===========
def addLastCons(combinedData:pd.DataFrame,nbDays:int=5) -> pd.DataFrame:
    if nbDays==0:
        return combinedData
    if "dateTime" in combinedData.columns:
        combinedData = combinedData.sort_values(['dateTime'])
    else:
        combinedData = combinedData.sort_values(['Date'])
    
    combinedData = combinedData.reset_index(drop=True)
    combinedShifted = combinedData["Consommation"].shift(list(range(1,nbDays+1)))
    combinedAugmented = combinedData.merge(combinedShifted,left_index=True,right_index=True)
    
    combinedAugmented = combinedAugmented.dropna().reset_index(drop=True) #* We drop the first row because they dont have previous consommation

    return combinedAugmented


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

class DaysTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, use_hour=True, use_weekDay=True, use_yearDay=True):
        self.use_hour = use_hour
        self.use_weekDay = use_weekDay
        self.use_yearDay = use_yearDay

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if "dateTime" in X.columns: #* for the ByHour part
            if self.use_hour:
                X["hour"] = X["dateTime"].dt.hour
            if self.use_weekDay:
                X["weekDay"] = X["dateTime"].dt.day_of_week
            if self.use_yearDay:
                X["yearDay"] = X["dateTime"].dt.day_of_year
            X.drop(["dateTime"],axis=1,inplace=True)
        else:
            if self.use_weekDay:
                X["weekDay"] = X["Date"].dt.day_of_week
            if self.use_yearDay:
                X["yearDay"] = X["Date"].dt.day_of_year
            X.drop(["Date"],axis=1,inplace=True)
            
        return X

class SinCosTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, use_hour=True, use_weekDay=True, use_yearDay=True):
        self.use_hour = use_hour
        self.use_weekDay = use_weekDay
        self.use_yearDay = use_yearDay

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        transformers = []
        if self.use_weekDay and "weekDay" in X.columns:
            transformers += [
                ("weekDaySin", sin_transformer(7),  ["weekDay"]),
                ("weekDayCos", cos_transformer(7),  ["weekDay"]),
                ("weekDayDrop", "drop",             ["weekDay"]),
            ]
        if self.use_yearDay and "yearDay" in X.columns:
            transformers += [
                ("yearDaySin", sin_transformer(364), ["yearDay"]),
                ("yearDayCos", cos_transformer(364), ["yearDay"]),
                ("yearDayDrop", "drop",              ["yearDay"]),
            ]
        if self.use_hour and "hour" in X.columns:
            transformers += [
                ("hourSin", sin_transformer(24), ["hour"]),
                ("hourCos", cos_transformer(24), ["hour"]),
                ("hourDrop", "drop",             ["hour"]),
            ]
        ct = ColumnTransformer(transformers, remainder='passthrough')
        return ct.fit_transform(X)

class PcaTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,nComp=0):
        self.nComp = nComp
    
    def fit(self, X, y=None):
        if self.nComp > 0:
            self.pca_ = PCA(self.nComp)
            self.pca_.fit(X)
        return self

    def transform(self, X, y=None):
        if self.nComp <= 0:
            return X
        return self.pca_.transform(X)
    


#* =========== GRID SEARCH =================

def loadPcaComponents(filepath: str = "configs/pcaParams.json") -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def getPcaComponentsForRegion(region: str, 
                              precision: str, 
                              delta: typingDelta, 
                              scale: typingScale,
                              pcaConfig: dict) -> int:

    key = f"{scale}_{delta}"
    if precision=="0":
        return -1
    try:
        return pcaConfig[key][precision][region]
    except KeyError:
        print(f"Warning: PCA components not found for {region} - {precision} - {key}")
        return None

def mapPcaComponentsToPrecision(pcaConfig: dict, delta: str, scale: str) -> dict:
    mapping = {}
    key = f"{scale}_{delta}"
    
    # Add special case for no PCA
    for region in pcaConfig[key]["0.01"].keys():
        mapping[(region, -1)] = "0"
    
    # Map each (region, nComp) to its precision
    for precision in ["0.01", "0.005"]:
        for region, nComp in pcaConfig[key][precision].items():
            mapping[(region, nComp)] = precision
    
    return mapping


def convertParamGrid(region: str, delta: typingDelta, scale: typingScale, baseParamGrid: dict,isGrid:bool) -> dict:

    pcaConfig = loadPcaComponents()
    dynamicParamGrid = baseParamGrid.copy()
    precisions = dynamicParamGrid.get("pca__precision",0)
    useHour = dynamicParamGrid.get("use_hour",True)
    useWeekDay = dynamicParamGrid.get("use_weekDay",True)
    useYearDay = dynamicParamGrid.get("use_yearDay",True)

    paramsToDel = ["pca__precision","use_hour","use_weekDay","use_yearDay"]
    for p in paramsToDel:
        if p in dynamicParamGrid:
                del dynamicParamGrid[p]


    if not isGrid:

        dynamicParamGrid["pca__nComp"] = getPcaComponentsForRegion(region, str(precisions), delta, scale, pcaConfig)
        dynamicParamGrid["days__use_hour"] = useHour
        dynamicParamGrid["sincos__use_hour"] = useHour
        dynamicParamGrid["days__use_weekDay"] = useWeekDay
        dynamicParamGrid["sincos__use_weekDay"] = useWeekDay
        dynamicParamGrid["days__use_yearDay"] = useYearDay
        dynamicParamGrid["sincos__use_yearDay"] = useYearDay

        return dynamicParamGrid
    
    pcaComponentsList = []  # include no PCA option
    
    for precision in precisions:
      
        nComp = getPcaComponentsForRegion(region, str(precision), delta, scale, pcaConfig)
        if nComp is not None:
            pcaComponentsList.append(nComp)
    
    dynamicParamGrid["pca__nComp"] = pcaComponentsList
    dynamicParamGrid["days__use_hour"] = useHour if type(useHour) is list else [useHour]
    dynamicParamGrid["sincos__use_hour"] = useHour if type(useHour) is list else [useHour]
    dynamicParamGrid["days__use_weekDay"] = useWeekDay if type(useWeekDay) is list else [useWeekDay]
    dynamicParamGrid["sincos__use_weekDay"] = useWeekDay if type(useWeekDay) is list else [useWeekDay]
    dynamicParamGrid["days__use_yearDay"] = useYearDay if type(useYearDay) is list else [useYearDay]
    dynamicParamGrid["sincos__use_yearDay"] = useYearDay if type(useYearDay) is list else [useYearDay]

    return dynamicParamGrid

def saveGridSearchResults(filepath:str, collectedInformations:dict) -> None:
    serializable = {}
    for region, results in collectedInformations.items():
        serializable[region] = []
        for entry in results:
            serializable[region].append({
                "bestIndex":  int(entry["bestIndex"]),
                "allResults": entry["allResults"]
            })
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"    Result saved in {filepath}")

def loadGridSearchResults(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def dictToPandasResult(collectedInformations: dict,delta: str,  scale: str, filters: dict = None,onlyMean=False,dropIndex: list = None) -> pd.DataFrame:
    statistics = ["meanTestScore", "meanTrainScore", "meanFitTime", "meanTestTime"]
    
    pcaConfig = loadPcaComponents()
    # Récupère les noms de paramètres depuis la première entrée
    first_entries = next(iter(collectedInformations.values()))
    param_keys = list(first_entries[0]["allResults"][0]["params"].keys())

    pcaPrecisionMapping = mapPcaComponentsToPrecision(pcaConfig, delta, scale)

    rows = []
    for region, entries in collectedInformations.items():
        for entry in entries:
            for result in entry["allResults"]:
                row = result["params"].copy()
                row["region"] = region
                if "pca__nComp" in row:
                    nComp = row.pop("pca__nComp")
                    precision = pcaPrecisionMapping.get((region, nComp), "unknown")
                    row["pca__precision"] = precision

                for stat in statistics:
                    row[stat] = result.get(stat, np.nan)
                rows.append(row)
    
    df = pd.DataFrame(rows)

    if "pca__nComp" in param_keys:
        param_keys.remove("pca__nComp")
        param_keys.append("pca__precision")

    if dropIndex is not None:
        param_keys = [key for key in param_keys if key not in dropIndex]
        
    if filters is not None:
        for paramName, acceptedValues in filters.items():
            if paramName in df.columns:
                if df[paramName].dtype in [np.float64, np.float32, np.float16]:

                    mask = np.zeros(len(df), dtype=bool)
                    for v in acceptedValues:
                        mask |= np.isclose(df[paramName], v)
                    df = df[mask]
                else:
                    df = df[df[paramName].isin(acceptedValues)]

    
    df[param_keys] = df[param_keys].fillna("None")
    
    
    df_pivot = df.pivot_table(
        index=param_keys,
        columns="region",
        values=statistics,
        aggfunc="mean"
    )
    

    df_pivot = df_pivot.swaplevel(axis=1)
    df_pivot = df_pivot.sort_index(axis=1)

    for stat in statistics:
        regionCols = [(region, stat) for region in sorted(collectedInformations.keys())]
        df_pivot[("Moyenne", stat)] = df_pivot[regionCols].mean(axis=1)
    if onlyMean:
        df_pivot = df_pivot[["Moyenne"]]
    return df_pivot

def displayResults(collectedInformations: dict,delta: typingDelta,  scale: typingScale, filters: dict = None,onlyMean=False,dropIndex: list = None) -> None:
    df = dictToPandasResult(collectedInformations, delta,  scale, filters, onlyMean,dropIndex)

    # Format ridge__alpha index if it exists
    if "ridge__alpha" in df.index.names:
        renamer = lambda x: str(int(float(x))) if float(x) == int(float(x)) else f"{float(x):.3f}"

        level_idx = df.index.names.index("ridge__alpha")
        df.rename(index = renamer,level=level_idx,inplace=True)

        
    return df.style \
        .highlight_max(color="green", axis=0, subset=pd.IndexSlice[:, pd.IndexSlice[:, ["meanTestScore"]]]) \
        .highlight_max(color="cyan", axis=0, subset=pd.IndexSlice[:, pd.IndexSlice[:, ["meanTrainScore"]]]) \
        .highlight_min(color="orange", axis=0, subset=pd.IndexSlice[:, pd.IndexSlice[:, ["meanFitTime"]]]) \
        .highlight_min(color="magenta", axis=0, subset=pd.IndexSlice[:, pd.IndexSlice[:, ["meanTestTime"]]]) \
        .format("{:.4f}") 

def getBestParams(collectedInformations, region=None):

    results = {}
    target = {region: collectedInformations[region]} if region else collectedInformations
    for reg, entries in target.items():
        best_entry = max(entries, key=lambda e: e["allresult"][e["bestIndex"]]["meanTestScore"])
        results[reg] = {
            "score":      best_entry["allresult"][best_entry["bestIndex"]]["meanTestScore"],
            "params":     best_entry["bestParams"]
        }
    return results[region] if region else results

def getGridSearchResultsName(delta:typingDelta,scale:typingScale,modelName:typingModelNames,saveN:int=1,fill:bool=False):
    if not os.path.exists("gridSearchResult"):
        os.makedirs("gridSearchResult")
    return os.path.join("gridSearchResult",f"gridsearch_{'filled_' if fill else ''}{scale}_{delta}_{modelName}_results_{saveN}.json")

def findBestScoreIdx(allScores:dict):
    return max([(i,infos) for i,infos in enumerate(allScores)],key=lambda x: x[1]["meanTestScore"],default=0)[0]


def launchGridSearch(delta:typingDelta,scale:typingScale,param_grid:dict,modelName:typingModelNames,saveN:int,fill:bool):

    
    combinedDatas = getCombinedDatas(delta,scale,fill)

    tsCv_day = TimeSeriesSplit(
        n_splits=2,
        gap=11, #* add least nb of last cons added
        max_train_size=3_000,
        test_size=670,
    )
    tsCv_Hour = TimeSeriesSplit(
        n_splits=2,
        gap=50, #* add least nb of last cons added
        max_train_size=30_000,
        test_size=8_000,
    )
    usedCv = tsCv_day if delta=="ByDay" else tsCv_Hour

    pipeLine = getPipeline(modelName)

    timer = time.time()
    lastConsToTry = param_grid.get("nbLstCons",[3] if delta=="ByDay" else [5])
    if "nbLstCons" in param_grid:
        del param_grid["nbLstCons"]

    collectedInformations = {}
    for combinedData, region in zip(combinedDatas,regions_departements.keys()):
        
        print(f"Gridsearch start for : {region}")

        dynamicParamGrid = convertParamGrid(
            region, delta, scale, param_grid, isGrid=True
        )

        allResults = []
        for nbLstCons in lastConsToTry:

            combineLastDay = addLastCons(combinedData,nbLstCons)

            Xtrain,yTrain,_,_ = getTestTrain(combineLastDay,"2023-01-01")

            gridSearch = GridSearchCV(pipeLine,param_grid=dynamicParamGrid,cv=usedCv,n_jobs=-1,return_train_score=True)


            gridSearch.fit(Xtrain,yTrain)

            allResults += [
                    {
                        "params": {**params, "nbLstCons":nbLstCons},
                        "meanTestScore": round(float(testScore), 6),
                        "meanTrainScore": round(float(trainScore), 6),
                        "meanFitTime": round(float(fitTime), 6),
                        "meanTestTime": round(float(testTime), 6),
                    }
                    for params, testScore,trainScore, fitTime,testTime in zip(
                        gridSearch.cv_results_["params"],
                        gridSearch.cv_results_["mean_test_score"],
                        gridSearch.cv_results_["mean_train_score"],
                        gridSearch.cv_results_["mean_fit_time"],
                        gridSearch.cv_results_["mean_score_time"],
                    )
                ]



        if region not in collectedInformations:
            collectedInformations[region] = []
        collectedInformations[region].append({
            "bestIndex":  findBestScoreIdx(allResults),
            "allResults": allResults
        })

        print(f"    bestParams: {gridSearch.best_params_} - bestScore: {gridSearch.best_score_:.5f}")
        print(f"    Runnig for {(time.time()-timer)/60} min ...")
        saveGridSearchResults(getGridSearchResultsName(delta,scale,modelName,saveN,fill), collectedInformations) #* Save in the loop to avoid loosing information if bug
    
    print(f"    Finish in {(time.time()-timer)/60} min at {datetime.datetime.now()}")
    return collectedInformations

#* ================ SCORING ===================

def getScoreResultName(delta:typingDelta,scale:typingScale,modelName:typingModelNames):
    if not os.path.exists("scoringResult"):
        os.makedirs("scoringResult")
    return os.path.join("scoringResult",f"score_{scale}_{delta}_{modelName}_results.json")

def saveScoreResult(filepath:str,scoresInfos:dict):

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(scoresInfos, f, indent=2, ensure_ascii=False)
    print("Result save in ",filepath)

def jsonScoreToDf(scores:json) -> pd.DataFrame:
    # Récupère le nom du modèle depuis la première entrée
    modelName = next(iter(scores.values()))["model"]

    rows = {
        region: {
            (modelName, "trainScore"): info["trainScore"],
            (modelName, "testScore"):  info["testScore"],
        }
        for region, info in scores.items()
    }

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["model", "score"])
    df.index.name = "region"

    return df


def loadScoreResult(filepath: str, how: Literal["json", "df"]):
    with open(filepath, "r", encoding="utf-8") as f:
        scores = json.load(f)

    if how == "json":
        return scores

    return jsonScoreToDf(scores) 

def loadAllScores(delta:typingDelta,scale:typingScale, modelNames:List[typingModelNames]):
    #summary = loadAllScores("ByDay", "Region", ["polyReg", "ridgeReg", "decTree", "AgaBoostDecTree"])

    dfs = []
    for modelName in modelNames:
        filepath = getScoreResultName(delta, scale, modelName)
        try:
            dfs.append(loadScoreResult(filepath, how="df"))
        except FileNotFoundError:
            print(f"Fichier manquant : {filepath}")

    return pd.concat(dfs, axis=1)
    
def launchScoring(delta:typingDelta,scale:typingScale,modelName:typingModelNames,onlyRegion:str,modelParams:dict,fill:bool,testTrainSepDate:str="2023-01-01"):
    
    pipeLine = getPipeline(modelName)
    combinedDatas = getCombinedDatas(delta,scale,fill)

    scoresInfos = {}
    print(modelParams)
    for combinedData, region in zip(combinedDatas,regions_departements.keys()):
        if onlyRegion and region != onlyRegion:
            continue

        pipeParams = modelParams if onlyRegion else modelParams[region]
        print(region)
        lastCons = pipeParams.get("nbLstCons",3 if delta=="ByDay" else 5)
        paramsCopy = pipeParams.copy()
        if "nbLstCons" in pipeParams:
            del pipeParams["nbLstCons"]
        if 'pca__precision' in pipeParams:
            pipeParams = convertParamGrid(
                region, delta, scale, pipeParams, isGrid=False
            )

        pipeLine.set_params(
            **pipeParams
        )


        combineLastDay = addLastCons(combinedData,lastCons)

        Xtrain,yTrain,Xtest,yTest = getTestTrain(combineLastDay,testTrainSepDate)

        pipeLine.fit(Xtrain,yTrain)

        testScore = pipeLine.score(Xtest,yTest)
        trainScore = pipeLine.score(Xtrain,yTrain)

        scoresInfos[region] = {
                        "model":modelName,
                        "params": paramsCopy,
                        "testScore":testScore ,
                        "trainScore":trainScore,
                    }
        

        print(f"  Test score : {testScore:.4f} - Train score : {trainScore:.4f}")

    
    return scoresInfos



def getPipeline(modelName:typingModelNames) -> Pipeline:
    preprocessPipe = Pipeline([
        ("days",    DaysTransformer(use_hour=True, use_weekDay=True, use_yearDay=True)),
        ("sincos",  SinCosTransformer(use_hour=True, use_weekDay=True, use_yearDay=True)),
        ("scaler", StandardScaler()),
        ("pca",     PcaTransformer(nComp=-1)),
    ])

    LinRegPipe = preprocessPipe.steps.extend([
        ("poly",    PolynomialFeatures(degree=2)),
        ("linReg",  LinearRegression())
    ])

    RidgePipe = preprocessPipe.steps.extend([
        ("poly", PolynomialFeatures(degree=2)),
        ("ridge",Ridge(alpha=1))
    ])

    TreePipe = preprocessPipe.steps.extend([
        ("decTree", DecisionTreeRegressor(max_depth=4)),
    ])

    AdaTreePipe = preprocessPipe.steps.extend([
        ("adaBoostTree", AdaBoostRegressor(DecisionTreeRegressor(max_depth=8), n_estimators=300)),
    ])

    match modelName:
        case "polyReg":
            return LinRegPipe
        case "ridgeReg":
            return RidgePipe
        case "decTree":
            return TreePipe
        case "adaBoostDecTree":
            return AdaTreePipe


def main():
    parser = argparse.ArgumentParser(
        description="Machine Learning Weather Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scoring --delta ByDay --scale Region --model polyReg --testTrainSepDate 2023-01-01
  python main.py gridsearch --delta ByDay --scale Region --model decTree --saveN 1
  python main.py gridsearch --delta ByHour --scale Dep --model ridgeReg --saveN 2 --param-grid pca__precision=[0,0.01]
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    scoringParser = subparsers.add_parser("scoring", help="Launch model scoring")
    scoringParser.add_argument(
        "--delta",
        type=str,
        required=True,
        choices=["ByDay", "ByHour"],
        help="Time delta type"
    )
    scoringParser.add_argument(
        "--scale",
        type=str,
        required=True,
        choices=["Region", "Dep"],
        help="Geographic scale"
    )
    scoringParser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["polyReg", "ridgeReg", "decTree", "adaBoostDecTree"],
        help="Model name"
    )
    scoringParser.add_argument(
        "--region",
        type=str,
        required=False,
        choices=list(regions_departements.keys()),
        default=None,
        help="If specified, will only compute the score for this region, if left empty compute all regions"
    )
    scoringParser.add_argument(
        "--fill",
        type=bool,
        default=False,
        help="Use the filled dataset (only available for ByDay delta)"
    )
    scoringParser.add_argument(
        "--params",
        type=str,
        nargs="*",
        default=[],
        help="Model-specific parameters as key=value pairs. Example: poly__degree=3 ridge__alpha=0.1"
    )
    scoringParser.add_argument(
        "--testTrainSepDate",
        type=str,
        default="2023-01-01",
        help="Date to separate train and test (YYYY-MM-DD format, default: 2023-01-01)"
    )
    scoringParser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="If the file should be safe at the end (default False)"
    )

    scoringParser.set_defaults(func=handleScoring)
    
    # ===== GRIDSEARCH COMMAND =====
    gridsearchParser = subparsers.add_parser("gridsearch", help="Launch grid search")
    gridsearchParser.add_argument(
        "--delta",
        type=str,
        required=True,
        choices=["ByDay", "ByHour"],
        help="Time delta type"
    )
    gridsearchParser.add_argument(
        "--scale",
        type=str,
        required=True,
        choices=["Region", "Dep"],
        help="Geographic scale"
    )
    gridsearchParser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["polyReg", "ridgeReg", "decTree", "adaBoostDecTree"],
        help="Model name"
    )

    gridsearchParser.add_argument(
        "--paramGrid",
        type=str,
        nargs="*",
        default=[],
        help="Override specific parameters from config. Example: ridge__alpha=[0.001,0.01,0.1]"
    )
    gridsearchParser.add_argument(
        "--saveN",
        type=int,
        required=True,
        help="number for saved file"
    )
    gridsearchParser.add_argument(
        "--fill",
        action="store_true",
        default=False,
        help="Use the filled dataset (only available for ByDay delta)"
    )
    gridsearchParser.set_defaults(func=handleGridSearch)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
 
 
def handleScoring(args):
    """
    Handle scoring command.
    """
    print(f"Starting scoring...")
    print(f"  Delta: {args.delta}")
    print(f"  Scale: {args.scale}")
    print(f"  Model: {args.model}")
    print(f"  Test-Train Sep Date: {args.testTrainSepDate}")
    
    # Build model parameters from CLI arguments
    defaultConfigPath = os.path.join("configs",f"defaultParams_{args.delta}_{args.scale}_{args.model}.json")
    try:
        with open(defaultConfigPath, "r", encoding="utf-8") as f:
            modelParams = json.load(f)
            print(f"  Loaded default parameters from: {defaultConfigPath}")
    except FileNotFoundError:
            print(f"Warning: Default config not found at {defaultConfigPath}")
            modelParams = {region:{} for region in regions_departements.keys()}
    if args.region:
        modelParams = modelParams[args.region]

    # Add model-specific parameters (passed as key=value)
    if args.params:
        for regionParam in args.params:
            if not args.region:
                region,param = regionParam.split("___",1)
            else:
                param = regionParam
            try:
                key, value = param.split("=", 1)
                # Try to parse value
                if value.lower() in ["true", "false"]:
                    if args.region:
                        modelParams[key] = value.lower() == "true"
                    else:
                        modelParams[region][key] = value.lower() == "true"
                elif value.lower() == "none":
                    if args.region:
                        modelParams[key] = None
                    else:
                        modelParams[region][key] = None
                  
                else:
                    try:
                        if args.region:
                            modelParams[key] = int(value)
                        else:
                            modelParams[region][key] = int(value)
                        
                    except ValueError:
                        try:
                            if args.region:
                                modelParams[key] = float(value)
                            else:
                                modelParams[region][key] = float(value)
                        except ValueError:
                            if args.region:
                                modelParams[key] = value
                            else:
                                modelParams[region][key] = value
            except ValueError:
                print(f"Error: Invalid parameter format '{param}'. Use format: key=value")
                return
    
    # print(f"  Model Parameters: {modelParams}")
    
    # Call the actual scoring function
    results = launchScoring(
        delta=args.delta,
        scale=args.scale,
        modelName=args.model,
        onlyRegion=args.region,
        modelParams=modelParams,
        fill=args.fill,
        testTrainSepDate=args.testTrainSepDate
    )
    
    print("Scoring completed successfully! \n")
    pprint(results)
    
    if args.save:
        saveScoreResult(getScoreResultName(args.delta,args.scale,args.model),results)
 
def handleGridSearch(args):
    """
    Handle grid search command.
    """
    print(f"Starting grid search...")
    print(f"  Delta: {args.delta}")
    print(f"  Scale: {args.scale}")
    print(f"  Model: {args.model}")
    print(f"  SaveN: {args.saveN}")
    print(f"  Fill: {args.fill}")
    
    # Load default parameter grid for the model
    paramGrid = defaultsSearchParamsGrid[args.model]
    
    
    # Override specific parameters if provided
    if args.paramGrid:
        for param in args.paramGrid:
            try:
                key, value = param.split("=", 1)
                # Parse JSON list
                paramGrid[key] = json.loads(value)
                print(f"  Override: {key} = {paramGrid[key]}")
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Error: Invalid parameter override '{param}': {str(e)}")
                return
    
    # Call the actual grid search function
    results = launchGridSearch(
        delta=args.delta,
        scale=args.scale,
        param_grid=paramGrid,
        modelName=args.model,
        saveN=args.saveN,
        fill=args.fill
    )
    
    print("Grid search completed successfully!")


if __name__ == "__main__":
    main()



