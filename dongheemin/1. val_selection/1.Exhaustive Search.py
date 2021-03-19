import pandas as pd
import statsmodels.formula.api as smf
import time
import itertools

#The function of calculating AIC by data
def processSubset(data, outcome, strr):
    # model = smf.ols(y, X[list(feature_set)])
    model = smf.ols(formula = outcome+" ~ "+strr, data = data)
    regr = model.fit()
    AIC = regr.aic
    return {"model" : regr, "AIC" : AIC, "Feature" : strr}

#The function of get Best AIC Model by all of combinations
def getBest(data, outcome, k):
    tic = time.time()
    results = []
    for combo in itertools.combinations(data.columns.difference([outcome]), k):

        combo = (list(combo)+['const'])
        strr = ""
        count = 1
        for column in combo:
            strr = strr + column
            count = count + 1
            if count < len(combo):
                strr = strr + "+"
            else:
                break

        results.append(processSubset(data=data, outcome=outcome, strr=strr))
    models = pd.DataFrame(results)
    best_model = models.loc[models['AIC'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on ", k, "predictors in ", (toc-tic),"seconds")

    return best_model

#Load Dataset
data = pd.read_csv('../0. dataset/diabetes3.csv')
#Set Target Column(Outcome Column]
Target = "DE1_dg"

#Make DataFrame for save result
models = pd.DataFrame(columns=["AIC", "model", "Feature"])

#Set StartTime
tic = time.time()

for i in range(1, len(data.columns.difference([Target]))+1):
    models.loc[i] = getBest(data, outcome=Target, k=i)
toc = time.time()
print("Total elapsed time:", (toc-tic),"Seconds")

for i in range(1, len(data.columns.difference([Target]))+1):
    print(models.loc[i, "AIC"])
    print(models.loc[i, "Feature"].split("+"))

print(models.shape)