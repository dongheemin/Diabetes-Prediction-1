import pandas as pd
import statsmodels.formula.api as smf

data = pd.read_csv('../0. dataset/diabetes2.csv')
strr = ""
count = 0
for column in data:
    strr = strr+column
    count = count+1
    if count != 11:
        strr = strr+"+"
    else:
        break

model = smf.ols(formula = "DE1_dg ~ "+strr, data = data)
result = model.fit()

print(result.summary())