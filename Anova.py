import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib as plt

df = pd.read_csv("d:\data\FuelConsumptionCo2.csv")

df.boxplot('FUELCONSUMPTION_COMB', by='CYLINDERS')

plt.show()
