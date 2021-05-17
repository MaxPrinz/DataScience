import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl #this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
import seaborn as sns #sets up styles and gives us more plotting options
#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

import warnings
warnings.filterwarnings('ignore') #hide warning

df = pd.read_csv("data/wine.csv") # Load wine dataset
df.head() # Found out that it is without header

#df = pd.read_csv("data/wine.csv", header=None) # Reloading the dataset, now without taking the first row as header
df = pd.read_csv("data/wine.csv", header=None) # Reloading the dataset, now without taking the first row as header
df.columns = ['type', 'alcohol', 'malic', 'ash', 'alcalinity', 'magnesium', 'phenols', 'flavanoids', 'nonflavanoids', 'proanthocyanins', 'color', 'hue', 'dilution', 'proline']

#%% md
print(df.head())

print(df.shape)
print(df.describe())

#%% md
print(len(np.where(df['ash'] < 0)[0]))
print(len(np.where(df['ash'] < 0)[0]) / len(df) * 100)

# Get rows with missing value
replace_ix = np.where(df['ash'] < 0)[0]

# Get mean from the rest of the data
clean_ix = np.where(df['ash'] >= 0)[0]
ash_mean = np.mean(df['ash'][clean_ix])

# Replace missing values with the mean
df['ash'][np.where(df['ash'] < 0)[0]] = ash_mean
df.describe()