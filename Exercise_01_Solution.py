#!/usr/bin/env python
# coding: utf-8

# # Exercise 01 - Sample Solution

# In[ ]:



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


# In[ ]:


mpath = ""


# ## Exercise 1: Data Preparation

# a) Load the wine dataset in the data folder and produce the following textual descriptions about:<br>
# - Overview of the data, e.g. what it is about? (Hint: Explore and try to find it on the WWW?)
# - The variables, e.g. how many are there, what are the types (quantitative, nominal, ordinal)

# In[ ]:


#df = pd.read_csv("data/wine.csv") # Load wine dataset
df = pd.read_csv(mpath+"data/wine.csv") # Load wine dataset
df.head() # Found out that it is without header


# Information available __[here](https://archive.ics.uci.edu/ml/datasets/wine)__ <br>
# These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines. 
# 
# 1. Alcohol 
# 2. Malic acid 
# 3. Ash 
# 4. Alcalinity of ash 
# 5. Magnesium 
# 6. Total phenols 
# 7. Flavanoids 
# 8. Nonflavanoid phenols 
# 9. Proanthocyanins 
# 10. Color intensity 
# 11. Hue 
# 12. OD280/OD315 of diluted wines 
# 13. Proline

# In[ ]:


#df = pd.read_csv("data/wine.csv", header=None) # Reloading the dataset, now without taking the first row as header
df = pd.read_csv(mpath+"data/wine.csv", header=None) # Reloading the dataset, now without taking the first row as header
df.columns = ['type', 'alcohol', 'malic', 'ash', 'alcalinity', 'magnesium', 'phenols', 'flavanoids', 'nonflavanoids', 'proanthocyanins', 'color', 'hue', 'dilution', 'proline']


# In[ ]:


df.head()


# Quantitative: Alcohol, Malic, Ash,Alcalinity, Magnesium, Phenols, Flavanoids, Nonflavanoids, Proanthocyanins, Color, Hue, Dilution, Proline <br>
# Nominal: Type

# In[ ]:


print(df.shape)
df.describe()


# b)	Are there any unusual in respect to one or more variables?<br>
# Maximum values for the color and to an extent flavanoids are rather high compared to their .75 percentile. Without plotting the distribution (and maybe statistical tests), it is unclear whether such a point is an outlier or not.
# 
# c) Inspect data quality: Are there missing values? Are there values hinting data quality problems?

# In[ ]:


print(len(np.where(df['ash'] < 0)[0]))
print(len(np.where(df['ash'] < 0)[0]) / len(df) * 100)


# Negative values on 30 rows (~17% fo the data) of the ash variables

# d) Remove observations that have missing values or replace missing value by for example the mean value or the most common element.

# In[ ]:


# Get rows with missing value
replace_ix = np.where(df['ash'] < 0)[0]

# Get mean from the rest of the data
clean_ix = np.where(df['ash'] >= 0)[0]
ash_mean = np.mean(df['ash'][clean_ix])

# Replace missing values with the mean
df['ash'][np.where(df['ash'] < 0)[0]] = ash_mean
df.describe()


# ## Exercise 2: Global Exploration
# a)	Distribution analysis: Pick an arbitrary “interesting” variable and:
# - Create a histogram to show the value distribution, and plot the mean and median lines on top of it.
# - What does the histogram tell you, what about the mean and median values, are they a representative measure of central tendency in this particular case?

# In[ ]:


plt.hist(df.flavanoids.values, bins=30)
plt.ylim([0, 25])
plt.axvline(df.flavanoids.mean(), 0, 0.75, color='r', label='Mean')
plt.axvline(df.flavanoids.median(), 0, 0.75, color='b', label='Median')
plt.xlabel("Flavanoids")
plt.ylabel("Counts")
plt.title("Wine Flavanoids Histogram")
plt.legend();


# The mean and median values are quite close to each other, however the distribution could be bimodal, ie. it might be a composition of two distributions. The suspicion of outlier(s) on the high end of this variable value seems to be warranted.<br><br>
# b)	Group analysis: Pick (or create) a second variable to group your first variable with and:
# - Summarize the grouped data by a measure, e.g. mean, median, or standard deviation.
# - Create a bar chart, and write down your observation.

# In[ ]:


# function to create a ordinal variable of alcohol content
def alcohol_content(abv):
    if abv <= 11.5: return 'low'
    elif abv <= 13.5: return 'medium'
    else: return 'high'

df['alcohol_content'] = df.alcohol.apply(lambda x: alcohol_content(x))
df['alcohol_content'].describe()


# In[ ]:


print('Mean: ', df.groupby('alcohol_content').flavanoids.mean(), '\n') # one metric at a time
print(df.groupby('alcohol_content').flavanoids.agg(['mean', 'std', 'median'])) # multiple metrics at once


# In[ ]:


avg_flav = df.groupby('alcohol_content').flavanoids.mean()
avg_flav[['low','medium','high']].plot(kind='barh')
plt.xlabel("Flavanoids")
plt.ylabel("Alcohol Content")
plt.title("Wine Flavanoids by Alcohol Content");


# Wines with medium alcohol content have on average lower amount of flavanoids phenols.

# c)	Relationship analysis: Pick two continuous variables and plot them against each other on a scatter plot. Is there any visible trend?

# In[ ]:


plt.scatter(df.flavanoids, df.alcohol)
plt.xlabel("Flavanoids")
plt.ylabel("Alcohol (%)")
plt.title("Flavanoids - Alcohol (%)");


# There is no clear linear trend. It seems like an V or U-shape. This indicates a quadratic relationship, ie. a quadratic model might be a good fit.

# ## Exercise 3: Group Exploration
# a)	Group comparison: Pick one continuous variable and one categorical variable. Compare the distributions of the continuous variables between each category using one or more of these seaborn functions boxplot, violinplot, stripplot, or swarmplot.

# In[ ]:


sns.set()
sns.boxplot(x="type", y="flavanoids", data=df)
plt.show()
sns.boxplot(data=df, x='type', y='flavanoids')
plt.title('Box Plot')
plt.show()

sns.swarmplot(data=df, x='type', y='flavanoids')
plt.title('Swarm Plot')

plt.show()


# b)	Explore all possible relations using Scatter Plot Matrix: Hint: Use the PairGrid function. What do you see?

# In[ ]:


g = sns.PairGrid(df, vars=df.columns[1:-1].tolist(), hue='type')
g.map_diag(sns.kdeplot)
g.map_offdiag(plt.scatter, s=15)


# Question: Can we separate the classes using the attributes?
# Overall the classes seem to be well-separable using combinations of pairs of attributes and for some attributes on their own; Some combinations of attributes work well to differentiate just a single class from the others.
# All attributes show quite some spread (different values), however, for some combinations of classes and attributes it is rather small. There are few paired values that might be considered outlying.
# 
# Examples:
# 2nd row: red class differs mainly along Y-axis, blue/green along X-axis; this means red has low variation (is fairly independent) of the attribute on the X-axis; blue/green fairly independenent on Y-axis;
# 7th row: classes are fairly well separable; this would hold even for the attribute on the Y-axis on its own, but a second attribute often helps, eg. columns 1 or 6, but not so much columns 3 and 5
# last row: green and red not well-separable; but blue is
# 3rd last row: top green point appears to be relatively far away from other points, but unclear if it is an outlier
# 
# 
# Question: Is the class helpful in modeling the relationship between attributes?
# Sometimes you also notice that there is strongly varying dependence on an attribute given the class that might be exploited for modeling. For example, consider the last row. Say we want to predict proline using color. It is easy to see that for the blue and green classes there seems to be a strong correlation indicating a strong increase in proline given an increase in color. For the red class there is almost no increase in proline given an increase in color.

# c)	Explore the correlations between at least 4 variables by creating a correlation matrix both in table and plot form. What do you notice?

# In[ ]:


g = sns.PairGrid(df, vars=['alcohol', 'alcalinity', 'color', 'hue', 'dilution'], hue='type')
g.map_diag(sns.kdeplot)
g.map_offdiag(plt.scatter, s=15)


# For separability, alcohol and dilution as well as color and dilution seems to work reasonably well, whereas e.g. dilution and hue results in a mix of blue and  green class instances.
# Individual variables as indidacted by the distributions on the diagonal don't allow for an easy separation.
# Some variables seem to be correlated, e.g. hue and dilution, others not, e.g. alcohol and dilution.
# The relationship seems to be class dependent, though this is not always very clear. For instance consider color and alcohol. We see that the color value for the red class increases steeply with an increase in alcohol, whereas it does not so much for the blue class. Thus, it might be worth to fit separate models per class.
# There are few obvious outliers, e.g. for hue/alcohol there is one pont from the green class that is an outlier candidate.
# 

# In[ ]:


df[['alcohol', 'alcalinity', 'color', 'hue', 'dilution']].corr()


# In[ ]:


dpal = sns.choose_colorbrewer_palette('diverging', as_cmap=True)
plt.matshow(df[['alcohol', 'alcalinity', 'color', 'hue', 'dilution']].corr(), cmap=dpal, vmin=-1, vmax=1)
ax = plt.gca()
ax.tick_params(axis='both', which='both',length=0)
ax.grid(False)
plt.title("Correlation Matrix")
plt.xticks(range(5), ['alcohol', 'alcalinity', 'color', 'hue', 'dilution'])
plt.yticks(range(5), ['alcohol', 'alcalinity', 'color', 'hue', 'dilution']);


# In[ ]:



# In[ ]:





# In[ ]:




