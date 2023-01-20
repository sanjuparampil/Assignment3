import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import numpy as np

col = [0, 1, 34, 44, 54, 63]


def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t-1950))
    return f


def norm(array):
    """ Returns array normalised to [0,1]."""
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled


def norm_df(df, first=0, last=None):
    """
        Returns all columns of the dataframe normalised to [0,1] with the
        exception of the first (containing the names)
        Calls function norm to do the normalisation of one column, but
        doing all in one function is also fine.
        First, last: columns from first to last (including) are normalised.
        Defaulted to all. None is the empty entry. The default corresponds"""
# iterate over all numerical columns
    for col in df.columns[first:last]:  # excluding the first column
        df[col] = norm(df[col])
    return df


# reading the file and data pre-processing
df_GDP = pd.read_csv("agri.csv")
data_GDP = df_GDP.drop(
    columns=["Country Code", "Indicator Name", "Indicator Code"])
data_GDP = data_GDP.replace(np.nan, 0)
countries = ["India", "Colombia", "Japan"]
data_GDP = data_GDP["Country Name"].isin(countries)
data_GDP = df_GDP[data_GDP]
data_GDP = data_GDP.drop(
    columns={"Country Name", "Country Code", "Indicator Name", "Indicator Code"})
GDP_tr = np.transpose(data_GDP)
GDP_tr = GDP_tr.reset_index()
GDP_tr = GDP_tr.rename(columns={"index": "year"})
GDP_tr = GDP_tr.rename(columns={109: "INDIA", 45: "COLOMBIA", 119: "JAPAN"})
GDP_tr = GDP_tr.dropna()
GDP_tr["JAPAN"] = pd.to_numeric(GDP_tr["JAPAN"])
GDP_tr["INDIA"] = pd.to_numeric(GDP_tr["INDIA"])
GDP_tr["year"] = pd.to_numeric(GDP_tr["year"])

# fit exponential growth
popt, covar = opt.curve_fit(exp_growth, GDP_tr["year"], GDP_tr["JAPAN"])

print("Fit parameter", popt)
# use *popt to pass on the fit parameters
GDP_tr["pop_exp"] = exp_growth(GDP_tr["year"], *popt)
plt.figure()
plt.plot(GDP_tr["year"], GDP_tr["JAPAN"], label="data")
plt.plot(GDP_tr["year"], GDP_tr["pop_exp"], label="fit")
plt.legend()
plt.title("Fit")
plt.xlabel("year")
plt.ylabel("Agriculture methane emmision")
plt.show()

# growth of 0.02 gives a reasonable start value
popt = [4e8, 0.01]
GDP_tr["pop_exp"] = exp_growth(GDP_tr["year"], *popt)
plt.figure()
plt.plot(GDP_tr["year"], GDP_tr["JAPAN"], label="data")
plt.plot(GDP_tr["year"], GDP_tr["JAPAN"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("Agriculture methane emmision")
plt.title("Improved")
plt.show()

# # fit exponential growth
popt, covar = opt.curve_fit(
    exp_growth, GDP_tr["year"], GDP_tr["JAPAN"], p0=[4e8, 0.02])
# # much better
print("Fit parameter", popt)
GDP_tr["pop_exp"] = exp_growth(GDP_tr["year"], *popt)
plt.figure()
plt.plot(GDP_tr["year"], GDP_tr["JAPAN"], label="data")
plt.plot(GDP_tr["year"], GDP_tr["pop_exp"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("Agriculture methane emission")
plt.title("Graph showing exponential fit")
plt.savefig('exponential.png', bbox_inches="tight", dpi=300)
plt.show()
print()


print(GDP_tr.describe())


pd.plotting.scatter_matrix(GDP_tr, figsize=(9.0, 9.0))
#to avoid overlap of labels
plt.tight_layout()
plt.show()


# extract columns for fitting
df_fit = GDP_tr[["INDIA", "JAPAN"]].copy()
# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_fit to affect df_fish. This make the plots with the
# original measurements
df_fit = norm_df(df_fit)
print(df_fit.describe())
print()


for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)

# extract labels and calculate silhoutte score
labels = kmeans.labels_
print(ic, skmet.silhouette_score(df_fit, labels))


# Plot for 6 clusters
kmeans = cluster.KMeans(n_clusters=6)
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
plt.scatter(df_fit["INDIA"], df_fit["JAPAN"], c=labels, cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for j in range(4):
    cx, cy = cen[j, :]
    plt.plot(cx, cy, "dk", markersize=10)

plt.xlabel("country")
plt.ylabel("y")
plt.title("6 clusters")
plt.savefig('6 cluster.png', bbox_inches="tight", dpi=300)
plt.show()
#-----------------------

# Plot for five clusters
kmeans = cluster.KMeans(n_clusters=5)
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
plt.scatter(df_fit["INDIA"], df_fit["JAPAN"], c=labels, cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for i in range(5):
    cx, cy = cen[i, :]

plt.plot(cx, cy, "dk", markersize=10)
plt.xlabel("country")
plt.ylabel("y")
plt.title("5 clusters")
plt.savefig('5 cluster.png', bbox_inches="tight", dpi=300)
plt.show()



