
# coding: utf-8

# In[354]:


from jupyterthemes import jtplot 
jtplot.style(theme='solarizedd')


# In[355]:


import os
import tarfile
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[356]:


fetch_housing_data()


# In[357]:


import pandas as pd
def	load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[358]:


load_housing_data()


# In[359]:


housing = load_housing_data()
housing.head()


# In[360]:


housing.info()


# In[361]:


housing["ocean_proximity"].value_counts()


# In[362]:


housing.describe()


# In[363]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
plt.show()


# In[364]:


# Simple split into test/train by random permutation (does not persist beyond dataset updates).
import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")


# In[366]:


# Split into test/train by hash ID so that assignments persist beyond dataset updates
import hashlib
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[-in_test_set], data.loc[in_test_set]


# In[367]:


housing_with_id = housing.reset_index() # adds an 'index' column as the dataset does not have an ID
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[368]:


# If cannot guarantee that existing rows will not be deleted in the future, or
# new data gets appended to the end, then use another feature for ID.
housing_with_id["id"]	=	housing["longitude"]	*	1000	+	housing["latitude"]
train_set,	test_set	=	split_train_test_by_id(housing_with_id,	0.2,	"id")


# In[369]:


# Scikit-Learn's in-built function for dataset test/train splits
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[370]:


# Create an income_category for stratified sampling.
# First, divide the median_income by 1.5 to limit the number of income categories:
limit_cats = housing["median_income"] / 1.5
# Then, round up to have discrete categories.
housing["income_cat"] = np.ceil(limit_cats)
# Finally, merge all categories greater than 5 into category 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[371]:


housing["income_cat"].hist(bins=10, figsize=(4, 3))
plt.show()


# In[372]:


# Now we can stratify based on income category using sklearn built-in
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[373]:


# Check if stratifying worked as expected by checking the 
# income category proportions in the full housing dataset:
housing["income_cat"].value_counts() / len(housing)
# Figure 2-10 shows how the Stratified test set is more representative
# of the income_cat proportions of the Overall data set, whereas
# the Random test set is skewed and not representative at all.


# In[374]:


# Remove the income_cat attribute so the data is back to its original state:
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# In[375]:


# Explore the data: put aside the test set by copying the training set
housing = strat_train_set.copy()


# In[376]:


housing.plot(kind="scatter", x="longitude", y="latitude", color="lightblue")
plt.show()


# In[377]:


# Use alpha=0.1 for easier identification of the high density areas.
housing.plot(kind="scatter", x="longitude", y="latitude", color="lightblue", alpha=0.1)
plt.show()


# In[378]:


# The radius of each circle represents the district's population (arg s), and
# the colour represents the price (arg c).
# Use a predefined colour map (arg cmap) called jet, which ranges from
# blue (low values) to red (high prices):
housing.plot(kind="scatter",	x="longitude",	y="latitude",	alpha=0.4,
             s=housing["population"]/100,	label="population",	figsize=(10,7),
             c="median_house_value",
             cmap=plt.get_cmap("jet"),	colorbar=True)
plt.legend()
plt.show()
# The plot suggests that housing prices are very much related to:
# - location (close to the ocean)
# - population density


# In[379]:


# Standard Correlation Coefficient (aka Pearson's R)
# Recall that this coefficient only measures linear correlations (i.e. misses nonlinear relationships)
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# median_income = 0.69 (close to +1.0) means strong correlation with median_house_value


# In[380]:


# Correlation matrix to visually check for correlation shapes.
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8), color="cyan")
plt.show()
# Other options (than histograms) are available for the main diagonal: see Pandas' docs)


# In[381]:


# median_income shows the most correlation with median_house_value
housing.plot(kind="scatter", x="median_income", y="median_house_value", color="cyan", alpha=0.1)
plt.show()
# - Correlation is obviously strong; seen by upward trend
# - Price cap imposed at data collection is visible as a horizontal line at $500,000
# - Other horizontal lines at $350,000 and $450,000 for unknown reasons (should be investigated)


# In[382]:


# Combining attributes can increase information usefulness.
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
# Re-visit the correlation matrix.
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# This has proven fruitful as the new attributes show stronger correlations over 
# their components suggesting that their information is more useful.


# In[383]:


# Prepare the data for machine learning algorithms.
# Start again with a fresh training set that separates the predictors from the labels.
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[384]:


# Data cleaning
# Fix missing values via three options:
# 1. get rid of the corresponding districts: housing.dropna(subset=["total_bedrooms"])
# 2. get rid of the whole attribute:         housing.drop("total_bedrooms", axis=1)
# 3. set the values to some value (used below):
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)
# The computed median value must be used to also replace missing values in the test set (and new data).


# In[385]:


# Sklean provides a handy class to take care of missing values:
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
# Exclude categorical attribute "ocean_proximity" in order to calculate the median:
housing_numerical = housing.drop("ocean_proximity", axis=1)
# Fit the imputer instance to the training data:
imputer.fit(housing_numerical)
# Imputer simply computes the median of each attribute and stores the result in "statistics_"
# Storing medians for all attributes saves time in case new data has missing values in the future.
imputer.statistics_ # This is equivalent to "housing_numerical.median().values"


# In[386]:


# Transform the training set by replacing missing values by the learned medians.
# The result is a plain numpy array containing the transformed features:
X = imputer.transform(housing_numerical) 
# Put the result back into a Pandas DataFrame:
housing_tr = pd.DataFrame(X, columns=housing_numerical.columns) 


# In[387]:


# Catergorical Attributes: convert text labels to numbers using sklearn LabelEncoder.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded


# In[388]:


print(encoder.classes_) # corresponding numerical mapping.


# In[389]:


# OneHotEncoder will assign a binary attribute to each category so as to 
# prevent an ML algorithm from interpreting relationships based on numerical distance.
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
# fit_transform() expects a 2D array, so housing_cat_encoded (1D array) needs to be reshaped. 
housing_cat_1hot1 = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot1 # Integer categorical values converted into one-hot vectors (SciPy sparse matrix).


# In[390]:


housing_cat_1hot1.toarray() # Convert to a dense NumPy array (wasteful in memory).


# In[391]:


# LabelBinarizer converts from text cats to int cats, then int cats to one-hot vectos in one command.
# Since the book's release, it has been changed:
# It was originally intended only for labels but worked with both X, and y,
# it has since been corrected for its intended use and no longer works with X data.
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot # Returns a dense NumPy array by default


# In[392]:


housing_cat_1hot1.toarray() == housing_cat_1hot


# In[393]:


encoder_sparse = LabelBinarizer(sparse_output=True)
housing_cat_1hot_sparse = encoder_sparse.fit_transform(housing_cat)
housing_cat_1hot_sparse # Returns a sparse SciPy array


# In[394]:


# Custom Transformers
# Need to implement 3 methods: fit(), transform(), and fit_transform() -- see book for details.
# E.g. small transformer class that adds the combined attributes discussed earlier
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    # BaseEstimator gives two extra methods: get_params() and set_params().
    # TransformerMixin gives fit_transform().
    def __init__(self, add_bedrooms_per_room = True): 
        # no *args or ** kargs thanks to BaseEstimator.
        # hyperparam add_bedrooms_per_room allows for easily testing this attributes usefulness.
        # Hyperparams can save time by gating any experimental data prep steps. 
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[395]:


# Scaling/Standardizing features
# Two common methods can be implemented by these Scikit transformers: 
# - MinMaxScaler for normalization; ranging from 0 to 1.
# - StandardScaler for standardization; mean = 0, sd = 1.
#   - No range bounding which may be unacceptable by some algos
#   - Much less affected by outliers than normalization.


# In[396]:


# Transformation Pipelines for sequencing/ordering transformations.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
# The Pipeline constructor takes a list of estimators defining a sequence of steps.
# All but the last must be transformers, i.e. have a fit_transform() method.
# fit() will sequentially call each estimators fit_transform() method, 
# passing the output of each as the param to the next call, until the
# final estimator which gets its fit() method called.
housing_num_tr = num_pipeline.fit_transform(housing_numerical)


# In[397]:


# Since Scikir cannot handle Pandas DataFrames, a custom transformer is needed.
from sklearn.base import BaseEstimator, TransformerMixin

# This class transforms by:
# 1. Selecting the desired attributes and dropping the rest,
# 2. Converting the resulting DataFrame to a Numpy array.
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, raw_column = False):
        self.attribute_names = attribute_names
        self.raw_column = raw_column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.raw_column:
            return X[self.attribute_names]
        else:
            return X[self.attribute_names].values


# In[398]:


# (custom made since LabelBinarizer, used in the book, on X data is deprecated)
class X_Binarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):          
        t = np.transpose(X) # Avoids warning: "column-vector found where 1d array expected"
        le = LabelEncoder().fit_transform(t[0])
        oh = OneHotEncoder().fit_transform(le.reshape(-1,1))
        return oh.toarray()
# This doesn't work as the resulting array is 2 columns short (possibly as y data not being used?)


# In[399]:


# Fix found at stackexchange 
class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)
    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)


# In[400]:


# DataFrameSelector examples: select numerical attributes only, or categorical attributes only.
num_attribs = list(housing_numerical)
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_attribs = ["ocean_proximity"] 
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer_pipes', LabelBinarizerPipelineFriendly()),
    #('x_binarizer', X_Binarizer()),
])
# ('label_binarizer', LabelBinarizer()), this line (found in the book) is now deprecated


# In[401]:


# FeatureUnion
# - concatenates outputs from a list of transformers (which can be entire pipelines)
# - runs list items in parallel.
from sklearn.pipeline import FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])


# In[402]:


housing_prepared = full_pipeline.fit_transform(housing)


# In[403]:


housing_prepared


# In[404]:


housing_prepared.shape


# In[405]:


# Train a Model: Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[406]:


# Try the model on some training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
some_data_prepared.shape
some_data_prepared[0]


# In[407]:


# Try the model on some training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
some_data_prepared[0]


# In[408]:


print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))


# In[409]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse 
# i.e. our model has an average error of $68,628 which isn't good considering that
# most districts' median_housing_values range between $120,000 and $265,000.
# The model appears to have underfit because:
# - the features do not provide enoguh info
# - the model itself is too weak
# Underfitting can be fixed by:
# - feed better features
# - selecting a more powerful model
# - reduce model constraints (which in this case is not possible as linear regression is not regularized)


# In[410]:


# Regression Decision Tree
# Stronger than linear regression, capable of finding complex nonlinear relationships.
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
# The model has overfit (RMSE = 0), need to validate its test performance.


# In[411]:


# Evaluation Using Cross-Validation
# Randomly split training data into 10 distinct subsets (folds).
# The model is trained and evaluated 10 times, each time:
# (1) a unique fold is picked for testing while,
# (2) the other 9 folds are used for training. 
# scikit's cv features expect a utility rather than a cost function resulting in a negative MSE value.
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[412]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[413]:


display_scores(tree_rmse_scores)
# As expected, test score = 71144 is higher than training score = 0.


# In[414]:


# Compute the scores for the previous linear regression example to compare to the decision tree scores.
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, 
                            scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
# Interestingly, it has performed better than decision trees that have overfit so badly.


# In[415]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[416]:


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, 
                            scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
# Random forests have outperformed both previous models by far.


# In[417]:


# Save the model to save time in re-running later
from sklearn.externals import joblib
joblib.dump(forest_reg, "my_model.pkl")         # Save
my_model_loaded = joblib.load("my_model.pkl") # Load
my_model_loaded


# In[418]:


# Grid Search
# Evaluates all possible combinations of hyperparameter values using cross-validation to
# find the best combination.
# When unsure, a simple approach is to try consecutive powers of 10.
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
# In this example grid, it specifies to:
# - first evaluate all 3 x 4 = 12 combinations of n_estimators and max_features values (the first dict)
# - then try all 2 x 3 = 6 combinations in the second dict while bootstrap is false rather than true (default)
# - All in all, 12 + 6 = 18 combinations of hyperparameter values are tried, each five times (five-fold cv)
# - This may take a while.


# In[419]:


forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)


# In[420]:


# The result can be used to give the best combination of hyperparameter values.
grid_search.best_params_
# The best values are found to also be the maximum's chosen,
# therefore it would be a good idea to repeat the process with higher possible values.


# In[421]:


# The best estimator is available directly (which can be saved as a model to save time)
grid_search.best_estimator_
# If GridSearchCV is initialized with refit=True (the default), then once it finds the best estimator
# using cv, it RETRAINS it on the whole training set. This is usually a good idea since feeding it more
# data will likely improve its performance.


# In[422]:


# Evaluation scores (RMSE) produced by each combination.
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
# The best params give the best RMSE = 49977


# In[423]:


# Randomized Search
# If the hyperparamter search space is large, this method is preferable.
# This class can be used in much the same way as GridSearchCV, but instead of trying out all possible
# combinations, it evaluates a given number of random combinations by selecting a random value for 
# each hyperparameter at every iteration. This approach has two main benefits:
# - Letting the search run for 1000 iterations will explore 1000 different values for each hyperparameter
#   instead of just a few values per hyperparameter with the grid search approach.
# - You have more control over the computing budget you want to allocate to hyperparameter search,
#   simply by setting the number of iterations.


# In[424]:


# Analyze the Best Models and Their Errors
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[425]:


# Display these importance scores next to their corresponding attribute names
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
# We can use this info to drop some of the less useful features.
# You should also look at the specific errors that your system makes, then try to understand
# why it makes them and what could fix the problem (adding extra features or cleaning up outliers, etc.)


# In[426]:


# Evaluate on the Test Set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test) # NOT fit_transform()
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,766.0
final_rmse
# Expect the performance to be slightly worse than what was measured in the previous cv stage.
# Resist the temptation to tweak hyperparameters to improve the final_rmse as it will likely 
# not generalize to new data.


# In[427]:


# Present your solution
# - highlight what you have learned 
# - what worked and what did not
# - what assumptions were made
# - what your system's limitations are
# - document everything
# - create nice presentations with clear visualizations and easy-to-remember statements 
#   (e.g. "the median income is the number one predictor of housing prices")

