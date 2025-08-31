# Python Data Analysis Report

**Author:** Manus AI

This report details the Python data analysis procedures, as extracted from the provided HTML file. The objective is to explain each step of the process, from data loading to statistical modeling and visualization.

## 1. Data Loading and Preparation

### 1.1 Importing Libraries and Initial Loading

The first step in any data analysis is importing the necessary libraries and loading the dataset. The following code demonstrates the import of essential libraries such as `numpy` for numerical operations, `pandas` for data manipulation, `statsmodels` for statistical modeling, and `matplotlib.pyplot` for visualization.

```python
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# === 1) load data ===
file_path = "tr_data_andescentral.xlsx"  
df = pd.read_excel(file_path)
df.head()
```

In this snippet, the `tr_data_andescentral.xlsx` file is loaded into a pandas DataFrame (`df`). The `df.head()` function is used to display the first few rows of the DataFrame, allowing for a quick inspection of the data's structure and content.

### 1.2 Variable Selection

After initial loading, it is common to select the variables of interest for the analysis. In the example, the `n_id` and `n_id_affect` columns are extracted into separate variables. This step is crucial for focusing on relevant data and simplifying future operations.

```python
# select variables for analyses

n_id = df["n_id"]
n_id_affect = df["n_id_affect"]
```

### 1.3 Descriptive Statistics

To understand the distribution and basic characteristics of the variables, descriptive statistics are calculated. The `describe()` method of pandas provides a statistical summary, including count, mean, standard deviation, minimum and maximum values, and quartiles.

```python
### descriptive statistics 

subset = df[["n_id", "n_id_affect"]]
print(subset.describe())
```

The output of this code block, as per the HTML, is:

```
n_id  n_id_affect
count  95.000000    95.000000
mean   20.431579     6.000000
std    18.876222     9.187179
min     1.000000     0.000000
25%     6.000000     1.000000
50%    14.000000     3.000000
75%    29.000000     6.000000
max    77.000000    52.000000
```

These values provide an overview of the centrality, dispersion, and shape of the distributions of the `n_id` and `n_id_affect` variables.

### 1.4 Visualization: Histograms

Histograms are powerful visual tools for understanding the distribution of a single variable. The following code generates histograms for `n_id` and `n_id_affect`, allowing observation of the frequency of different values.

```python
# Ploting histogram of variables: 

# Figure and subplots configuration
fig, axs = plt.subplots(2, 1, figsize=(6, 8))

# Plotting the first histogram
axs[0].hist(n_id, bins=\'auto\')
axs[0].set_title(\'Histogram of Observed\')

# Plotting the second histogram
axs[1].hist(n_id_affect, bins=\'auto\')
axs[1].set_title(\'Histogram of Absolute Number of Affected by Trauma\')

# Adjusting spacing between subplots
plt.tight_layout()

# Displaying the figure
plt.show()
```

This visualization helps identify patterns, such as skewness or multiple peaks, in the variable distributions.

### 1.5 Relative Frequency per Period

To analyze the relationship between variables and time, the relative frequency per period is calculated. This procedure involves grouping the data by `period` and calculating the proportion of `n_id_affect` relative to `n_id`.

```python
# Calculating the relative frequency per period
fr = df[["period", "n_id", "n_id_affect"]]
fr = fr.fillna(0)

# Grouping and calculating sums
sum = fr.groupby("period").sum()
fr_result = sum["n_id_affect"] / sum["n_id"]

# Displaying the result
print(fr_result)
```

The output of this code block, as per the HTML, is:

```
period
EIP    0.165829
LIP    0.375685
MH     0.234017
dtype: float64
```

These values indicate the relative frequency of individuals affected by trauma in each period (`EIP`, `LIP`, `MH`).

### 1.6 Visualization: Relative Frequency per Period

To visualize the relative frequency per period, a bar chart is generated. This facilitates comparison between different periods.

```python
# Ploting relative frequence per period: 

import matplotlib.pyplot as plt

# Example data (replace with your actual data)
period = [\'EIP\', \'MH\', \'LIP\']
fr_result = [0.165, 0.23, 0.375]

fig, ax = plt.subplots()

# Plot bars in black with adjusted width of 0.6
ax.bar(period, fr_result, color=\'black\', width=0.6)

# Increase the maximum y-axis value to 0.50
ax.set_ylim(0, 0.50)

# Label and title configuration
ax.set_xlabel(\'Period\')
ax.set_ylabel(\'Frequency (%)\' )
plt.title(\'Frequency by Period\')

# Convert decimal values to percentages on the y-axis
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: \"{:.0%}\".format(x)))

# Display the bar chart
plt.show()
```

This chart provides a clear representation of the differences in relative frequencies between periods.

## 2. Statistical Modeling: GLM with Negative Binomial Distribution

### 2.1 Data Preparation for Modeling

Before fitting a statistical model, the data needs to be prepared. This includes selecting relevant variables, handling missing values, and transforming variables, such as creating an `offset` for count models.

```python
# Select the variables of interest 
data = df[["period", "n_id", "n_id_affect"]].copy()
data["n_id"] = pd.to_numeric(data["n_id"], errors=\'coerce\')
data["n_id_affect"] = pd.to_numeric(data["n_id_affect"], errors=\'coerce\')

# drop NA data
data = data.dropna(subset=["period","n_id","n_id_affect"]).copy()
```

```python
# avoid log(0)
data = data[data["n_id"] > 0].copy()
data["offset"] = np.log(data["n_id"])
```

```python
# Define the MH period as baseline for analyses 
data["period"] = pd.Categorical(data["period"], categories=["MH","EIP","LIP"], ordered=True)
```

These steps ensure that the data is in the correct format and that problematic values (like `log(0)`) are handled, in addition to defining a reference category for the `period` variable.

### 2.2 Mean and Variance Analysis

For count models, it is important to check the relationship between the mean and variance of the data. In Poisson distributions, the mean equals the variance. Significant deviations indicate the need for more flexible models, such as the Negative Binomial.

```python
###########################################

import numpy as np

y = data["n_id_affect"]

media = np.mean(y)
variancia = np.var(y, ddof=1)

print("Mean:", media)
print("Variance:", variancia)
print("Var/Mean Ratio:", variancia/media)
```

The output of this code block, as per the HTML, is:

```
Mean: 6.0
Variance: 84.40425531914893
Var/Mean Ratio: 14.067375886524822
```

A Variance/Mean ratio significantly greater than 1 indicates overdispersion, justifying the use of a Negative Binomial model.

### 2.3 Estimation of `mu` via Poisson Model

Before estimating the dispersion parameter (`alpha`) for the Negative Binomial model, a Poisson model is fitted to obtain the `mu` values (expected mean), which are used in the `alpha` estimation formula.

```python
# model formula (maintaining period comparison)
formula = "n_id_affect ~ C(period)"

# === 2) estimate mu via Poisson (same formula and offset) ===
pois = smf.glm(
    formula=formula,
    data=data,
    family=sm.families.Poisson(),
    offset=data["offset"]
).fit()

mu = pois.fittedvalues.values
y  = data["n_id_affect"].astype(float).values
print(mu,y)
```

The output of this code block, as per the HTML, is:

```
[ 0.16582915  0.66331658  0.99497487  2.34016888  2.57418577  0.70205066
  3.27623643  5.8504222   1.17008444  3.74427021  2.34016888  3.04221954
  9.12665862  9.12665862  6.08443908  6.76232202  4.13253012  2.57418577
  1.17008444  5.63526835  3.75684556  7.13800657  6.38663746  0.37568456
 23.66812705 27.04928806 10.14348302  6.0109529   1.12705367 15.21109771
  7.95657419  4.97487437  5.14070352 10.51916758 20.28696605  3.38116101
 12.39759036 18.78422782  5.25958379  3.75684556  2.25410734  1.12705367
  1.32663317  0.66331658  1.65829146  6.46733668  3.9798995   1.65829146
  0.33165829  1.16080402  2.57418577  1.17008444  2.80820265  4.68033776
  0.70205066  0.70205066  1.40410133  1.40410133  7.02050663  5.14837153
  1.17008444  0.93606755  0.23401689 12.40289505 11.23281062 14.74306393
  1.98994975  1.8241206  10.29674306  8.42460796  6.55247286  8.42460796
  1.8721351   1.8721351   0.23401689  0.46803378  3.74427021  3.74427021
  0.93606755  1.17008444  1.63811821  5.25958379 22.54107338 16.15443593
  9.76779847  4.13253012  5.63526835  5.25958379  6.0109529  28.92771084
 23.2924425  22.54107338  3.38116101  1.87842278  8.26506024] [ 1.  2.  0.  5.  7.  0.  4.  6.  0.  4.  5.  4. 11. 16.  5. 12.  9.  2.
  0. 12.  5. 11. 11.  1. 36. 45.  3. 11.  3. 52. 15.  6.  3. 10. 37.  4.
  8. 20.  6.  3.  4.  0.  2.  2.  4.  4.  3.  1.  0.  0.  3.  1.  3.  1.
  1.  0.  3.  1.  7.  5.  4.  1.  1.  0.  2.  2.  4.  1.  3.  6.  3.  2.
  3.  1.  0.  1.  3.  0.  0.  1.  0.  2.  3.  2. 15.  4.  3.  7.  6. 30.
  8.  9.  1.  0.  2.]
```

The `mu` values represent the expected means of `n_id_affect` under the Poisson model, considering `period` and `offset`.

### 2.4 Estimation of the Dispersion Parameter (`alpha`) for NB2

The `alpha` parameter in the Negative Binomial (NB2) distribution is crucial for modeling overdispersion. It is estimated using the method of moments, which compares the observed variance with the expected variance under the Poisson model.

```python
# === 3) method of moments for alpha (NB2): Var(Y) \u2248 mu + alpha*mu^2 ===
num = ((y - mu)**2).sum() - mu.sum()
den = (mu**2).sum()
alpha_hat = max(num / den, 0.0)

print(f"alpha_hat (NB2, with period + offset) = {alpha_hat:.6f}")
```

The output of this code block, as per the HTML, is:

```
alpha_hat (NB2, with period + offset) = 0.488932
```

A value of `alpha_hat` greater than zero confirms the presence of overdispersion and the suitability of the Negative Binomial model.

### 2.5 Negative Binomial Model Fitting and Comparison with Poisson

With the estimated `alpha_hat`, the GLM model with Negative Binomial distribution is fitted. Subsequently, a comparison with the Poisson model is performed using the likelihood ratio test (LR stat) to assess whether the inclusion of the dispersion parameter significantly improves the model fit.

```python
from scipy import stats
import statsmodels.api as sm


# Fit NB model with estimated alpha
nb = sm.GLM(y, pois.model.exog, \
            family=sm.families.NegativeBinomial(alpha=alpha_hat), \
            offset=data["offset"]).fit()

# Compare log-likelihoods
ll_pois = pois.llf
ll_nb   = nb.llf

LR = 2 * (ll_nb - ll_pois)
p_value = stats.chi2.sf(LR, df=1)  # 1 extra parameter: alpha

print(f"LR stat: {LR:.4f}, p-value: {p_value:.4f}")
```

The output of this code block, as per the HTML, is:

```
LR stat: 158.8971, p-value: 0.0000
```

A very low p-value (close to zero) indicates that the Negative Binomial model fits the data significantly better than the Poisson model, confirming the importance of considering overdispersion.

### 2.6 Variance and Mean Analysis (Review)

This code block reiterates the check of the relationship between variance and mean, reinforcing the need for the Negative Binomial model.

```python
resid_var = np.var(y - mu, ddof=1)
print("Var(y):", np.var(y, ddof=1))
print("Mean(y):", np.mean(y))
print("Var/Mean (Poisson should be ~1):", np.var(y, ddof=1) / np.mean(y))
```

The output of this code block, as per the HTML, is:

```
Var(y): 84.40425531914893
Mean(y): 6.0
Var/Mean (Poisson should be ~1): 14.067375886524822
```

### 2.7 AIC Comparison

Akaike Information Criterion (AIC) is a measure of the relative quality of statistical models for a given set of data. Models with lower AIC are generally preferred. The comparison of AIC between Poisson and Negative Binomial models reinforces the choice of the most suitable model.

```python
print("Poisson AIC:", pois.aic)
print("NB AIC:", nb.aic)
```

The output of this code block, as per the HTML, is:

```
Poisson AIC: 625.2790553399149
NB AIC: 466.3820037026725
```

The significantly lower AIC for the Negative Binomial model confirms its superiority over the Poisson model.

### 2.8 GLM NB Model Summary and Diagnostics

The summary of the GLM Negative Binomial model provides a comprehensive overview of the fitting results, including coefficients, standard errors, p-values, and other statistics. In addition, important diagnostics such as log-likelihood, AIC, BIC, and Pearson Chi2/df are calculated to assess the quality of the fit and the presence of remaining overdispersion.

```python
# === 4) refit GLM NB with alpha_hat and calculate diagnostics ===
nb = smf.glm(
    formula=formula,
    data=data,
    family=sm.families.NegativeBinomial(alpha=alpha_hat),
    offset=data["offset"]
).fit()

print(nb.summary())

# main diagnostics
loglik = nb.llf
aic    = nb.aic
bic    = nb.bic if hasattr(nb, "bic") else (2*nb.df_model - 2*loglik)  # fallback if version doesn't have .bic

# Pearson Chi2/df (check for remaining dispersion)
pearson_chi2_df = (nb.resid_pearson**2).sum() / nb.df_resid

print("\n=== GLM NB Diagnostics (with alpha_hat) ===")
print(f"logLik : {loglik:.4f}")
print(f"AIC    : {aic:.4f}")
print(f"BIC    : {bic:.4f}")
print(f"Pearson Chi2/df: {pearson_chi2_df:.4f}")
```

The output of this code block, as per the HTML, is:

```
Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:            n_id_affect   No. Observations:                   95
Model:                            GLM   Df Residuals:                       92
Model Family:        NegativeBinomial   Df Model:                            2
Link Function:                    Log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -230.19
Date:                Fri, 29 Aug 2025   Deviance:                       91.689
Time:                        01:41:53   Pearson chi2:                     73.1
No. Iterations:                     7   Pseudo R-squ. (CS):             0.1002
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept           -1.4401      0.139    -10.334      0.000      -1.713      -1.167
C(period)[T.EIP]    -0.2225      0.300     -0.741      0.459      -0.811       0.366
C(period)[T.LIP]     0.5235      0.196      2.676      0.007       0.140       0.907
====================================================================================

=== GLM NB Diagnostics (with alpha_hat) ===
logLik : -230.1910
AIC    : 466.3820
BIC    : -327.2672
Pearson Chi2/df: 0.7940
```

The model summary and diagnostics provide detailed information about the significance of predictors and the overall model fit.

### 2.9 Effect Size Table (IRR)

For count models, coefficients are often interpreted as Incidence Rate Ratios (IRR). The IRR table and its 95% confidence intervals are calculated to facilitate the interpretation of the effects of predictor variables.

```python
# ==== 3) Effect sizes table: IRR = exp(coef) with 95% CI ====
coef = nb.params
se   = nb.bse

irr = np.exp(coef)
lcl = np.exp(coef - 1.96*se)
ucl = np.exp(coef + 1.96*se)
pvl = nb.pvalues

irr_tbl = pd.DataFrame({
    "Term": coef.index,
    "IRR": irr.values,
    "CI95_low": lcl.values,
    "CI95_high": ucl.values,
    "p_value": pvl.values
})
```

```python
# Map labels for readability
label_map = {
    "Intercept": "Intercept (MH)",
    "C(period)[T.EIP]": "EIP vs MH",
    "C(period)[T.LIP]": "LIP vs MH",
    
}
irr_tbl["Comparison"] = irr_tbl["Term"].map(label_map).fillna(irr_tbl["Term"])

cols = ["Comparison", "IRR", "CI95_low", "CI95_high", "p_value"]
irr_tbl = irr_tbl[cols]

# Print rounded summary
print("\n=== Effect sizes (IRR) and 95% CI ===")
print(irr_tbl.round({"IRR":3, "CI95_low":3, "CI95_high":3, "p_value":3}).to_string(index=False))
```

The output of this code block, as per the HTML, is:

```
=== Effect sizes (IRR) and 95% CI ===
    Comparison   IRR  CI95_low  CI95_high  p_value
Intercept (MH) 0.237     0.180      0.311    0.000
     EIP vs MH 0.801     0.444      1.442    0.459
     LIP vs MH 1.688     1.150      2.477    0.007
```

This table is fundamental for interpreting the impact of each period on the incidence rate of `n_id_affect`.

## 3. Trauma Type Analysis

### 3.1 Selection and Aggregation of Trauma Variables

To compare trauma types (antemortem and perimortem) across periods, corresponding variables are selected and values are aggregated (summed).

```python
#Comparing antimortem and perimortem trauma types

#selecting variables 

group_eip_anti = df.loc[df["period"] ==\'EIP\', \'n_id_antimortem\']
eip_anti = group_eip_anti.dropna() 
group_mh_anti = df.loc[df["period"] ==\'MH\', \'n_id_antimortem\']
mh_anti = group_mh_anti.dropna()
group_lip_anti = df.loc[df["period"] ==\'LIP\', \'n_id_antimortem\']
lip_anti = group_lip_anti.dropna()

group_eip_peri = df.loc[df["period"] ==\'EIP\', \'n_id_perimortem\']
eip_peri = group_eip_peri.dropna()
group_mh_peri = df.loc[df["period"] ==\'MH\', \'n_id_perimortem\']
mh_peri = group_mh_peri.dropna()
group_lip_peri = df.loc[df["period"] ==\'LIP\', \'n_id_perimortem\']
lip_peri = group_lip_peri.dropna()
```

```python
sum_eip_anti = eip_anti.sum()
sum_mh_anti = mh_anti.sum()
sum_lip_anti = lip_anti.sum()

sum_eip_peri = eip_peri.sum()
sum_mh_peri = mh_peri.sum()
sum_lip_peri = lip_peri.sum()
```

These steps prepare the data for visualization and comparison of total trauma by period and type.

### 3.2 Visualization: Sum of Antemortem Trauma by Period

A bar chart is used to visualize the sum of antemortem traumas in each period, facilitating the identification of trends or differences.

```python
import matplotlib.pyplot as plt

# Example data (replace with your actual data)
period = [\'EIP\', \'MH\', \'LIP\']
total = [sum_eip_anti, sum_mh_anti, sum_lip_anti]

fig, ax = plt.subplots()

# Plot bars in black with adjusted width of 0.6
ax.bar(period, total, color=\'black\', width=0.6)

# Increase the maximum y-axis value to 120
ax.set_ylim(0, 120)

# Label and title configuration
ax.set_xlabel(\'Period\')
ax.set_ylabel(\'Sum\')
plt.title(\'Sum antimortem by Period\')

# Display the bar chart
plt.show()
```

### 3.3 Visualization: Sum of Perimortem Trauma by Period

Similarly, a bar chart is generated for the sum of perimortem traumas by period, allowing for a visual comparison between trauma types.

```python
import matplotlib.pyplot as plt

# Example data (replace with your actual data)
period = [\'MH\', \'LIP\']
total = [sum_mh_peri, sum_lip_peri]

fig, ax = plt.subplots()

# Plot bars in black with adjusted width of 0.6
ax.bar(period, total, color=\'black\', width=0.6)

# Increase the maximum y-axis value 50
ax.set_ylim(0, 50)

# Label and title configuration
ax.set_xlabel(\'Period\')
ax.set_ylabel(\'Sum\')
plt.title(\'Sum perimortem by Period\')

# Display the bar chart
plt.show()
```

These charts provide insights into the distribution of trauma types across periods, complementing the statistical analysis.

## Conclusion

This report detailed the Python data analysis procedures, covering everything from data preparation and exploration to advanced statistical modeling with GLM Negative Binomial and result visualization. The analysis demonstrated the importance of considering overdispersion in count data and how different periods can influence trauma incidence. Visualizations complemented the statistical analysis, providing clear insights into distributions and comparisons.

This document serves as a guide to understanding the steps and decisions made during the data analysis process, as presented in the original HTML file.


