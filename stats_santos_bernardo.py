# %% [markdown]
# ## **Santos and Bernardo 2025 supplementary material**
# 
# **Authors:** Felipe Pinto dos Santos and Danilo Vicensotto Bernardo<br>
# **Year:** 2025<br>
# **Software:** Jupiter notebook<br>
# **Programming language:** Python
# 
# 
# #### **Abstract:** This Supplementary material contain the analytical procedures of the study "Evolutionary implications of violence in complex societies in the central Andean region, a bioarchaeological analysis" a chapter of the volume "Routledge Handbook of the Archeology of Violence in the America. The main purpose of this analysis is to compare trauma incidence in central Peru across three archaeological periods of Andean societies: the Early Intermediate Period (EIP), the Middle Horizon (MH), and the Late Intermediate Period (LIP). Based on the results, this study discusses the possible application of the theoretical framework of Cultural Multilevel Selection in the context of Andean history.
# 

# %% [markdown]
# ## 1. Data Loading and Preparation
# 
# ### 1.1 Importing Libraries and Initial Loading

# %% [markdown]
# The first step is importing the necessary libraries and loading the dataset. The following code demonstrates the import of essential libraries such as `numpy` for numerical operations, `pandas` for data manipulation, `statsmodels` for statistical modeling, and `matplotlib.pyplot` for visualization.

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
import matplotlib.pyplot as plt 

# === 1) load data ===
file_path = "tr_data_andescentral.xlsx"  
df = pd.read_excel(file_path)
df.head() # show the structure of the data (coluns and rows)

# %% [markdown]
# ### 1.2 Variable Selection
# 
# In the following cell, the variables of interesting was select, that is  `n_id` and `n_id_affect`. This two columns are extracted into separate variables. This step is crucial for focusing on relevant data and simplifying future operations.

# %%
# select variables for analyses

n_id = df['n_id']
n_id_affect = df['n_id_affect']



# %% [markdown]
# ### 1.3 Descriptive Statistics
# 
# To understand the distribution and basic characteristics of the variables, descriptive statistics are calculated. The `describe()` method of pandas provides a statistical summary, including count, mean, standard deviation, minimum and maximum values, and quartiles.

# %%
subset = df[['n_id','n_id_affect']] # select coluns 
print(subset.describe())

# %% [markdown]
# These values provide an overview of the centrality, dispersion, and shape of the distributions of the `n_id` and `n_id_affect` variables.
# 

# %% [markdown]
# ### 1.4 Visualization: Histograms
# 
# Histograms are powerful visual tools for understanding the distribution of a single variable. The following code generates histograms for `n_id` and `n_id_affect`, allowing observation of the the total number of individual observed and the count events of truma. 

# %%

# Ploting histogram of variables: 

# Figure and subplots configuration
fig, axs = plt.subplots(2, 1, figsize=(6, 8))

# Plotting the first histogram
axs[0].hist(n_id, bins='auto')
axs[0].set_title('Histogram of Observed')

# Plotting the second histogram
axs[1].hist(n_id_affect, bins='auto')
axs[1].set_title('Histogram of Absolute Number of Affected by Trauma')

# Adjusting spacing between subplots
plt.tight_layout()

# Displaying the figure
plt.show()

# %% [markdown]
# Observing the plots, it is notable that the values of both variables are concentrated at the lower values at the x-axis, with higher values more dispersed. Visually, this type of distribution is very similar to a Poisson or Negative Binomial distribution 

# %% [markdown]
# ### 1.5 Relative Frequency per Period
# 
# To analyze the relationship between trauma occurrence and the periods, the relative frequency per period was calculated. This procedure involves grouping the data by period and calculating the proportion of `n_id_affect` (the number of individuals affected) relative to `n_id` (the number of individuals observed).

# %%
# Calculating the relative frequency per period
fr = df[['period', 'n_id', 'n_id_affect']]
fr = fr.fillna(0)

# Grouping and calculating sums
sum = fr.groupby('period').sum()
fr_result = sum['n_id_affect'] / sum['n_id']

# Displaying the result
print(fr_result)

# %% [markdown]
# These values indicate that the relative frequency of individuals affected by trauma in the LIP period was higher than in the previous one, as expected based on the assumptions of this study. In the following code cell, a bar chart was generated to visualize the differences in relative frequency by period.

# %%
# Ploting relative frequence per period: 

import matplotlib.pyplot as plt

# Example data (replace with your actual data)
period = ['EIP', 'MH', 'LIP']
fr_result = [0.165, 0.23, 0.375]

fig, ax = plt.subplots()

# Plot bars in black with adjusted width of 0.6
ax.bar(period, fr_result, color='black', width=0.6)

# Increase the maximum y-axis value to 0.50
ax.set_ylim(0, 0.50)

# Label and title configuration
ax.set_xlabel('Period')
ax.set_ylabel('Frequency (%)')
plt.title('Frequency by Period')

# Convert decimal values to percentages on the y-axis
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0%}".format(x)))

# Display the bar chart
plt.show()

# %% [markdown]
# The bar chart shows that the relative frequency of trauma tended to be higher in the more recent periods.

# %% [markdown]
# ## 2. Statistical Modeling: GLM with Negative Binomial Distribution
# 

# %% [markdown]
# ### 2.1 Data Preparation for Modeling
# 
# Before fitting a statistical model, the data was prepared. The relevant variables was selected, missing values was handeling , and variables was transforming, such as creating an `offset` for count models. An offset term, defined as the natural logarithm of the total individuals observed (`n_id`), is included in the model to adjust the analysis of trauma counts (`n_id_affect`) by accounting for differences in exposure (`n_id`). This procedure adjusts for unequal sample sizes, ensuring that trauma counts are modeled as rates relative to the number of individuals observed rather than as raw counts. In practical terms, this prevents samples of different sizes from being treated as equally reliable, since larger samples provide more precise information about trauma rates than smaller ones 

# %%
# Select the variables of interest 
data = df[['period', 'n_id', 'n_id_affect']].copy()
data['n_id'] = pd.to_numeric(data['n_id'], errors='coerce')
data['n_id_affect'] = pd.to_numeric(data['n_id_affect'], errors='coerce')

# drop NA data
data = data.dropna(subset=['period','n_id','n_id_affect']).copy()

# avoid log(0)
data = data[data['n_id'] > 0].copy()
data['offset'] = np.log(data['n_id']) # define n_id as offset

# Define the MH period as baseline for analyses 
data['period'] = pd.Categorical(data['period'], categories=['MH','EIP','LIP'], ordered=True)


# %% [markdown]
# The steps above ensure that the data are in the correct format and that problematic values (such as log(0)) are properly handled, in addition to defining a reference category for the period variable. In this analysis, the Middle Horizon (MH) is used as the reference.

# %% [markdown]
# ### 2.2 Mean and Variance Analysis
# 
# For count models, it is important to check the relationship between the mean and variance of the data. In Poisson distributions, the mean equals the variance. Significant deviations indicate the need for more flexible models, such as the Negative Binomial.
# 

# %%
y = data["n_id_affect"]

media = np.mean(y)
variancia = np.var(y, ddof=1)

print("Média:", media)
print("Variância:", variancia)
print("Razão Var/Média:", variancia/media)


# %% [markdown]
# The results above indicate that the variance/Mean ratio is significantly greater than 1 indicates overdispersion, justifying the use of a Negative Binomial model.

# %% [markdown]
# ### 2.3 Estimation of `mu` via Poisson Model
# 
# For a better application of the Negative Binomial distribution model, a dispersion parameter (`alpha`) was calculated. Before estimating this value, a Poisson model is fitted to obtain the mu values (expected means), which are then used in the formula to estimate alpha

# %%

# fórmula do modelo (mantendo a comparação por período)
formula = "n_id_affect ~ C(period)"

# === 2) estimar mu via Poisson (mesma fórmula e offset) ===
pois = smf.glm(
    formula=formula,
    data=data,
    family=sm.families.Poisson(),
    offset=data['offset']
).fit()

mu = pois.fittedvalues.values
y  = data['n_id_affect'].astype(float).values
print(mu,y)

# %% [markdown]
# The mu values represent the expected means of `n_id_affect` under the Poisson model, taking into account `period` and `offset`. It is possible to see the differences between the values: the observed values contain higher numbers compared to the predicted ones. In the cell below, a histogram comparing the two sets of values is plotted:

# %%
import matplotlib.pyplot as plt

# Histogram of observed values
plt.hist(y, bins='auto', alpha=0.6, label="Observed (y)")

# Histogram of predicted values (mu)
plt.hist(mu, bins='auto', alpha=0.6, label="Predicted (mu)")

plt.xlabel("Count values")
plt.ylabel("Frequency")
plt.title("Observed vs Predicted Counts")
plt.legend()
plt.show()

# %% [markdown]
# Visually, it is possible to perceive that the observed data have much more dispersion than the predicted Poisson values. This reinforces the use of a Negative Binomial distribution

# %% [markdown]
# ### 2.4 Estimation of the Dispersion Parameter (`alpha`) for NB2
# 
# The `alpha` parameter in the Negative Binomial (NB2) distribution is crucial for modeling overdispersion. It is estimated using the method of moments, which compares the observed variance with the expected variance under the Poisson model.

# %%

# === 3) método dos momentos para alpha (NB2): Var(Y) ≈ mu + alpha*mu^2 ===
num = ((y - mu)**2).sum() - mu.sum()
den = (mu**2).sum()
alpha_hat = max(num / den, 0.0)

print(f"alpha_hat (NB2, com period + offset) = {alpha_hat:.6f}")


# %% [markdown]
# A value of `alpha_hat` greater than zero confirms the presence of overdispersion and the suitability of the Negative Binomial model.

# %% [markdown]
# ### 2.5 Negative Binomial Model Fitting and Comparison with Poisson
# 
# With the estimated `alpha_hat`, the GLM model with Negative Binomial distribution is fitted. Subsequently, a comparison with the Poisson model is performed using the likelihood ratio test (LR stat) to assess whether the inclusion of the dispersion parameter significantly improves the model fit.

# %%
from scipy import stats
import statsmodels.api as sm


# Ajustar modelo NB com alpha estimado
nb = sm.GLM(y, pois.model.exog, 
            family=sm.families.NegativeBinomial(alpha=alpha_hat), 
            offset=data['offset']).fit()

# Comparar log-likelihoods
ll_pois = pois.llf
ll_nb   = nb.llf

LR = 2 * (ll_nb - ll_pois)
p_value = stats.chi2.sf(LR, df=1)  # 1 parâmetro extra: alpha

print(f"LR stat: {LR:.4f}, p-valor: {p_value:.4f}")


# %% [markdown]
# A very low p-value (close to zero) indicates that the Negative Binomial model fits the data significantly better than the Poisson model, confirming the importance of considering overdispersion.

# %% [markdown]
# ### 2.7 AIC Comparison
# 
# Akaike Information Criterion (AIC) is a measure of the relative quality of statistical models for a given set of data. Models with lower AIC are generally preferred. The comparison of AIC between Poisson and Negative Binomial models reinforces the choice of the most suitable model:

# %%
print("Poisson AIC:", pois.aic)
print("NB AIC:", nb.aic)

# %% [markdown]
# The significantly lower AIC for the Negative Binomial model confirms your relevance aplication over the Poisson model.

# %% [markdown]
# ### 2.8 GLM NB Model Summary and Diagnostics
# 
# The summary aplying below of the GLM Negative Binomial model provides a comprehensive overview of the fitting results, including coefficients, standard errors, p-values, and other statistics. In addition, important diagnostics such as log-likelihood, AIC, BIC, and Pearson Chi2/df are calculated to assess the quality of the fit and the presence of remaining overdispersion.

# %%

# === 4) fit the GLM NB with alpha_hat and caculate the model results ===
nb = smf.glm(
    formula=formula,
    data=data,
    family=sm.families.NegativeBinomial(alpha=alpha_hat),
    offset=data['offset']
).fit()

print(nb.summary())

# model parameters 
loglik = nb.llf
aic    = nb.aic

# Pearson Chi2/df (residual dispersion check)
pearson_chi2_df = (nb.resid_pearson**2).sum() / nb.df_resid

print("\n=== Diagnósticos do GLM NB (com alpha_hat) ===")
print(f"logLik : {loglik:.4f}")
print(f"AIC    : {aic:.4f}")
print(f"Pearson Chi2/df: {pearson_chi2_df:.4f}")


# %% [markdown]
# The model summary above and diagnostics provide detailed information about the significance of predictors and the overall model fit.

# %% [markdown]
# ### 2.9 Effect Size Table (IRR)
# 
# For count models, coefficients are often interpreted as Incidence Rate Ratios (IRR). The IRR table and its 95% confidence intervals was calculated to facilitate the interpretation of the effects of predictor variables.

# %%
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

# %%
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

# %% [markdown]
# These results show that the comparison between the reference period (C(period) = MH period) and the EIP does not indicate a statistical difference, which contradicts the initial expectation. However, the comparison with the LIP period does reveal such a difference, confirming the assumption of this study that the LIP period concentrated a higher incidence of trauma than the previous one. Only the second comparison aligns with the hypotheses derived from cultural multilevel selection regarding the role of warfare and conflict in the evolution of complex societies.

# %% [markdown]
# ## 3. Trauma Type Analysis comparation 
# 
# ### 3.1 Selection and Aggregation of Trauma Variables
# 
# To compare trauma types (antemortem and perimortem) across periods, corresponding variables was selected and values aggregated (summed).
# 

# %%
#selecting variables 

group_eip_anti = df.loc[df['period'] =='EIP', 'n_id_antimortem']
eip_anti = group_eip_anti.dropna() 
group_mh_anti = df.loc[df['period'] =='MH', 'n_id_antimortem']
mh_anti = group_mh_anti.dropna()
group_lip_anti = df.loc[df['period'] =='LIP', 'n_id_antimortem']
lip_anti = group_lip_anti.dropna()

group_eip_peri = df.loc[df['period'] =='EIP', 'n_id_perimortem']
eip_peri = group_eip_peri.dropna()
group_mh_peri = df.loc[df['period'] =='MH', 'n_id_perimortem']
mh_peri = group_mh_peri.dropna()
group_lip_peri = df.loc[df['period'] =='LIP', 'n_id_perimortem']
lip_peri = group_lip_peri.dropna()

# %% [markdown]
# 
# These steps below prepare the data for visualization and comparison of total trauma by period and type:

# %%
sum_eip_anti = eip_anti.sum()
sum_mh_anti = mh_anti.sum()
sum_lip_anti = lip_anti.sum()

sum_eip_peri = eip_peri.sum()
sum_mh_peri = mh_peri.sum()
sum_lip_peri = lip_peri.sum()

# %% [markdown]
# ### 3.2 Visualization: Sum of Antemortem Trauma by Period
# 
# A bar chart is used to visualize the sum of antemortem traumas in each period, facilitating the identification of trends or differences.

# %%
import matplotlib.pyplot as plt

# Example data (replace with your actual data)
period = ['EIP', 'MH', 'LIP']
total = [sum_eip_anti, sum_mh_anti, sum_lip_anti]

fig, ax = plt.subplots()

# Plot bars in black with adjusted width of 0.6
ax.bar(period, total, color='black', width=0.6)

# Increase the maximum y-axis value to 120
ax.set_ylim(0, 120)

# Label and title configuration
ax.set_xlabel('Period')
ax.set_ylabel('Sum')
plt.title('Sum antimortem by Period')

# Display the bar chart
plt.show()

# %% [markdown]
# The graph above shows that the MH and LIP periods have very similar values of antemortem trauma. The EIP concentrates very few cases, and since this information is limited in this period, your role in the comparison is not relevant

# %% [markdown]
# ### 3.3 Visualization: Sum of Perimortem Trauma by Period
# 
# Similarly, a bar chart is generated for the sum of perimortem traumas by period, allowing for a visual comparison between trauma types.

# %%
import matplotlib.pyplot as plt

# Example data (replace with your actual data)
period = ['MH', 'LIP']
total = [sum_mh_peri, sum_lip_peri]

fig, ax = plt.subplots()

# Plot bars in black with adjusted width of 0.6
ax.bar(period, total, color='black', width=0.6)

# Increase the maximum y-axis value 50
ax.set_ylim(0, 50)

# Label and title configuration
ax.set_xlabel('Period')
ax.set_ylabel('Sum')
plt.title('Sum perimortem by Period')

# Display the bar chart
plt.show()

# %% [markdown]
# The comparison shows that the LIP period has much higher levels of perimortem trauma than the MH, confirming the expectation that this period was significantly more lethal than the previous one.


