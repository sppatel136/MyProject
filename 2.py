import streamlit as st
import pandas as pd 
from plotly.graph_objs import Pie
from plotly.graph_objs import Pie
import pickle
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from plotly.graph_objs import Figure

def introduction():

	st.title("World Temperature Data Analysis")
	st.write("### Introduction")
	st.info("This study aims to leverage machine learning for analyzing historical temperature data and forecasting global temperature changes to support climate change mitigation efforts.")

	st.write("### Objectives")
	st.write("- Analyze temperature anomaly data to identify global patterns")
	st.write("- Investigate relationships between temperature and factors like GDP, population,CO2")
	st.write("- Develop predictive models for forecasting temperature changes")

	st.markdown("""
	<style>
		.right-align {
			text-align: right;
		}
	</style>
	""", unsafe_allow_html=True)

	st.markdown('<p class="right-align">Presenter: Sweta Patel</p>', unsafe_allow_html=True)
	st.markdown('<p class="right-align">Date: May 24, 2024</p>', unsafe_allow_html=True)

def data_exploration():
	st.title("Data Exploration")
	st.header("Data Sources")

	# Load OWID data
	owid_df = pd.read_csv("owid.csv")
	"""This project utilizes two main datasets:"""
	with st.expander("**OWID Data**"):
		"""
	   - Contains data on CO2, gas emissions, energy, GDP, population, and other related metrics across countries and years.
	   - Has 48,058 entries and 79 columns.
	   - Freely available online.
		"""
		st.write("Preview of the OWID File:")
		st.dataframe(owid_df.head())
		"""
	
		- Many columns have a high percentage of missing values."""
		missing_percentage = (owid_df.isnull().sum() / len(owid_df)) * 100
		missing_percentage = missing_percentage.round(2)
		missing_info_df = pd.DataFrame({'Column Name': missing_percentage.index, 'Missing Percentage': missing_percentage.values})
		st.write("Missing Percentage Information:")
		st.write(missing_info_df)

	# Load Surface Temperature Anomaly data
	temp_df = pd.read_csv("surface-temperature.csv")
	with st.expander("**Surface Temperature Anomaly Data**"):
		"""
		- Contains data on surface temperature anomalies for different countries over time.
		- Has 29,566 entries and 4 columns.
		- Freely available online.
		"""
		st.write("Preview of the Surface Temperature Anomaly File:")
		st.dataframe(temp_df.head())
		"""
		 **Surface Temperature Anomaly File**:
		
		- No missing values for the critical "Surface Temperature Anomaly" column.
		
		"""

	st.header("Data Merging and Cleaning")
	"""
	- Combine OWID and Surface Temperature datasets.
	- Clean the data by removing rows with missing values.
	- Enrich the dataset by adding continent information."""

	data =pd.read_csv("perfect_data.csv")
	with st.expander("**Cleaned Data**"):
	   
		st.write("Preview of the Cleaned Data:")
		st.dataframe(data.head())

		"Exploring Variables"
		"""
		**Population:**
		Total population of the studied area or region, reflecting demographic trends and the scale of human activities.

		**Gross Domestic Product (GDP):**
		GDP measures the economic output of the region over a specific period, providing insight into economic activity and prosperity.
		"""

		
		"""
		**Nitrous Oxide:**
		Total emissions of nitrous oxide, a potent greenhouse gas contributing to climate change.

		**Nitrous Oxide per Capita:**
		Average emissions of nitrous oxide per person, indicating individual-level contributions to greenhouse gas emissions.
		"""

		"""
		**Energy per Capita:**
		Energy consumption per person, reflecting average energy usage within the population.

		**Energy per GDP:**
		Energy intensity of GDP, indicating the energy efficiency of economic activities.

		**Primary Energy Consumption:**
		Total energy consumption before any transformation or conversion, providing an overview of energy usage patterns.
		"""

		"""
		**Temperature Change from CH4, GHG, N2O, CO2:**
		Contributions of methane, greenhouse gases, nitrous oxide, and carbon dioxide emissions to changes in temperature, aiding in understanding the impact of different emissions on climate change.
		"""

		"""
		**Total GHG:**
		Total greenhouse gas emissions, encompassing all gases contributing to the greenhouse effect.

		**GHG per Capita:**
		Average greenhouse gas emissions per person, indicating individual-level carbon footprint.

		**GHG Excluding LUCF per Capita:**
		Greenhouse gas emissions per capita excluding emissions from land use change and forestry.

		**Total GHG Excluding LUCF:**
		Total greenhouse gas emissions excluding emissions from land use change and forestry, providing a more focused perspective on human-induced emissions.
		"""

		"""
		**Trade CO2:**
		CO2 emissions associated with trade activities, highlighting the carbon footprint of global trade.

		**Trade CO2 Share:**
		Proportion of total CO2 emissions attributed to trade activities, indicating the significance of trade in carbon emissions.
		"""
		"""
		**Surface Temperature Anomaly:**
		Deviation from the average surface temperature, providing insights into temperature variations and climate anomalies.
		"""

		"""
		**CO2, Cement CO2, Coal CO2, Flaring CO2, Gas CO2, Land Use Change CO2, Oil CO2, Other Industry CO2:**
		Emissions of carbon dioxide from various sectors, including cement production, coal combustion, gas combustion, flaring, oil combustion, land use change, and other industrial sources.
		"""

		"""
		**Methane:**
		Total emissions of methane, another potent greenhouse gas.

		**Methane per Capita:**
		Average emissions of methane per person, providing insight into individual-level contributions to greenhouse gas emissions.
		"""

def visualization():
   
	st.header("Visualization of Surface Temperature Anomaly")
	  
	data =pd.read_csv("perfect_data.csv") 
	data['year'] = pd.to_datetime(data['year'], format='%Y').dt.year.apply(lambda x: '{:0}'.format(x))

	group_continent_year = data.groupby(['Continent', 'year'])['Surface temperature anomaly'].mean().reset_index()
	group_year_temp = data.groupby('year')['Surface temperature anomaly'].mean().reset_index()

	pivot_data = group_continent_year.pivot(index='year', columns='Continent', values='Surface temperature anomaly')
	pivot_data = pivot_data.round(2)
	pivot_data = pivot_data.sort_index()
	
	def color_map(val):
		if pd.isnull(val):
			return 'color: white'
		color = int((val - pivot_data.min().min()) / (pivot_data.max().max() - pivot_data.min().min()) * 255)
		return f'background-color: rgb({255 - color}, {255 - color}, 255)'

	# year and temp    
	st.subheader("Surface Temperature Anomaly Over the Years")
	st.line_chart(group_year_temp.set_index('year'))

	# continents vs year
	st.subheader(" Surface Temperature Anomaly Over the Years - All Continents")
	group_continent_year = group_continent_year.set_index(['year', 'Continent'])
	st.line_chart(group_continent_year['Surface temperature anomaly'].unstack(['Continent']))
	
	# Display the table
	st.subheader(" Surface Temperature Anomaly by Year and Continent")
	st.dataframe(pivot_data.style.applymap(color_map).format('{:.2f}', na_rep='-'))

	selected_continent = st.selectbox("Select Continent", data['Continent'].unique())
	filtered_data = data[data['Continent'] == selected_continent]
	group_filter_country_year = filtered_data.groupby(['country', 'year'])['Surface temperature anomaly'].mean().reset_index()
	group_filter_country_year = group_filter_country_year.set_index(['year', 'country'])
	
	# Contries for selected continent
	st.subheader(f"Surface Temperature Anomaly Over the Years - {selected_continent}")
	st.line_chart(group_filter_country_year['Surface temperature anomaly'].unstack('country'))
	
	# Display the table
	st.subheader(" Surface Temperature Anomaly by Country and Year")
	pivot_data_selected = group_filter_country_year.reset_index().pivot(index='year', columns='country', values='Surface temperature anomaly')
	st.dataframe(pivot_data_selected.style.applymap(lambda val: color_map(val), subset=pivot_data_selected.columns).format('{:.2f}', na_rep='-'))

def dashboard():
	
	st.header("Dashboard")
	
	data =pd.read_csv("perfect_data.csv") 
	year_range = st.sidebar.slider("Select Year Range", min(data["year"]), max(data["year"]), (min(data["year"]), max(data["year"])), 1)
	Continent = st.sidebar.multiselect("Select Continent", data["Continent"].unique())
	filtered_data = data[(data["year"].between(year_range[0], year_range[1])) & (data["Continent"].isin(Continent))]

  

   
	#cement_co2 coal_co2 flaring_co2 gas_co2 land_use_change_co2 oil_co2 other_industry_co2
	specific_cols = ["cement_co2", "coal_co2", "flaring_co2", "gas_co2", "land_use_change_co2", "oil_co2", "other_industry_co2"]
	st.subheader("CO2 Emissions Breakdown")
	filtered_data["total_co2"] = filtered_data[specific_cols].sum(axis=1)
	pie_chart_data = filtered_data[specific_cols]
	pie_chart = [Pie(labels=specific_cols, values=pie_chart_data.sum())]  # Wrap in a list
	st.plotly_chart(pie_chart)

	filtered_data["total_co2"] = filtered_data[specific_cols].sum(axis=1)

	# Create bar chart data
	bar_chart_data = filtered_data[["year", "total_co2"]]  # Select year and total_co2 columns

	# Display bar chart with Streamlit
	st.subheader("Total CO2 Emissions Over the Years")
	st.bar_chart(bar_chart_data.set_index('year'), use_container_width=True)

	#Surface temperature anomaly and year
	group_temp_year = filtered_data.groupby('year')['Surface temperature anomaly'].mean().reset_index()
	st.subheader("Surface Temperature Anomaly Over the Years")
	st.bar_chart(group_temp_year.set_index('year'),use_container_width=True)

	#primary energy consuption
	st.subheader("Primary Energy Consuption Over the Years")
	group_energy_year = filtered_data.groupby('year')['primary_energy_consumption'].mean().reset_index()
	st.bar_chart(group_energy_year.set_index('year'), use_container_width=True)
	
	#GDP
	st.subheader("GDP over the Years")
	group_gdp_year = filtered_data.groupby('year')['gdp'].mean().reset_index()
	st.bar_chart(group_gdp_year.set_index('year'), use_container_width=True)

	# Population
	st.subheader("Population Over the Years")
	group_pop_year = filtered_data.groupby('year')['population'].mean().reset_index()
	st.bar_chart(group_pop_year.set_index('year'), use_container_width=True)

	#methane vs methane_per_capita
	#nitrous_oxide vs nitrous_oxide_per_capita
	#energy_per_capita vs energy_per_gdp
	
	#temperature_change_from_ch4 vs temperature_change_from_ghg vs temperature_change_from_n2o vs temperature_change_from_co2

	

	# # Create two columns for parallel line graphs
	# col1, col2 = st.columns(2)

	# with col1:
	# 	st.subheader("Temperature Change from GHG")
		

	# with col2:
	# 	st.subheader("Surface Temperature Anomaly")

def conclusion():
	st.header("Conclusion")

	st.info("The Random Forest Regressor demonstrates superior performance compared to the Linear Regression model in this case. Random Forest Regressor is more capable of generalizing to unseen data and making accurate predictions.  The model's test set accuracy (R-squared of 0.4495) might seem low, it's important to consider the complexity of predicting global temperatures.")
	
	"""	- Data quality limitations, especially missing values, presented significant challenges. """
	
	""" - I want to express my  gratitude to the entire team and a special thank you to my mentor, Tarik, for his support throughout this project."""

	""" - Moving forward, I believe further research with higher quality datasets can lead to even more robust models for climate change prediction."""
def prediction():
	def load_model():
		with open('rfrm_2.pkl', 'rb') as model_file:
			model = pickle.load(model_file)
		return model

	def get_features(year, nitrous_oxide_per_capita, co2_growth_prct, methane_per_capita, co2_growth_abs, consumption_co2_per_gdp, energy_per_gdp, gas_co2_per_capita, trade_co2_share, land_use_change_co2_per_capita):
		features = np.array([year, nitrous_oxide_per_capita, co2_growth_prct, methane_per_capita, co2_growth_abs, consumption_co2_per_gdp, energy_per_gdp, gas_co2_per_capita, trade_co2_share, land_use_change_co2_per_capita])
		return features.reshape(1, -1)

	def predict_surface_temperature(features):
		model = load_model()
		prediction = model.predict(features)
		return np.round(prediction, 3)

	st.header("Prediction")
	st.subheader('Prediction Simulation with Random Forest Regressor')

	# Load the DataFrame
	data = pd.read_csv("perfect_data.csv")
	df3 = data.copy()

	# Get the minimum and maximum values for each feature from the DataFrame
	year_min, year_max = df3['year'].min(), df3['year'].max()
	nitrous_oxide_per_capita_min, nitrous_oxide_per_capita_max = df3['nitrous_oxide_per_capita'].min(), df3['nitrous_oxide_per_capita'].max()
	co2_growth_prct_min, co2_growth_prct_max = df3['co2_growth_prct'].min(), df3['co2_growth_prct'].max()
	methane_per_capita_min, methane_per_capita_max = df3['methane_per_capita'].min(), df3['methane_per_capita'].max()
	co2_growth_abs_min, co2_growth_abs_max = df3['co2_growth_abs'].min(), df3['co2_growth_abs'].max()
	consumption_co2_per_gdp_min, consumption_co2_per_gdp_max = df3['consumption_co2_per_gdp'].min(), df3['consumption_co2_per_gdp'].max()
	energy_per_gdp_min, energy_per_gdp_max = df3['energy_per_gdp'].min(), df3['energy_per_gdp'].max()
	gas_co2_per_capita_min, gas_co2_per_capita_max = df3['gas_co2_per_capita'].min(), df3['gas_co2_per_capita'].max()
	trade_co2_share_min, trade_co2_share_max = df3['trade_co2_share'].min(), df3['trade_co2_share'].max()
	land_use_change_co2_per_capita_min, land_use_change_co2_per_capita_max = df3['land_use_change_co2_per_capita'].min(), df3['land_use_change_co2_per_capita'].max()

	year_value = df3['year'].max()
	nitrous_oxide_per_capita_value = df3['nitrous_oxide_per_capita'].mean()
	co2_growth_prct_value = df3['co2_growth_prct'].mean()
	methane_per_capita_value = df3['methane_per_capita'].mean()
	co2_growth_abs_value = df3['co2_growth_abs'].mean()
	consumption_co2_per_gdp_value = df3['consumption_co2_per_gdp'].mean()
	energy_per_gdp_value = df3['energy_per_gdp'].mean()
	gas_co2_per_capita_value = df3['gas_co2_per_capita'].mean()
	trade_co2_share_value = df3['trade_co2_share'].mean()
	land_use_change_co2_per_capita_value = df3['land_use_change_co2_per_capita'].mean()

	# Feature inputs
	col1, col2 = st.columns(2)

	with col1:
		year = st.slider("Year", min_value=int(year_min), max_value=int(year_max), step=1,value=year_value)
		nitrous_oxide_per_capita = st.slider("Nitrous Oxide Per Capita", min_value=float(nitrous_oxide_per_capita_min), max_value=float(nitrous_oxide_per_capita_max), value= nitrous_oxide_per_capita_value)
		co2_growth_prct = st.slider("CO2 Growth Percentage", min_value=float(co2_growth_prct_min), max_value=float(co2_growth_prct_max), value=co2_growth_prct_value)
		methane_per_capita = st.slider("Methane Per Capita", min_value=float(methane_per_capita_min), max_value=float(methane_per_capita_max), value=methane_per_capita_value)
		co2_growth_abs = st.slider("CO2 Growth Absolute", min_value=float(co2_growth_abs_min), max_value=float(co2_growth_abs_max), value=co2_growth_abs_value)

	with col2:
		consumption_co2_per_gdp = st.slider("Consumption CO2 Per GDP", min_value=float(consumption_co2_per_gdp_min), max_value=float(consumption_co2_per_gdp_max), value=consumption_co2_per_gdp_value)
		energy_per_gdp = st.slider("Energy Per GDP", min_value=float(energy_per_gdp_min), max_value=float(energy_per_gdp_max), value=energy_per_gdp_value)
		gas_co2_per_capita = st.slider("Gas CO2 Per Capita", min_value=float(gas_co2_per_capita_min), max_value=float(gas_co2_per_capita_max), value=gas_co2_per_capita_value)
		trade_co2_share = st.slider("Trade CO2 Share", min_value=float(trade_co2_share_min), max_value=float(trade_co2_share_max), value=trade_co2_share_value)
		land_use_change_co2_per_capita = st.slider("Land Use Change CO2 Per Capita", min_value=float(land_use_change_co2_per_capita_min), max_value=float(land_use_change_co2_per_capita_max), value=land_use_change_co2_per_capita_value)

	# Add a button for prediction
	if st.button("Predict"):
		features = get_features(year, nitrous_oxide_per_capita, co2_growth_prct, methane_per_capita, co2_growth_abs, consumption_co2_per_gdp, energy_per_gdp, gas_co2_per_capita, trade_co2_share, land_use_change_co2_per_capita)
		prediction = predict_surface_temperature(features)
		st.write(f"The prediction of the surface temperature anomaly is: {prediction[0]}")

def modeling():
	st.title("Models")
	a()
	b()
def a():
	st.header("Random Forest Regressor Model")

	# Load data
	df3 = pd.read_csv("perfect_data.csv")

	# Encoding categorical columns
	df3['ID_country'] = pd.factorize(df3['country'])[0]
	df3['ID_iso_code'] = pd.factorize(df3['iso_code'])[0]
	df3['ID_continent'] = pd.factorize(df3['Continent'])[0]
	df3.drop(['country', 'iso_code', 'Continent'], axis=1, inplace=True)

	# Split data into features and target
	X = df3.drop(['Surface temperature anomaly'], axis=1)
	y = df3['Surface temperature anomaly']

	# Split data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Initialize and train the model
	rfrm = RandomForestRegressor(max_depth=15, random_state=42)
	rfrm.fit(X_train, y_train)

	# Make predictions
	y_train_pred = rfrm.predict(X_train)
	y_test_pred = rfrm.predict(X_test)
	y_pred = rfrm.predict(X)

	# Calculate evaluation metrics
	r2_train = r2_score(y_train, y_train_pred)
	r2_test = r2_score(y_test, y_test_pred)
	mae_train = mean_absolute_error(y_train, y_train_pred)
	mse_train = mean_squared_error(y_train, y_train_pred)
	rmse_train = np.sqrt(mse_train)
	mae_test = mean_absolute_error(y_test, y_test_pred)
	mse_test = mean_squared_error(y_test, y_test_pred)
	rmse_test = np.sqrt(mse_test)

	# Display model results
	st.write("Random Forest Regressor Model Results:")
	st.write(f"R^2 (Train): {r2_train:.4f}")
	st.write(f"R^2 (Test): {r2_test:.4f}")
	st.write(f"MAE Train: {mae_train:.4f}")
	st.write(f"MSE Train: {mse_train:.4f}")
	st.write(f"RMSE Train: {rmse_train:.4f}")
	st.write(f"MAE Test: {mae_test:.4f}")
	st.write(f"MSE Test: {mse_test:.4f}")
	st.write(f"RMSE Test: {rmse_test:.4f}")

	# Plot feature importances
	feature_importances = rfrm.feature_importances_
	sorted_indices = np.argsort(feature_importances)[::-1]
	sorted_features = X.columns[sorted_indices]
	sorted_importances = feature_importances[sorted_indices]

	fig, ax = plt.subplots(figsize=(5, 4))
	ax.barh(sorted_features[:10], sorted_importances[:10])
	ax.set_xlabel('Importance')
	ax.set_title('Top 10 Feature Importances')
	ax.invert_yaxis()
	st.pyplot(fig)

	# Calculate residuals
	residuals = y_test - y_test_pred

	# Create a 2x2 subplot
	fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

	# Residuals dispersion plot
	sns.scatterplot(x=y_test_pred, y=residuals, ax=axes[0, 0])
	axes[0, 0].axhline(y=0, color='r', linestyle='--')
	axes[0, 0].set_title("Residuals Dispersion Plot")
	axes[0, 0].set_xlabel("Predictions")
	axes[0, 0].set_ylabel("Residuals")
	axes[0, 0].grid(True)

	# Histogram of residuals
	sns.histplot(residuals, kde=True, ax=axes[0, 1])
	axes[0, 1].set_title("Histogram of Residuals")
	axes[0, 1].set_xlabel("Residuals")
	axes[0, 1].set_ylabel("Frequency")
	axes[0, 1].grid(True)

	# Actual vs. predicted values
	sns.scatterplot(x=y_test, y=y_test_pred, ax=axes[1, 0])
	axes[1, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
	axes[1, 0].set_title("Comparison of Actual vs. Predicted Values")
	axes[1, 0].set_xlabel("Actual Values")
	axes[1, 0].set_ylabel("Predictions")
	axes[1, 0].grid(True)

	# Q-Q plot of residuals
	stats.probplot(residuals, plot=axes[1, 1])
	axes[1, 1].set_title("QQ Plot of Residuals")
	axes[1, 1].set_xlabel("Theoretical Quantiles")
	axes[1, 1].set_ylabel("Sample Quantiles")
	axes[1, 1].grid(True)

	# Adjust layout to prevent overlap
	plt.tight_layout()

	# Show the plot
	st.pyplot(fig)
def b():
	st.header("Linear Regression Model")	
   # Load data
	df3 = pd.read_csv("perfect_data.csv")

	# Encoding categorical columns
	df3['ID_country'] = pd.factorize(df3['country'])[0]
	df3['ID_iso_code'] = pd.factorize(df3['iso_code'])[0]
	df3['ID_continent'] = pd.factorize(df3['Continent'])[0]
	df3.drop(['country', 'iso_code', 'Continent'], axis=1, inplace=True)

	X = df3.drop(['Surface temperature anomaly'], axis=1)
	y = df3['Surface temperature anomaly']

    # Split data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
	LRM = LinearRegression()
	LRM.fit(X_train, y_train)

	y_train_pred = LRM.predict(X_train)
	y_test_pred = LRM.predict(X_test)
	y_pred = LRM.predict(X)

	r2_train = r2_score(y_train, y_train_pred)
	r2_test = r2_score(y_test, y_test_pred)
	mae_train = mean_absolute_error(y_train, y_train_pred)
	mse_train = mean_squared_error(y_train, y_train_pred)
	rmse_train = np.sqrt(mse_train)
	mae_test = mean_absolute_error(y_test, y_test_pred)
	mse_test = mean_squared_error(y_test, y_test_pred)
	rmse_test = np.sqrt(mse_test)

	st.write("Linear Regression Model Results:")
	st.write(f"R^2 (Train): {r2_train:.4f}")
	st.write(f"R^2 (Test): {r2_test:.4f}")
	st.write(f"MAE Train: {mae_train:.4f}")
	st.write(f"MSE Train: {mse_train:.4f}")
	st.write(f"RMSE Train: {rmse_train:.4f}")
	st.write(f"MAE Test: {mae_test:.4f}")
	st.write(f"MSE Test: {mse_test:.4f}")
	st.write(f"RMSE Test: {rmse_test:.4f}")

    # Plot feature importances
	feature_importances = np.abs(LRM.coef_)
	top_indices = np.argsort(feature_importances)[::-1][:10]
	top_features = X.columns[top_indices]
	top_importances = feature_importances[top_indices]

	fig, ax = plt.subplots(figsize=(5, 4))
	ax.barh(top_features, top_importances)
	ax.set_xlabel('Importance')
	ax.set_title('Top 10 Feature Importances')
	ax.invert_yaxis()
	st.pyplot(fig)

    # Diagnostics
	residuals = y_test - y_test_pred
	fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

	sns.scatterplot(x=y_test_pred, y=residuals, ax=axes[0, 0])
	axes[0, 0].set_title("Residuals Dispersion Plot")
	axes[0, 0].set_xlabel("Predictions")
	axes[0, 0].set_ylabel("Residuals")
	axes[0, 0].axhline(y=0, color='r', linestyle='--')

	sns.histplot(residuals, ax=axes[0, 1], kde=True)
	axes[0, 1].set_title("Histogram of Residuals")
	axes[0, 1].set_xlabel("Residuals")

	sns.scatterplot(x=y_test, y=y_test_pred, ax=axes[1, 0])
	axes[1, 0].set_title("Comparison of Actual vs. Predicted Values")
	axes[1, 0].set_xlabel("Actual Values")
	axes[1, 0].set_ylabel("Predictions")
	axes[1, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')

	stats.probplot(residuals, plot=axes[1, 1])
	axes[1, 1].set_title("QQ Plot of Residuals")

	plt.tight_layout()
	st.pyplot(fig)
	
def main():
			pages = {
				"Introduction": introduction,
				"Data Exploration": data_exploration,
				"Visualization": visualization,
				"Dashboard": dashboard,
				"Modelling": modeling,
				"Prediction": prediction,
				"Conclusion": conclusion
				
				
			}

			# Sidebar navigation
			st.sidebar.title("Navigation")
			selected_page = st.sidebar.radio("Go to", list(pages.keys()))

			# Display the selected page
			if selected_page in pages:
				pages[selected_page]()

if __name__ == "__main__":
	main()



