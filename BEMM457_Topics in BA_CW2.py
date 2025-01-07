
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = "C:\\Users\\Freddie\\Downloads\\hip_fuel_efficiency (1).csv"
data = pd.read_csv("C:\\Users\\Freddie\\Downloads\\ship_fuel_efficiency (1).csv")

# Check for Missing Values
missing_values = data.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values)

# Standardize or Categorize Variables
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
    'June': 6, 'July': 7, 'August': 8, 'September': 9,
    'October': 10, 'November': 11, 'December': 12
}
data['month_ordinal'] = data['month'].map(month_mapping)

# Calculate Derived Metrics
# Handle cases where 'distance' is zero or missing
data['distance'] = data['distance'].replace(0, pd.NA)
data.dropna(subset=['distance'], inplace=True) 

# Calculate emissions per kilometer (fuel efficiency) and fuel consumption rate 
data['emissions_per_km'] = data['CO2_emissions'] / data['distance']
data['fuel_consumption_rate'] = data['fuel_consumption'] / data['distance']

# Step 6: Normalize Numerical Data
# Select numerical columns to normalize
numerical_columns = ['distance', 'fuel_consumption', 'CO2_emissions', 'engine_efficiency', 
                     'emissions_per_km', 'fuel_consumption_rate']
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Exploratory Data Analysis (EDA)
# Check if seaborn is properly imported and define 'sns' explicitly
if 'sns' not in globals():
    import seaborn as sns

# Plotting fuel consumption and CO2 emissions trends by ship_type and route_id
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='ship_type', y='fuel_consumption', palette='Blues')
plt.title('Fuel Consumption by Ship Type')
plt.xlabel('Ship Type')
plt.ylabel('Normalized Fuel Consumption')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='route_id', y='CO2_emissions', palette='Oranges')
plt.title('CO2 Emissions by Route ID')
plt.xlabel('Route ID')
plt.ylabel('Normalized CO2 Emissions')
plt.xticks(rotation=90)
plt.show()

# Correlation matrix for numerical features
correlation_matrix = data[numerical_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Assess the impact of weather conditions on fuel consumption
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='weather_conditions', y='fuel_consumption', palette='Greens')
plt.title('Fuel Consumption by Weather Conditions')
plt.xlabel('Weather Conditions')
plt.ylabel('Normalized Fuel Consumption')
plt.show()

# Save the processed dataset to a new CSV file for further analysis
data.to_csv('prepared_dataset.csv', index=False)

# Step 9: Display Processed Data Sample
print("Sample of Processed Data:")
print(data.head())

#Regression Analysis
#Import Required Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = "C:\\Users\\Freddie\\Downloads\\hip_fuel_efficiency (1).csv"
data = pd.read_csv("C:\\Users\\Freddie\\Downloads\\ship_fuel_efficiency (1).csv")

# Handle cases where 'distance' might be zero or missing
data['distance'] = data['distance'].replace(0, pd.NA)  # Replace zeros with NA
data.dropna(subset=['distance'], inplace=True)  # Drop rows with NA in 'distance'

# Fuel Consumption per km
data['fuel_consumption_per_km'] = data['fuel_consumption'] / data['distance']

# Prepare Independent Variables and Encode Categorical Data
encoded_data = pd.get_dummies(data, columns=['fuel_type', 'weather_conditions'], drop_first=True)
X = encoded_data[['distance', 'CO2_emissions', 'engine_efficiency'] +
                 [col for col in encoded_data.columns if 'fuel_type_' in col or 'weather_conditions_' in col]]
y = encoded_data['fuel_consumption_per_km']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform Regression Analysis
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Get regression results
coefficients = regressor.coef_
intercept = regressor.intercept_
feature_names = X.columns

# Visualize the Regression Results
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients, color='lightblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Regression Coefficients for Fuel Consumption per km')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print the Regression Intercept
print("Regression Intercept:", intercept)

# Clustering Analysis
# Ensure KMeans is properly imported
from sklearn.cluster import KMeans

# Use KMeans clustering to group routes or ships based on emission patterns
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(data[numerical_columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='distance', y='CO2_emissions', hue='cluster', palette='viridis')
plt.title('Clustering Based on CO2 Emissions and Distance')
plt.xlabel('Normalized Distance')
plt.ylabel('Normalized CO2 Emissions')
plt.legend(title='Cluster')
plt.show()

Time-Series Analysis
# Explore monthly trends in emissions and fuel consumption
monthly_data = data.groupby('month_ordinal')[['fuel_consumption', 'CO2_emissions']].mean()

plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['fuel_consumption'], marker='o', label='Fuel Consumption')
plt.plot(monthly_data.index, monthly_data['CO2_emissions'], marker='o', label='CO2 Emissions', linestyle='--')
plt.title('Monthly Trends in Fuel Consumption and CO2 Emissions')
plt.xlabel('Month')
plt.ylabel('Normalized Values')
plt.legend()
plt.grid()
plt.show()

# Save the processed dataset to a new CSV file for further analysis
data.to_csv('prepared_dataset.csv', index=False)

# Display Processed Data Samples
print("Sample of Processed Data:")
print(data.head())