# %%
import numpy as np
import pandas as pd  

# %% [markdown]
# ### Data cleaning , main disaese outbreak dataset. 

# %%
# read in main dataset
sal_data = pd.read_csv('/Users/kristof/data/salmon/raw_data/ila_pd.csv')

# %%
# Translate column names to English
sal_data.columns = [
    'Week', 'Year', 'LocationNumber', 'LocationName', 'Disease', 'Status', 
    'FromDate', 'ToDate', 'MunicipalityNumber', 'Municipality', 'CountyNumber', 
    'County', 'Latitude', 'Longitude', 'ProductionAreaId', 'ProductionArea', 
    'OutbreakId', 'Subtype', 'SuspectedDate', 'ConfirmedDate', 'ClearedDate', 'ClosedDate'
]
sal_data = sal_data[['Week', 'Year','Disease', 'Status', 
    'FromDate', 'ToDate', 'County', 'Latitude', 'Longitude', 'Subtype', 'SuspectedDate', 'ConfirmedDate', 'ClearedDate', 'ClosedDate']]

# Convert all date columns to datetime
date_columns = ['FromDate', 'ToDate', 'SuspectedDate', 'ConfirmedDate', 'ClearedDate', 'ClosedDate']
for col in date_columns:
    sal_data[col] = pd.to_datetime(sal_data[col])

# %%

# Sort by 'Year', 'Week', and 'County' to ensure proper ordering
sal_data.sort_values(by=['Year', 'Week', 'County'], inplace=True)

sal_data

# %%
# as there are different columns for start and ed date , Function to determine the best start and end date in order to detemrine an outbreak duration
def get_best_dates(row):
    if pd.notna(row['FromDate']) and pd.notna(row['ToDate']):
        return row['FromDate'], row['ToDate']
    
    if pd.notna(row['SuspectedDate']) and pd.notna(row['ClearedDate']):
        return row['SuspectedDate'], row['ClearedDate']
    
    if pd.notna(row['ConfirmedDate']) and pd.notna(row['ClosedDate']):
        return row['ConfirmedDate'], row['ClosedDate']
    
    return pd.NaT, pd.NaT

# %%
import pandas as pd
from datetime import timedelta

# Apply the function to each row to get the best start and end dates
sal_data[['BestFromDate', 'BestToDate']] = sal_data.apply(get_best_dates, axis=1, result_type='expand')
#  convert to weekly dates
sal_data['BestFromDate'] = pd.to_datetime(sal_data['BestFromDate'])
sal_data['BestToDate'] = pd.to_datetime(sal_data['BestToDate'])


# %%
sal_data


# %%
# function too count and clear outbreaks
def count_active_outbreaks(row, df):
    current_date = row['BestFromDate'] 
    county = row['County']
    
    # Count how many outbreaks are ongoing in each county
    return df[
        (df['County'] == county) &
        (df['BestFromDate'] <= current_date) &
        (df['BestToDate'] >= current_date)
    ].shape[0]


sal_data['Active_Outbreaks'] = sal_data.apply(lambda row: count_active_outbreaks(row, sal_data), axis=1)


# %%
sal_data

# %%
# create  a mapping dictonary as some dataframes have different spelling with the counties
# array of unique county names, taken from all the unique names
import pandas as pd

unique_counties = ['Nordland', 'Troms', 'Finnmark', 'Møre og Romsdal',
                         'Sør-Trøndelag','¯vrige fylker', 'Sogn og Fjordane', 'Rogaland', 
                         'Hordaland', 'Nord-Trøndelag', 'Trøndelag', 
                         'Vestland', 'Troms og Finnmark']

# Create a mapping dictionary for merging counties
county_mapping = {
    'Troms og Finnmark': 'Troms og Finnmark',
    'Finnmark': 'Troms og Finnmark',
    'Troms': 'Troms og Finnmark',
    'Møre og Romsdal': 'Møre og Romsdal',
    'M¿re og Romsdal': 'Møre og Romsdal',
    'Trøndelag': 'Trøndelag',
    'Tr¿ndelag': 'Trøndelag',
    'Sør-Trøndelag': 'Trøndelag',
    'Nord-Trøndelag': 'Trøndelag',
    'Sogn og Fjordane': 'Vestland',
    'Hordaland': 'Vestland',
    'Vestland': 'Vestland',
    'Rogaland': 'Rogaland',
    'Nordland': 'Nordland'
}

# %%
# Replace the county names with the correct spellings
sal_data['County'] = sal_data['County'].replace(county_mapping)

# %%
sal_data["County"].unique()

# %%
# clean the start stop dates of the disease , keeping only essential columns.
sal_data_2 = sal_data[['Week','Year','Disease','County','Latitude','Longitude', 'Active_Outbreaks']]

# %%
sal_data_2

# %%
sal_data_2.to_csv('sal_disease_model.csv', index=False)

# %% [markdown]
# ### load the above df, 

# %%
sal_data_2  = pd.read_csv('/Users/kristof/data/salmon/sal_disease_model.csv')

# %% [markdown]
# ### Farmed salmon license data , to index each region. 

# %%
# read in licenses data
license_data = pd.read_csv('/Users/kristof/data/salmon/raw_data/laks_licences.csv', encoding='iso-8859-1')

# %%
# Create DataFrame for the number of sites
sites_df = pd.DataFrame(license_data)
# Replace the county names with the correct spellings
sites_df['County'] = sites_df['County'].replace(county_mapping)

# %%
#  sum values to merge duplicates like 'Troms og Finnmark' as its over 3 rows
sites_df.iloc[:, 1:] = sites_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
# no site information for 2024, just duplicate 2023
sites_df["2024"] = sites_df["2023"]

# %%
sites_df

# %%
# Step 2: Melt to long format
site_long = sites_df.melt(id_vars='County', var_name='Year', value_name='Sites')
site_long['Sites'] = pd.to_numeric(site_long['Sites'], errors='coerce')
site_long['Year'] = pd.to_numeric(site_long['Year'])
sal_data['Year'] = sal_data['Year'].astype(int)  # make sure this matches
merged = pd.merge(sal_data_2, site_long, on=['County', 'Year'], how='left')

# Step 6: Normalize outbreak count
merged['Norm_Outbreak'] = merged['Active_Outbreaks'] / merged['Sites']

# %%
merged

# %% [markdown]
# ### salmon lice count and sea temp

# %%
# read losses data
lice_count = pd.read_csv('/Users/kristof/data/salmon/raw_data/lakselus_per_fisk.csv')

# %%
lice_count = lice_count[['Uke','År','Voksne hunnlus','Lus i bevegelige stadier','Fastsittende lus','Fylke','Lat','Lon','Sjøtemperatur']]
lice_count.columns= ['Week','Year','Adult female lice','Lice in motile stages','Sedentary lice','County','Lat','Lon','Sea temperature']
lice_count['County'] = lice_count['County'].replace(county_mapping)

# Ensure the specified columns are converted to numeric types
numeric_columns = ['Adult female lice', 'Lice in motile stages', 'Sedentary lice']
for column in numeric_columns:
    lice_count[column] = pd.to_numeric(lice_count[column], errors='coerce')

# Sum the values in the specified columns into a new column 'Total lice counts'
lice_count['Total lice counts'] = lice_count[numeric_columns].sum(axis=1)

# Filter out the rows where the County column contains any of the specified values
lice_count = lice_count[~lice_count['County'].isin(['Agder', 'Aust-Agder', 'Vest-Agder'])]

lice_count = lice_count[['Week','Year','County','Sea temperature','Total lice counts']]

# Remove rows where 'Total lice counts' is 0 or NaN
lice_count = lice_count[lice_count['Total lice counts'] > 0]

# Sort by 'Year', 'Week', and 'County' to ensure proper ordering
lice_count.sort_values(by=['Year', 'Week', 'County'], inplace=True)


# %%
lice_count

# %%
# First, group lice_count to get average lice counts and sea temperature per Week-Year-County
aggregated_lice = lice_count.groupby(['Week', 'Year', 'County']).agg({
    'Sea temperature': 'mean',
    'Total lice counts': 'mean'
}).reset_index()

# Merge with sal_data_2 on Week, Year, County
combined_df = pd.merge(
    merged,
    aggregated_lice,
    on=['Week', 'Year', 'County'],
    how='left'  # or 'inner' if you want only matching rows
)

# %%
combined_df

# %%
combined_df = combined_df.dropna()

# %%
combined_df.to_csv('combined_df.csv', index=False)

# %% [markdown]
# ### load clean df

# %%
combined_df = pd.read_csv('combined_df.csv')

# %% [markdown]
# ### visualisation 

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 7))
correlation_matrix = combined_df[['Norm_Outbreak','Sea temperature','Total lice counts','Sites']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# %%

combined_df['Date'] = pd.to_datetime(combined_df['Year'].astype(str) + '-W' + combined_df['Week'].astype(str) + '-1', format='%Y-W%W-%w')


# %%
plt.figure(figsize=(14, 7))
sns.lineplot(data= combined_df, x='Date', y='Norm_Outbreak', hue='County' )

# Add vertical lines for each year
for year in combined_df['Year'].unique():
    plt.axvline(pd.to_datetime(f"{year}-01-01"), color='gray', linestyle='--', linewidth=0.5)

plt.title('Normalised Outbreak Count Over Time')
plt.xlabel('Date')
plt.ylabel('Outbreak Count Normalised')
plt.xticks(rotation=45)
plt.legend(title='County')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(data= combined_df, x='Date', y='Total lice counts', hue='County' )

# Add vertical lines for each year
for year in combined_df['Year'].unique():
    plt.axvline(pd.to_datetime(f"{year}-01-01"), color='gray', linestyle='--', linewidth=0.5)

plt.title('Lice Count (per fish) Over Time')
plt.xlabel('Date')
plt.ylabel('Lice Count (per fish)')
plt.xticks(rotation=45)
plt.legend(title='County')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### map information

# %%
from datetime import timedelta
import folium
from folium.plugins import TimestampedGeoJson, HeatMap
import pandas as pd
import branca.colormap as cm

# Load data
df = pd.read_csv("combined_df.csv")
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-W' + df['Week'].astype(str) + '-1', format='%Y-W%W-%w')

# Normalize outbreak
df['color_intensity'] = (df['Norm_Outbreak'] / df['Norm_Outbreak'].max() * 255).astype(int)

# Color for outbreak dots
def color_map(val):
    intensity = min(255, max(0, val))
    return f"#{intensity:02x}0000"  # red shades

# Create map
m = folium.Map(location=[65, 15], zoom_start=5)

#  Add sea surface temperature background as a HeatMap
heat_data = [
    [row['Latitude'], row['Longitude'], row['Sea temperature']]
    for _, row in df.iterrows()
]

HeatMap(
    heat_data,
    radius=25,
    max_zoom=5,
    blur=15,
    min_opacity=0.4
).add_to(m)

# 🐟 Add outbreak dots as time-based features

features = []
for _, row in df.iterrows():
    if row['Active_Outbreaks'] > 0:
        start_time = row['Date']
        end_time = row['Date'] + timedelta(days=6)  # show for 1 week only
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [float(row['Longitude']), float(row['Latitude'])],
            },
            'properties': {
                'times': [start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d')],
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': color_map(row['color_intensity']),
                    'fillOpacity': 0.8,
                    'stroke': False,
                    'radius': 4
                },
                'popup': f"Active Outbreaks at site (normalised): {row['Norm_Outbreak']:.2f}<br>Sea Temp: {row['Sea temperature']:.2f}°C<br>Lice Count per fish: {row['Total lice counts']:.2f}<br>Total no. sites in county: {row['Sites']}"
            }
        })

TimestampedGeoJson({
    'type': 'FeatureCollection',
    'features': features
}, period='P1W', duration='P1W', auto_play=True, loop=False,
   date_options='YYYY-MM-DD', time_slider_drag_update=True).add_to(m)

# Add outbreak intensity legend (color scale)
outbreak_colormap = cm.LinearColormap(
    colors=['#000000', '#800000', '#ff0000'],  # black → dark red → red
    vmin=0, vmax=df['Norm_Outbreak'].max(),
    caption='Outbreak Intensity (Normalized)'
)
outbreak_colormap.add_to(m)

# Add sea temperature legend
sea_temp_colormap = cm.LinearColormap(
    colors=['blue', 'cyan', 'yellow', 'orange', 'red'],
    vmin=df['Sea temperature'].min(),
    vmax=df['Sea temperature'].max(),
    caption='Sea Surface Temperature (°C)'
)
sea_temp_colormap.add_to(m)

# Save
m.save('outbreak_map_with_sst_heatmap.html')

# %%
from datetime import timedelta
import folium
from folium.plugins import TimestampedGeoJson, HeatMap
import pandas as pd

# Load and preprocess data
df = pd.read_csv("combined_df.csv")
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-W' + df['Week'].astype(str) + '-1', format='%Y-W%W-%w')

# 🔽 Reduce data points for performance
df = df.sample(frac=0.7, random_state=42)  # keep 30% of data

# Normalize and simplify outbreak colors
df['color_intensity'] = (df['Norm_Outbreak'] / df['Norm_Outbreak'].max() * 255).astype(int)

def color_map(val):
    val = min(255, max(0, val))
    return f"#{val:02x}0000"

# Initialize map
m = folium.Map(location=[65, 15], zoom_start=5)

# Simplify HeatMap data: round and group
df['LatRound'] = df['Latitude'].round(2)
df['LonRound'] = df['Longitude'].round(2)
heat_df = df.groupby(['LatRound', 'LonRound']).agg({'Sea temperature': 'mean'}).reset_index()
heat_data = heat_df[['LatRound', 'LonRound', 'Sea temperature']].values.tolist()

HeatMap(
    heat_data,
    radius=20,
    max_zoom=5,
    blur=10,
    min_opacity=0.4
).add_to(m)

# 🐟 Add outbreak points with reduced detail
features = []
for _, row in df.iterrows():
    if row['Active_Outbreaks'] > 0:
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [float(row['Longitude']), float(row['Latitude'])],
            },
            'properties': {
                'times': [row['Date'].strftime('%Y-%m-%d')],
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': color_map(row['color_intensity']),
                    'fillOpacity': 0.7,
                    'stroke': False,
                    'radius': 4
                },
                'popup': f"Active Outbreaks at site (normalised): {row['Norm_Outbreak']:.2f}<br>Sea Temp: {row['Sea temperature']:.2f}°C<br>Lice Count per fish: {row['Total lice counts']:.2f}<br>Total no. sites in county: {row['Sites']}"
            }
        })

TimestampedGeoJson({
    'type': 'FeatureCollection',
    'features': features
}, period='P1W', duration='P1W', auto_play=True, loop=False,
   date_options='YYYY-MM-DD', time_slider_drag_update=True).add_to(m)

# Legends 

# Add outbreak intensity legend (color scale)
outbreak_colormap = cm.LinearColormap(
    colors=['#000000', '#800000', '#ff0000'],  # black → dark red → red
    vmin=0, vmax=df['Norm_Outbreak'].max(),
    caption='Outbreak Intensity (Normalized)'
)
outbreak_colormap.add_to(m)

# Add sea temperature legend
sea_temp_colormap = cm.LinearColormap(
    colors=['blue', 'cyan', 'yellow', 'orange', 'red'],
    vmin=df['Sea temperature'].min(),
    vmax=df['Sea temperature'].max(),
    caption='Sea Surface Temperature (°C)'
)
sea_temp_colormap.add_to(m)


# Save with external JS/CSS references to shrink HTML
m.save("outbreak_map_with_sst_heatmap_light.html")

# %%
m


