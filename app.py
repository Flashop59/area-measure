import streamlit as st
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import folium
from folium import plugins
from geopy.distance import geodesic
import requests
from datetime import datetime, timedelta
from streamlit_folium import folium_static

# Function to fetch data from the API
def fetch_data(vehicle, start_time, end_time):
    API_KEY = "3330d953-7abc-4bac-b862-ac315c8e2387-6252fa58-d2c2-4c13-b23e-59cefafa4d7d"
    url = f"https://admintestapi.ensuresystem.in/api/locationpull/orbit?vehicle={vehicle}&from={start_time}&to={end_time}"
    headers = {"token": API_KEY}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"Error fetching data: {response.status_code}")
        return None

    data = response.json()
    if not isinstance(data, list):
        st.error(f"Unexpected data format: {data}")
        return None
    
    # Sort data by time
    data.sort(key=lambda x: x['time'])
    return data

# Function to calculate the area of a field in square meters using convex hull
def calculate_convex_hull_area(points):
    if len(points) < 3:  # Not enough points to form a polygon
        return 0
    try:
        hull = ConvexHull(points)
        poly = Polygon(points[hull.vertices])
        return poly.area  # Area in square degrees
    except Exception:
        return 0

# Function to calculate centroid of a set of points
def calculate_centroid(points):
    return np.mean(points, axis=0)

# Function to process the fetched data and return the map and field areas
def process_data(data):
    # Create a DataFrame from the fetched data
    gps_data = pd.DataFrame(data)
    gps_data['Timestamp'] = pd.to_datetime(gps_data['time'], unit='ms')
    gps_data['lat'] = gps_data['lat']
    gps_data['lng'] = gps_data['lon']
    
    # Cluster the GPS points to identify separate fields
    coords = gps_data[['lat', 'lng']].values
    db = DBSCAN(eps=0.00008, min_samples=11).fit(coords)
    labels = db.labels_

    # Add labels to the data
    gps_data['field_id'] = labels

    # Calculate the area for each field
    fields = gps_data[gps_data['field_id'] != -1]  # Exclude noise points
    field_areas = fields.groupby('field_id').apply(
        lambda df: calculate_convex_hull_area(df[['lat', 'lng']].values))

    # Convert the area from square degrees to square meters (approximation)
    field_areas_m2 = field_areas * 0.77 * (111000 ** 2)  # rough approximation

    # Convert the area from square meters to gunthas (1 guntha = 101.17 m^2)
    field_areas_gunthas = field_areas_m2 / 101.17

    # Calculate time metrics for each field
    field_times = fields.groupby('field_id').apply(
        lambda df: (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds() / 60.0
    )

    # Extract start and end dates for each field
    field_dates = fields.groupby('field_id').agg(
        start_date=('Timestamp', 'min'),
        end_date=('Timestamp', 'max')
    )

    # Filter out fields with area less than 5 gunthas
    valid_fields = field_areas_gunthas[field_areas_gunthas >= 5].index
    field_areas_gunthas = field_areas_gunthas[valid_fields]
    field_times = field_times[valid_fields]
    field_dates = field_dates.loc[valid_fields]

    # Calculate centroids of each field
    centroids = fields.groupby('field_id').apply(
        lambda df: calculate_centroid(df[['lat', 'lng']].values)
    )

    # Calculate traveling distance and time between field centroids
    travel_distances = []
    travel_times = []
    field_ids = list(valid_fields)
    
    if len(field_ids) > 1:
        for i in range(len(field_ids) - 1):
            centroid1 = centroids.loc[field_ids[i]]
            centroid2 = centroids.loc[field_ids[i + 1]]
            distance = geodesic(centroid1, centroid2).kilometers
            time = (field_dates.loc[field_ids[i + 1], 'start_date'] - field_dates.loc[field_ids[i], 'end_date']).total_seconds() / 60.0
            travel_distances.append(distance)
            travel_times.append(time)

        # Calculate distance from last point of one field to first point of the next field
        for i in range(len(field_ids) - 1):
            end_point = fields[fields['field_id'] == field_ids[i]][['lat', 'lng']].values[-1]
            start_point = fields[fields['field_id'] == field_ids[i + 1]][['lat', 'lng']].values[0]
            distance = geodesic(end_point, start_point).kilometers
            time = (field_dates.loc[field_ids[i + 1], 'start_date'] - field_dates.loc[field_ids[i], 'end_date']).total_seconds() / 60.0
            travel_distances.append(distance)
            travel_times.append(time)

        # Append NaN for the last field
        travel_distances.append(np.nan)
        travel_times.append(np.nan)
    else:
        travel_distances.append(np.nan)
        travel_times.append(np.nan)

    # Ensure lengths match for DataFrame
    if len(travel_distances) != len(field_areas_gunthas):
        travel_distances = travel_distances[:len(field_areas_gunthas)]
        travel_times = travel_times[:len(field_areas_gunthas)]

    # Combine area, time, dates, and travel metrics into a single DataFrame
    combined_df = pd.DataFrame({
        'Field ID': field_areas_gunthas.index,
        'Area (Gunthas)': field_areas_gunthas.values,
        'Time (Minutes)': field_times.values,
        'Start Date': field_dates['start_date'].values,
        'End Date': field_dates['end_date'].values,
        'Travel Distance to Next Field (km)': travel_distances,
        'Travel Time to Next Field (minutes)': travel_times
    })
    
    # Calculate total metrics
    total_area = field_areas_gunthas.sum()
    total_time = field_times.sum()
    total_travel_distance = np.nansum(travel_distances)
    total_travel_time = np.nansum(travel_times)

    # Create a satellite map
    map_center = [gps_data['lat'].mean(), gps_data['lng'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)
    
    # Add Mapbox satellite imagery
    mapbox_token = 'pk.eyJ1IjoiZmxhc2hvcDAwNyIsImEiOiJjbHo5NzkycmIwN2RxMmtzZHZvNWpjYmQ2In0.A_FZYl5zKjwSZpJuP_MHiA'
    folium.TileLayer(
        tiles='https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{z}/{x}/{y}?access_token=' + mapbox_token,
        attr='Mapbox Satellite Imagery',
        name='Satellite',
        overlay=True,
        control=True
    ).add_to(m)
    
    # Add fullscreen control
    plugins.Fullscreen(position='topright').add_to(m)

    # Plot the points on the map
    for idx, row in gps_data.iterrows():
        color = 'blue' if row['field_id'] in valid_fields else 'red'  # Blue for fields, red for noise
        folium.CircleMarker(
            location=(row['lat'], row['lng']),
            radius=2,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    return m, combined_df, total_area, total_time, total_travel_distance, total_travel_time

# Streamlit UI
st.title("Field Area and Time Calculation from GPS Data")

# Display logo
st.markdown("""
    <style>
        .header { display: flex; align-items: center; }
        .header img { height: 80px; margin-right: 35px; }
    </style>
    <div class="header">
        <img src="https://i.ibb.co/JjWJLpd/image.png" alt="Logo">
        <h1>Field Area and Time Calculation from GPS Data</h1>
    </div>
""", unsafe_allow_html=True)

st.write("Enter vehicle details and date range to calculate field areas and visualize them on a satellite map.")

vehicle = st.text_input("Enter Vehicle ID (e.g., BR1):")
start_date = st.date_input("Select Start Date", value=datetime.now().date() - timedelta(days=1))
end_date = st.date_input("Select End Date", value=datetime.now().date())

if st.button("Fetch and Process Data"):
    if vehicle and start_date and end_date:
        # Convert dates to timestamps in milliseconds since epoch
        start_time_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_time_ms = int(datetime.combine(end_date, datetime.max.time()).timestamp() * 1000)

        # Fetch data in 24-hour intervals and combine it
        current_start_time = start_time_ms
        combined_data = []
        while current_start_time < end_time_ms:
            current_end_time = min(current_start_time + 86400000, end_time_ms)  # 86400000 ms = 24 hours
            data = fetch_data(vehicle, current_start_time, current_end_time)
            if data:
                combined_data.extend(data)
            current_start_time = current_end_time

        if combined_data:
            # Process the data and generate the map and DataFrame
            map_obj, combined_df, total_area, total_time, total_travel_distance, total_travel_time = process_data(combined_data)
            
            # Display the map
            st.subheader("Field Map")
            folium_static(map_obj)
            
            # Display the DataFrame
            st.subheader("Field Area and Time Data")
            st.dataframe(combined_df)
            
            # Display total metrics
            st.subheader("Total Metrics")
            st.write(f"Total Area: {total_area:.2f} Gunthas")
            st.write(f"Total Time: {total_time:.2f} Minutes")
            st.write(f"Total Travel Distance: {total_travel_distance:.2f} km")
            st.write(f"Total Travel Time: {total_travel_time:.2f} Minutes")
            
            # Downloadable CSV of the DataFrame
            csv = combined_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV", data=csv, file_name=f'{vehicle}_field_data.csv', mime='text/csv')
    else:
        st.warning("Please enter all required fields.")
