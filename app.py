import streamlit as st
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import folium
from folium import plugins
from geopy.distance import geodesic
from datetime import datetime
from streamlit_folium import folium_static

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

# Function to process the uploaded data and return the map and field areas
def process_data(data):
    gps_data = pd.DataFrame(data)
    gps_data['Timestamp'] = pd.to_datetime(gps_data['time'], unit='ms')
    gps_data['lat'] = gps_data['lat']
    gps_data['lng'] = gps_data['lng']

    coords = gps_data[['lat', 'lng']].values
    db = DBSCAN(eps=0.00008, min_samples=11).fit(coords)
    labels = db.labels_
    gps_data['field_id'] = labels

    fields = gps_data[gps_data['field_id'] != -1]
    field_areas = fields.groupby('field_id').apply(
        lambda df: calculate_convex_hull_area(df[['lat', 'lng']].values))
    field_areas_m2 = field_areas * 0.77 * (111000 ** 2)
    field_areas_gunthas = field_areas_m2 / 101.17
    field_times = fields.groupby('field_id').apply(
        lambda df: (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds() / 60.0)
    field_dates = fields.groupby('field_id').agg(
        start_date=('Timestamp', 'min'),
        end_date=('Timestamp', 'max'))

    valid_fields = field_areas_gunthas[field_areas_gunthas >= 5].index
    field_areas_gunthas = field_areas_gunthas[valid_fields]
    field_times = field_times[valid_fields]
    field_dates = field_dates.loc[valid_fields]

    centroids = fields.groupby('field_id').apply(
        lambda df: calculate_centroid(df[['lat', 'lng']].values))

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

        for i in range(len(field_ids) - 1):
            end_point = fields[fields['field_id'] == field_ids[i]][['lat', 'lng']].values[-1]
            start_point = fields[fields['field_id'] == field_ids[i + 1]][['lat', 'lng']].values[0]
            distance = geodesic(end_point, start_point).kilometers
            time = (field_dates.loc[field_ids[i + 1], 'start_date'] - field_dates.loc[field_ids[i], 'end_date']).total_seconds() / 60.0
            travel_distances.append(distance)
            travel_times.append(time)

        travel_distances.append(np.nan)
        travel_times.append(np.nan)
    else:
        travel_distances.append(np.nan)
        travel_times.append(np.nan)

    if len(travel_distances) != len(field_areas_gunthas):
        travel_distances = travel_distances[:len(field_areas_gunthas)]
        travel_times = travel_times[:len(field_areas_gunthas)]

    combined_df = pd.DataFrame({
        'Field ID': field_areas_gunthas.index,
        'Area (Gunthas)': field_areas_gunthas.values,
        'Time (Minutes)': field_times.values,
        'Start Date': field_dates['start_date'].values,
        'End Date': field_dates['end_date'].values,
        'Travel Distance to Next Field (km)': travel_distances,
        'Travel Time to Next Field (minutes)': travel_times
    })

    total_area = field_areas_gunthas.sum()
    total_time = field_times.sum()
    total_travel_distance = np.nansum(travel_distances)
    total_travel_time = np.nansum(travel_times)

    map_center = [gps_data['lat'].mean(), gps_data['lng'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    mapbox_token = 'pk.eyJ1IjoiZmxhc2hvcDAwNyIsImEiOiJjbHo5NzkycmIwN2RxMmtzZHZvNWpjYmQ2In0.A_FZYl5zKjwSZpJuP_MHiA'
    folium.TileLayer(
        tiles='https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{z}/{x}/{y}?access_token=' + mapbox_token,
        attr='Mapbox Satellite Imagery',
        name='Satellite',
        overlay=True,
        control=True
    ).add_to(m)
    plugins.Fullscreen(position='topright').add_to(m)

    for idx, row in gps_data.iterrows():
        color = 'blue' if row['field_id'] in valid_fields else 'red'
        folium.CircleMarker(
            location=(row['lat'], row['lng']),
            radius=2,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    return m, combined_df, total_area, total_time, total_travel_distance, total_travel_time

# Streamlit UI
st.title("Field Area and Time Calculation from GPS CSV")

st.write("Upload your CSV file containing GPS points with a column named 'Ignition' having 'lat,lon' values and timestamp as the first column.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'time'}, inplace=True)

        df[['lat', 'lon']] = df['Ignition'].str.split(",", expand=True)
        df.dropna(subset=['lat', 'lon'], inplace=True)
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df.dropna(subset=['lat', 'lon', 'time'], inplace=True)

        formatted_data = df[['lat', 'lon', 'time']].rename(columns={'lon': 'lng'})
        formatted_data['time'] = formatted_data['time'].astype(np.int64) // 10**6

        map_obj, combined_df, total_area, total_time, total_travel_distance, total_travel_time = process_data(formatted_data.to_dict('records'))

        st.subheader("Field Map")
        folium_static(map_obj)

        st.subheader("Field Area and Time Data")
        st.dataframe(combined_df)

        st.subheader("Total Metrics")
        st.write(f"Total Area: {total_area:.2f} Gunthas")
        st.write(f"Total Time: {total_time:.2f} Minutes")
        st.write(f"Total Travel Distance: {total_travel_distance:.2f} km")
        st.write(f"Total Travel Time: {total_travel_time:.2f} Minutes")

        csv = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name='field_data.csv', mime='text/csv')

    except Exception as e:
        st.error(f"Error processing file: {e}")
