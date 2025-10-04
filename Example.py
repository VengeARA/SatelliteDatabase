#%% Ultimate Satellite Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Ultimate Satellite Dashboard",
    page_icon="🛰️",
    layout="wide"
)
st.title("Ultimate Satellite Database Dashboard")

# --- LOAD DATA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "database.csv")
df = pd.read_csv(csv_path)

# Preprocess
df['Launch Year'] = pd.to_datetime(df['Date of Launch'], errors='coerce').dt.year
df['Expected Lifetime (Years)'] = pd.to_numeric(df['Expected Lifetime (Years)'], errors='coerce')
df['Perigee (km)'] = pd.to_numeric(df['Perigee (Kilometers)'], errors='coerce')
df['Apogee (km)'] = pd.to_numeric(df['Apogee (Kilometers)'], errors='coerce')
df['Inclination (°)'] = pd.to_numeric(df['Inclination (Degrees)'], errors='coerce')

# Filter satellites with valid orbital info
orbital_df = df.dropna(subset=['Perigee (km)', 'Apogee (km)', 'Inclination (°)'])

# --- 1️⃣ COLLISION RISK & SAFETY ---
st.subheader("Collision Risk Estimator (Simplified)")

st.markdown("Select your satellite's planned orbit to check for nearby satellites:")

user_perigee = st.number_input("Your Satellite Perigee (km)", value=400, key="risk_perigee")
user_apogee = st.number_input("Your Satellite Apogee (km)", value=420, key="risk_apogee")
user_inclination = st.number_input("Your Satellite Inclination (°)", value=51.6, key="risk_inclination")

# Simple distance-based risk: difference in perigee, apogee, inclination
orbital_df['Risk Score'] = np.sqrt(
    (orbital_df['Perigee (km)'] - user_perigee)**2 +
    (orbital_df['Apogee (km)'] - user_apogee)**2 +
    (orbital_df['Inclination (°)'] - user_inclination)**2
)

# Top 10 closest satellites
closest_sats = orbital_df.nsmallest(10, 'Risk Score')[['Official Name of Satellite', 'Perigee (km)', 'Apogee (km)', 'Inclination (°)', 'Risk Score']]
st.dataframe(closest_sats.style.highlight_min('Risk Score', color='red'))

# --- 2️⃣ SATELLITE COMPARISON TOOL ---
st.subheader("Satellite Comparison Tool")

sat_selection = st.multiselect(
    "Select Satellites to Compare",
    options=df['Official Name of Satellite'].dropna().sort_values(),
    default=df['Official Name of Satellite'].dropna().iloc[:3]
)

if sat_selection:
    compare_df = df[df['Official Name of Satellite'].isin(sat_selection)][['Official Name of Satellite', 'Perigee (km)', 'Apogee (km)', 'Inclination (°)', 'Launch Year', 'Expected Lifetime (Years)', 'Launch Mass (Kilograms)', 'Class of Orbit', 'Type of Orbit']]
    st.dataframe(compare_df)

    # Radar chart for orbital parameters
    st.markdown("**Radar Chart: Orbital Parameters**")
    radar_fig = go.Figure()
    for _, row in compare_df.iterrows():
        radar_fig.add_trace(go.Scatterpolar(
            r=[row['Perigee (km)'], row['Apogee (km)'], row['Inclination (°)']],
            theta=['Perigee', 'Apogee', 'Inclination'],
            fill='toself',
            name=row['Official Name of Satellite']
        ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(radar_fig, use_container_width=True)

# --- 3️⃣ HISTORICAL TRENDS ---
st.subheader("Historical Trends: Satellite Launches Over Time")

launch_counts = df.groupby('Launch Year').size().reset_index(name='Count')
fig_trend = px.line(launch_counts, x='Launch Year', y='Count', markers=True, title="Number of Satellites Launched Per Year")
st.plotly_chart(fig_trend, use_container_width=True)

# Country-wise trend
st.markdown("**Country-wise Satellite Launches Over Time**")
top_countries = df['Country/Organization of UN Registry'].value_counts().nlargest(5).index
country_trend_df = df[df['Country/Organization of UN Registry'].isin(top_countries)]
country_trend_df = country_trend_df.groupby(['Launch Year', 'Country/Organization of UN Registry']).size().reset_index(name='Count')
fig_country_trend = px.line(country_trend_df, x='Launch Year', y='Count', color='Country/Organization of UN Registry', markers=True)
st.plotly_chart(fig_country_trend, use_container_width=True)

# --- 4️⃣ ADVANCED ORBIT VISUALIZATION ---
st.subheader("3D Satellite Orbit Viewer")

# Color by orbit class
orbit_classes = orbital_df['Class of Orbit'].dropna().unique()
color_map = {orbit: f"hsl({i*360/len(orbit_classes)},70%,50%)" for i, orbit in enumerate(orbit_classes)}

fig_3d = go.Figure()
for orbit_class in orbit_classes:
    subset = orbital_df[orbital_df['Class of Orbit'] == orbit_class]
    fig_3d.add_trace(go.Scatter3d(
        x=subset['Perigee (km)'],
        y=subset['Apogee (km)'],
        z=subset['Inclination (°)'],
        mode='markers',
        marker=dict(size=4, color=color_map[orbit_class]),
        name=orbit_class,
        text=subset['Official Name of Satellite'],
        hoverinfo='text'
    ))

# Add user satellite
fig_3d.add_trace(go.Scatter3d(
    x=[user_perigee],
    y=[user_apogee],
    z=[user_inclination],
    mode='markers',
    marker=dict(size=8, color='red', symbol='diamond'),
    name='Your Satellite',
    text=['Planned Orbit'],
    hoverinfo='text'
))

fig_3d.update_layout(
    scene=dict(
        xaxis_title='Perigee (km)',
        yaxis_title='Apogee (km)',
        zaxis_title='Inclination (°)'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(title="Orbit Class")
)
st.plotly_chart(fig_3d, use_container_width=True)
