#%% Analysis
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

import streamlit as st


st.set_page_config(
    page_title="Apoapsis",   
    page_icon="🛰️",                    
    layout="wide"                      
)


st.title("Ultimate Satellite Database Analysis")
st.write("https://www.kaggle.com/datasets/ucsusa/active-satellites")
st.write("Kaggle Dataset from 2016, may be outdated.")
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "database.csv")
df = pd.read_csv(csv_path)


st.subheader("Database Preview Rows")
st.dataframe(df)



filtered_df = df.copy()

# BAR CHART: Top Countries
st.subheader("Top Countries/Organizations by Satellite Count")
country_counts = filtered_df['Country/Organization of UN Registry'].value_counts().nlargest(15)
fig_bar = px.bar(country_counts, x=country_counts.index, y=country_counts.values,
                 labels={'x':'Country/Organization', 'y':'Number of Satellites'},
                 title="Top 15 Countries by Satellites")
st.plotly_chart(fig_bar, use_container_width=True)

# PIE & DONUT CHARTS: Orbit Types & Purposes
st.subheader("Orbit Type Distribution (Pie Chart)")
orbit_type_counts = filtered_df['Type of Orbit'].value_counts()
fig_pie = px.pie(values=orbit_type_counts.values, names=orbit_type_counts.index, hole=0.3)
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Satellite Purpose Breakdown (Donut Chart)")
purpose_counts = filtered_df['Purpose'].value_counts().nlargest(10)
fig_donut = px.pie(values=purpose_counts.values, names=purpose_counts.index, hole=0.5)
st.plotly_chart(fig_donut, use_container_width=True)


# 3D SCATTER: Perigee vs Apogee vs Inclination
st.subheader("3D Orbital Scatter")
orbital_df = filtered_df[['Official Name of Satellite', 'Perigee (Kilometers)', 'Apogee (Kilometers)', 'Inclination (Degrees)']]
orbital_df = orbital_df.dropna()

fig_3d = px.scatter_3d(orbital_df,
                       x='Perigee (Kilometers)',
                       y='Apogee (Kilometers)',
                       z='Inclination (Degrees)',
                       hover_name='Official Name of Satellite',
                       color='Inclination (Degrees)',
                       size_max=12)
fig_3d.update_layout(scene=dict(
    xaxis_title='Perigee (km)',
    yaxis_title='Apogee (km)',
    zaxis_title='Inclination (°)'
))
st.plotly_chart(fig_3d, use_container_width=True)


# SCATTER MATRIX: Relationships Between Orbital Parameters
st.subheader("Scatter Matrix of Orbital Parameters")
scatter_matrix_df = orbital_df[['Perigee (Kilometers)', 'Apogee (Kilometers)', 'Inclination (Degrees)']]
fig_matrix = px.scatter_matrix(scatter_matrix_df,
                               dimensions=scatter_matrix_df.columns,
                               color=orbital_df['Inclination (Degrees)'],
                               title="Orbital Parameter Relationships")
st.plotly_chart(fig_matrix, use_container_width=True)


# 3D BUBBLE CHART: Mass vs Perigee vs Apogee
st.subheader("3D Bubble Chart: Launch Mass vs Orbit")
mass_df = filtered_df[['Official Name of Satellite', 'Perigee (Kilometers)', 'Apogee (Kilometers)', 'Launch Mass (Kilograms)']]
mass_df = mass_df.dropna()
mass_df['Launch Mass (Kilograms)'] = pd.to_numeric(mass_df['Launch Mass (Kilograms)'], errors='coerce')
mass_df = mass_df.dropna()

fig_bubble = px.scatter_3d(mass_df,
                           x='Perigee (Kilometers)',
                           y='Apogee (Kilometers)',
                           z='Launch Mass (Kilograms)',
                           size='Launch Mass (Kilograms)',
                           hover_name='Official Name of Satellite',
                           color='Launch Mass (Kilograms)')
st.plotly_chart(fig_bubble, use_container_width=True)


# POLAR CHART: Inclination Distribution
st.subheader("Polar Chart: Inclination Distribution")
incl_counts = orbital_df['Inclination (Degrees)'].value_counts().nlargest(20)
fig_polar = px.line_polar(r=incl_counts.values, theta=incl_counts.index,
                          line_close=True, title="Top 20 Inclinations")
st.plotly_chart(fig_polar, use_container_width=True)

# SUNBURST: Country -> Purpose -> Orbit Class
st.subheader("Sunburst: Country -> Purpose -> Orbit Class")
sunburst_df = filtered_df[['Country/Organization of UN Registry', 'Purpose', 'Class of Orbit']].dropna()
fig_sunburst = px.sunburst(sunburst_df,
                           path=['Country/Organization of UN Registry', 'Purpose', 'Class of Orbit'],
                           title="Satellites Hierarchy")
st.plotly_chart(fig_sunburst, use_container_width=True)

# HEATMAP: Perigee vs Apogee
st.subheader("Heatmap: Perigee vs Apogee Density")
heatmap_df = orbital_df[['Perigee (Kilometers)', 'Apogee (Kilometers)']]
fig_heat = go.Figure(data=go.Histogram2d(
    x=heatmap_df['Perigee (Kilometers)'],
    y=heatmap_df['Apogee (Kilometers)'],
    colorscale='Viridis'
))
fig_heat.update_layout(
    xaxis_title='Perigee (km)',
    yaxis_title='Apogee (km)',
    title='Density Heatmap of Perigee vs Apogee'
)
st.plotly_chart(fig_heat, use_container_width=True)

#%% 3D Simulation
st.subheader("3D Satellite Orbit Viewer")


current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "database.csv")
df = pd.read_csv(csv_path)

orbital_df = df[['Official Name of Satellite', 'Class of Orbit', 'Perigee (Kilometers)', 'Apogee (Kilometers)', 'Inclination (Degrees)']].dropna()


orbit_classes = orbital_df['Class of Orbit'].unique()
color_map = {orbit: f"hsl({i*360/len(orbit_classes)}, 70%, 50%)" for i, orbit in enumerate(orbit_classes)}


fig = go.Figure()

for orbit_class in orbit_classes:
    subset = orbital_df[orbital_df['Class of Orbit'] == orbit_class]
    fig.add_trace(go.Scatter3d(
        x=subset['Perigee (Kilometers)'],
        y=subset['Apogee (Kilometers)'],
        z=subset['Inclination (Degrees)'],
        mode='markers',
        marker=dict(size=4, color=color_map[orbit_class]),
        name=orbit_class,
        text=subset['Official Name of Satellite'],
        hoverinfo='text'
    ))


st.sidebar.subheader("Your Satellite Orbit")
your_perigee = st.sidebar.number_input("Perigee (km)", value=400)
your_apogee = st.sidebar.number_input("Apogee (km)", value=420)
your_inclination = st.sidebar.number_input("Inclination (°)", value=51.6)

fig.add_trace(go.Scatter3d(
    x=[your_perigee],
    y=[your_apogee],
    z=[your_inclination],
    mode='markers',
    marker=dict(size=8, color='red', symbol='diamond'),
    name='Your Satellite',
    text=['Planned Orbit'],
    hoverinfo='text'
))


fig.update_layout(
    scene=dict(
        xaxis_title='Perigee (km)',
        yaxis_title='Apogee (km)',
        zaxis_title='Inclination (°)'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(title="Orbit Class")
)

st.plotly_chart(fig, use_container_width=True)


#%% Extras
import numpy as np



current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "database.csv")
df = pd.read_csv(csv_path)


df['Launch Year'] = pd.to_datetime(df['Date of Launch'], errors='coerce').dt.year
df['Expected Lifetime (Years)'] = pd.to_numeric(df['Expected Lifetime (Years)'], errors='coerce')
df['Perigee (km)'] = pd.to_numeric(df['Perigee (Kilometers)'], errors='coerce')
df['Apogee (km)'] = pd.to_numeric(df['Apogee (Kilometers)'], errors='coerce')
df['Inclination (°)'] = pd.to_numeric(df['Inclination (Degrees)'], errors='coerce')

orbital_df = df.dropna(subset=['Perigee (km)', 'Apogee (km)', 'Inclination (°)'])


st.subheader("Collision Risk")

st.markdown("Select your satellite's planned orbit to check for nearby satellites:")

user_perigee = st.number_input("Your Satellite Perigee (km)", value=400, key="risk_perigee")
user_apogee = st.number_input("Your Satellite Apogee (km)", value=420, key="risk_apogee")
user_inclination = st.number_input("Your Satellite Inclination (°)", value=51.6, key="risk_inclination")

orbital_df['Risk Score'] = np.sqrt(
    (orbital_df['Perigee (km)'] - user_perigee)**2 +
    (orbital_df['Apogee (km)'] - user_apogee)**2 +
    (orbital_df['Inclination (°)'] - user_inclination)**2
)


closest_sats = orbital_df.nsmallest(10, 'Risk Score')[['Official Name of Satellite', 'Perigee (km)', 'Apogee (km)', 'Inclination (°)', 'Risk Score']]
st.dataframe(closest_sats.style.highlight_min('Risk Score', color='red'))


st.subheader("Satellite Comparison Tool")

sat_selection = st.multiselect(
    "Select Satellites to Compare",
    options=df['Official Name of Satellite'].dropna().sort_values(),
    default=df['Official Name of Satellite'].dropna().iloc[:3]
)

if sat_selection:
    compare_df = df[df['Official Name of Satellite'].isin(sat_selection)][['Official Name of Satellite', 'Perigee (km)', 'Apogee (km)', 'Inclination (°)', 'Launch Year', 'Expected Lifetime (Years)', 'Launch Mass (Kilograms)', 'Class of Orbit', 'Type of Orbit']]
    st.dataframe(compare_df)

    
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


st.subheader("Historical Trends: Satellite Launches Over Time")

launch_counts = df.groupby('Launch Year').size().reset_index(name='Count')
fig_trend = px.line(launch_counts, x='Launch Year', y='Count', markers=True, title="Number of Satellites Launched Per Year")
st.plotly_chart(fig_trend, use_container_width=True)


st.markdown("**Country-wise Satellite Launches Over Time**")
top_countries = df['Country/Organization of UN Registry'].value_counts().nlargest(5).index
country_trend_df = df[df['Country/Organization of UN Registry'].isin(top_countries)]
country_trend_df = country_trend_df.groupby(['Launch Year', 'Country/Organization of UN Registry']).size().reset_index(name='Count')
fig_country_trend = px.line(country_trend_df, x='Launch Year', y='Count', color='Country/Organization of UN Registry', markers=True)
st.plotly_chart(fig_country_trend, use_container_width=True)



#%% Expected Lifetime
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "database.csv")
df = pd.read_csv(csv_path)

st.subheader("Satellite End-of-Life Calculator")


df['Launch Year'] = pd.to_datetime(df['Date of Launch'], errors='coerce').dt.year
df['Expected Lifetime (Years)'] = pd.to_numeric(df['Expected Lifetime (Years)'], errors='coerce')


life_df = df.dropna(subset=['Launch Year', 'Expected Lifetime (Years)'])


satellite_name = st.selectbox("Select a Satellite", life_df['Official Name of Satellite'].sort_values())

sat_info = life_df[life_df['Official Name of Satellite'] == satellite_name].iloc[0]
launch_year = int(sat_info['Launch Year'])
expected_life = int(sat_info['Expected Lifetime (Years)'])
end_year = launch_year + expected_life

st.write(f"**Satellite:** {satellite_name}")
st.write(f"**Launch Year:** {launch_year}")
st.write(f"**Expected Lifetime:** {expected_life} years")
st.success(f"**Expected End-of-Life Year:** {end_year}")
#%% Prediction
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


import os
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "database.csv")
df = pd.read_csv(csv_path)

st.subheader("Train a Prediction Model")


if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None


if st.button("Train Model"):
    st.info("Training the Random Forest model...")


    model_df = df[['Perigee (Kilometers)', 'Apogee (Kilometers)', 'Inclination (Degrees)', 'Eccentricity']].dropna()
    
    X = model_df[['Apogee (Kilometers)', 'Inclination (Degrees)', 'Eccentricity']]
    y = model_df['Perigee (Kilometers)']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


    st.session_state.model = model
    st.session_state.model_trained = True


    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.success(f"Model trained successfully! Mean Squared Error: {mse:.2f}")


if st.session_state.model_trained:
    st.subheader("Predict Perigee for Your Satellite")
    apogee_input = st.number_input("Apogee (km)", value=400.0, key="pred_apogee")
    inclination_input = st.number_input("Inclination (°)", value=51.6, key="pred_inclination")
    eccentricity_input = st.number_input("Eccentricity", value=0.001, key="pred_eccentricity")


    if st.button("Predict Perigee"):
        pred_perigee = st.session_state.model.predict([[apogee_input, inclination_input, eccentricity_input]])
        st.write(f"Predicted Perigee: {pred_perigee[0]:.2f} km")



