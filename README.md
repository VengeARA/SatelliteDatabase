Ultimate Satellite Database Dashboard
Interactive Visualization and Analysis of Global Satellites

This Streamlit app provides an interactive, data-driven exploration of active satellites using the UCS Satellite Database. It allows users to analyze, visualize, and even predict satellite orbital parameters, offering insights into orbital mechanics, global satellite distribution, and launch trends.

Features
1. Database Overview

Displays the full satellite dataset from Kaggle: UCS Satellite Database (2016)
.

Offers a searchable, scrollable dataframe preview.

2. Interactive Visualizations

Gain insights from a series of analytical charts:

Visualization	Description
Bar Chart	Top 15 countries/organizations by active satellite count
Pie Chart	Distribution of orbit types
Donut Chart	Breakdown of satellite purposes
3D Scatter Plot	Visualizes the relationship between perigee, apogee, and inclination
Scatter Matrix	Compares relationships between orbital parameters
3D Bubble Chart	Plots mass against orbital dimensions
Polar Chart	Inclination distribution of satellites
Sunburst Chart	Hierarchical view: Country → Purpose → Orbit Class
Heatmap	Density of Perigee vs Apogee values
3. 3D Orbit Simulation

Visualizes satellites in 3D orbital space.

Color-coded by orbit class.

Add your own satellite’s parameters (perigee, apogee, inclination) to see where it would fit among existing satellites.

4. Collision Risk Estimator

Enter your satellite’s orbit data.

The app calculates the distance in parameter space to nearby satellites.

Highlights the 10 closest satellites with potential risk.

5. Satellite Comparison Tool

Select multiple satellites to compare key parameters:

Perigee, Apogee, Inclination, Launch Year, Lifetime, Mass, Orbit Type.

Generates an interactive radar chart for visual comparison.

6. Historical Launch Trends

Line chart showing total satellite launches per year.

Country-wise trend comparing the top 5 satellite-launching nations.
7. End-of-Life Calculator

Choose a satellite and view:

Launch Year

Expected Lifetime

Predicted End-of-Life Year

8. Machine Learning Model

Train a Random Forest Regression model on orbital parameters.

Predict a satellite’s Perigee (km) using:

Apogee

Inclination

Eccentricity

Displays Mean Squared Error (MSE) after training.

Dataset: UCS Satellite Database (2016)

Note: The data may be slightly outdated but still serves as an excellent reference for orbital and satellite analysis.

Future Improvements

Update to 2025 datasets (via NORAD / Celestrak APIs).

Add real-time orbit tracking using TLE (Two-Line Element) data.

Improve ML prediction accuracy with additional orbital and physical parameters.

Integrate debris density analysis and sustainability visualization.

Author

Amit Raj Anand

Exploring the future of orbital analytics and AI in space.
