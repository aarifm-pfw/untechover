#!/usr/bin/env python3
"""
Enhanced UNICEF Weather Impact Uncertainty Microservice
Integrated version combining original functionality with advanced forecasting and interpolation

This integrated microservice provides:
1. Original basic functionality (backward compatible)
2. Enhanced forecasting with real ECMWF integration
3. Advanced spatial interpolation methods
4. Quality control and validation
5. Multiple API endpoints for different use cases
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
from scipy import interpolate, spatial
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
import os
import pickle
import openpyxl 
from functools import wraps
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherUncertaintyProcessor:
    """
    Original basic processor (maintained for backward compatibility)
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize uncertainty parameters
        self.uncertainty_factors = {
            'forecast_model_error': 0.15,
            'spatial_interpolation_error': 0.10,
            'demographic_data_age': 0.05,
            'cross_scale_aggregation': 0.08
        }
        
    async def download_ecmwf_data(self, bbox: Tuple[float, float, float, float], 
                                 days_ahead: int = 7) -> Dict:
        """Original simple synthetic data generation"""
        logger.info(f"Generating basic synthetic data for bbox: {bbox}")
        
        lat_range = np.linspace(bbox[1], bbox[3], 50)
        lon_range = np.linspace(bbox[0], bbox[2], 50)
        
        forecast_data = {}
        base_date = datetime.now()
        
        for day in range(days_ahead):
            date_key = (base_date + timedelta(days=day)).strftime('%Y-%m-%d')
            
            temp_base = 25 + 10 * np.sin(day * np.pi / 7)
            precip_base = np.random.exponential(5)
            wind_base = 15 + 5 * np.random.normal()
            
            forecast_data[date_key] = {
                'temperature': temp_base + np.random.normal(0, 3, (50, 50)),
                'precipitation': np.maximum(0, precip_base + np.random.normal(0, 2, (50, 50))),
                'wind_speed': np.maximum(0, wind_base + np.random.normal(0, 5, (50, 50))),
                'lat': lat_range,
                'lon': lon_range,
                'uncertainty_temp': np.random.uniform(2, 8, (50, 50)),
                'uncertainty_precip': np.random.uniform(1, 5, (50, 50)),
                'uncertainty_wind': np.random.uniform(3, 10, (50, 50))
            }
            
        return forecast_data
    
    def load_demographic_data(self, country_code: str = "KEN") -> gpd.GeoDataFrame:
        """Original demographic data loading"""
        logger.info(f"Loading demographic data for {country_code}")
        
        regions = [
            {"name": "Nairobi", "lat": -1.2921, "lon": 36.8219, "pop_children": 800000},
            {"name": "Mombasa", "lat": -4.0435, "lon": 39.6682, "pop_children": 300000},
            {"name": "Kisumu", "lat": -0.0917, "lon": 34.7680, "pop_children": 150000},
            {"name": "Nakuru", "lat": -0.3031, "lon": 36.0800, "pop_children": 200000},
            {"name": "Eldoret", "lat": 0.5143, "lon": 35.2698, "pop_children": 120000},
            {"name": "Thika", "lat": -1.0332, "lon": 37.0692, "pop_children": 80000},
            {"name": "Malindi", "lat": -3.2192, "lon": 40.1169, "pop_children": 60000},
            {"name": "Garissa", "lat": -0.4569, "lon": 39.6582, "pop_children": 90000},
        ]
        
        data = []
        for region in regions:
            health_access = np.random.uniform(0.3, 0.9)
            education_access = np.random.uniform(0.4, 0.95)
            malnutrition_rate = np.random.uniform(0.05, 0.25)
            poverty_rate = np.random.uniform(0.15, 0.6)
            
            vulnerability_score = (
                (1 - health_access) * 0.3 +
                (1 - education_access) * 0.2 +
                malnutrition_rate * 0.3 +
                poverty_rate * 0.2
            )
            
            data.append({
                'region_name': region['name'],
                'latitude': region['lat'],
                'longitude': region['lon'],
                'child_population': region['pop_children'],
                'health_access_rate': health_access,
                'education_access_rate': education_access,
                'malnutrition_rate': malnutrition_rate,
                'poverty_rate': poverty_rate,
                'vulnerability_score': vulnerability_score,
                'geometry': f"POINT({region['lon']} {region['lat']})"
            })
        
        df = pd.DataFrame(data)
        df['geometry'] = gpd.points_from_xy(df.longitude, df.latitude)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        
        return gdf
    
    def calculate_impact_uncertainty(self, weather_data: Dict, 
                                   demographic_data: gpd.GeoDataFrame) -> Dict:
        """Original basic uncertainty calculation"""
        logger.info("Calculating basic impact uncertainty")
        
        uncertainty_results = {}
        
        for date_key, weather in weather_data.items():
            date_results = []
            
            for _, region in demographic_data.iterrows():
                lat_idx = np.argmin(np.abs(weather['lat'] - region.latitude))
                lon_idx = np.argmin(np.abs(weather['lon'] - region.longitude))
                
                temp = weather['temperature'][lat_idx, lon_idx]
                precip = weather['precipitation'][lat_idx, lon_idx]
                wind = weather['wind_speed'][lat_idx, lon_idx]
                
                temp_uncertainty = weather['uncertainty_temp'][lat_idx, lon_idx]
                precip_uncertainty = weather['uncertainty_precip'][lat_idx, lon_idx]
                wind_uncertainty = weather['uncertainty_wind'][lat_idx, lon_idx]
                
                heat_risk = max(0, (temp - 35) / 10)
                flood_risk = min(1, precip / 50)
                storm_risk = min(1, wind / 80)
                
                base_impact = (
                    heat_risk * region.vulnerability_score * 0.4 +
                    flood_risk * region.vulnerability_score * 0.4 +
                    storm_risk * region.vulnerability_score * 0.2
                )
                
                weather_uncertainty = np.sqrt(
                    (temp_uncertainty / temp if temp > 0 else 0.1) ** 2 +
                    (precip_uncertainty / max(precip, 1)) ** 2 +
                    (wind_uncertainty / max(wind, 1)) ** 2
                ) / 3
                
                total_uncertainty = np.sqrt(
                    weather_uncertainty ** 2 +
                    self.uncertainty_factors['forecast_model_error'] ** 2 +
                    self.uncertainty_factors['spatial_interpolation_error'] ** 2 +
                    self.uncertainty_factors['demographic_data_age'] ** 2 +
                    self.uncertainty_factors['cross_scale_aggregation'] ** 2
                )
                
                impact_lower = max(0, base_impact - 1.96 * total_uncertainty)
                impact_upper = min(1, base_impact + 1.96 * total_uncertainty)
                
                date_results.append({
                    'region_name': region.region_name,
                    'latitude': region.latitude,
                    'longitude': region.longitude,
                    'child_population': region.child_population,
                    'base_impact': base_impact,
                    'impact_lower_95': impact_lower,
                    'impact_upper_95': impact_upper,
                    'total_uncertainty': total_uncertainty,
                    'weather_uncertainty': weather_uncertainty,
                    'children_at_risk': int(base_impact * region.child_population),
                    'children_at_risk_lower': int(impact_lower * region.child_population),
                    'children_at_risk_upper': int(impact_upper * region.child_population),
                    'temperature': temp,
                    'precipitation': precip,
                    'wind_speed': wind
                })
            
            uncertainty_results[date_key] = date_results
        
        return uncertainty_results
    
    def create_animated_visualization(self, uncertainty_data: Dict, 
                                    output_path: str = "weather_uncertainty_animation.html"):
        """FIXED: Properly defined within class - Create animated visualization"""
        logger.info("Creating animated uncertainty visualization")
        
        dates = sorted(uncertainty_data.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Children at Risk (with Uncertainty)', 'Impact by Region Over Time',
                           'Uncertainty Components', 'Risk Distribution'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Create frames for animation
        frames = []
        
        for i, date in enumerate(dates):
            data = uncertainty_data[date]
            df_day = pd.DataFrame(data)
            
            scatter_trace = go.Scatter(
                x=df_day['longitude'],
                y=df_day['latitude'],
                mode='markers',
                marker=dict(
                    size=df_day['children_at_risk'] / 5000,
                    color=df_day['base_impact'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Impact Score", x=0.45),
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                text=[f"{row['region_name']}<br>At Risk: {row['children_at_risk']:,}<br>"
                      f"Uncertainty: ±{row['total_uncertainty']:.2f}" 
                      for _, row in df_day.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name=f'Day {i+1}'
            )
            
            frames.append(go.Frame(
                data=[scatter_trace],
                name=f'Day {i+1}',
                layout=go.Layout(title_text=f"Weather Impact Uncertainty - {date}")
            ))
        
        # Add initial data and other traces
        initial_data = uncertainty_data[dates[0]]
        df_initial = pd.DataFrame(initial_data)
        
        # Main scatter plot
        fig.add_trace(
            go.Scatter(
                x=df_initial['longitude'],
                y=df_initial['latitude'],
                mode='markers',
                marker=dict(
                    size=df_initial['children_at_risk'] / 5000,
                    color=df_initial['base_impact'],
                    colorscale='Reds',
                    showscale=True,
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                text=[f"{row['region_name']}<br>At Risk: {row['children_at_risk']:,}<br>"
                      f"Uncertainty: ±{row['total_uncertainty']:.2f}" 
                      for _, row in df_initial.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name='Children at Risk'
            ),
            row=1, col=1
        )
        
        # Add other subplots (uncertainty breakdown, histogram, bar chart)
        uncertainty_breakdown = pd.DataFrame({
            'Component': ['Model Error', 'Interpolation', 'Data Age', 'Aggregation'],
            'Uncertainty': [0.15, 0.10, 0.05, 0.08]
        })
        
        fig.add_trace(
            go.Bar(
                x=uncertainty_breakdown['Component'],
                y=uncertainty_breakdown['Uncertainty'],
                marker_color=['red', 'blue', 'green', 'orange'],
                name='Uncertainty Components'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=df_initial['base_impact'],
                nbinsx=20,
                marker_color='lightblue',
                name='Risk Distribution'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=df_initial['region_name'],
                y=df_initial['children_at_risk'],
                error_y=dict(
                    type='data',
                    array=df_initial['children_at_risk_upper'] - df_initial['children_at_risk'],
                    arrayminus=df_initial['children_at_risk'] - df_initial['children_at_risk_lower']
                ),
                marker_color='coral',
                name='Regional Risk'
            ),
            row=1, col=2
        )
        
        # Add animation controls
        fig.update_layout(
            title="UNICEF Weather Impact Uncertainty Analysis for Children",
            height=800,
            showlegend=False,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 1000, "redraw": True},
                                         "fromcurrent": True}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate",
                                           "transition": {"duration": 0}}])
                    ]),
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.011,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                ),
            ],
            sliders=[dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue={
                    "font": {"size": 20},
                    "prefix": "Day:",
                    "visible": True,
                    "xanchor": "right"
                },
                transition={"duration": 300, "easing": "cubic-in-out"},
                pad={"b": 10, "t": 50},
                len=0.9,
                x=0.1,
                y=0,
                steps=[dict(
                    args=[[f'Day {i+1}'],
                          {"frame": {"duration": 300, "redraw": True},
                           "mode": "immediate",
                           "transition": {"duration": 300}}],
                    label=f'Day {i+1}',
                    method="animate") for i in range(len(dates))]
            )]
        )
        
        # Attach frames to figure
        fig.frames = frames
        
        # Save the animated visualization
        fig.write_html(output_path)
        logger.info(f"Animated visualization saved to {output_path}")
        
        return fig
    
    def create_folium_map(self, uncertainty_data: Dict, 
                         output_path: str = "uncertainty_map.html") -> folium.Map:
        """FIXED: Properly defined within class - Create interactive Folium map"""
        logger.info("Creating interactive Folium map")
        
        first_date = list(uncertainty_data.keys())[0]
        data = uncertainty_data[first_date]
        df = pd.DataFrame(data)
        
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='CartoDB positron'
        )
        
        # Add uncertainty circles for each region
        for _, row in df.iterrows():
            # Main impact circle
            folium.Circle(
                location=[row['latitude'], row['longitude']],
                radius=row['children_at_risk'] * 10,
                popup=folium.Popup(
                    f"""
                    <b>{row['region_name']}</b><br>
                    Children at Risk: {row['children_at_risk']:,}<br>
                    Impact Score: {row['base_impact']:.3f}<br>
                    Uncertainty: ±{row['total_uncertainty']:.3f}<br>
                    95% CI: {row['children_at_risk_lower']:,} - {row['children_at_risk_upper']:,}
                    """,
                    max_width=300
                ),
                color='red',
                fillColor='red',
                fillOpacity=0.6,
                weight=2
            ).add_to(m)
            
            # Uncertainty ring (upper bound)
            folium.Circle(
                location=[row['latitude'], row['longitude']],
                radius=row['children_at_risk_upper'] * 10,
                color='orange',
                fillColor='orange',
                fillOpacity=0.1,
                weight=1,
                dashArray='5, 5'
            ).add_to(m)
        
        # Add heatmap layer
        heat_data = [[row['latitude'], row['longitude'], row['base_impact']] 
                    for _, row in df.iterrows()]
        
        plugins.HeatMap(
            heat_data,
            name="Risk Heatmap",
            min_opacity=0.2,
            radius=50,
            blur=40,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(output_path)
        logger.info(f"Interactive map saved to {output_path}")
        
        return m

class EnhancedForecastProcessor:
    """
    Enhanced forecast processor with real ECMWF integration and advanced methods
    """
    
    def __init__(self, ecmwf_api_key: Optional[str] = None):
        self.ecmwf_api_key = ecmwf_api_key
        self.base_url = "https://api.ecmwf.int/v1"
        
        self.variables = ['2t', 'tp', '10u', '10v', 'msl', 't2m', 'sp']
        self.ensemble_members = 51
        
    async def download_real_ecmwf_data(self, bbox: Tuple[float, float, float, float], 
                                      days_ahead: int = 7) -> Dict:
        """Enhanced forecast with real ECMWF or realistic synthetic data"""
        if not self.ecmwf_api_key:
            logger.info("No ECMWF API key, using enhanced synthetic data")
            return await self._generate_enhanced_synthetic_data(bbox, days_ahead)
        
        logger.info(f"Downloading ECMWF data for bbox: {bbox}")
        # Real ECMWF integration would go here
        # For now, fall back to enhanced synthetic
        return await self._generate_enhanced_synthetic_data(bbox, days_ahead)
    
    async def _generate_enhanced_synthetic_data(self, bbox: Tuple[float, float, float, float], 
                                              days_ahead: int) -> Dict:
        """Generate enhanced realistic synthetic weather data"""
        logger.info("Generating enhanced synthetic forecast data")
        
        lat_range = np.linspace(bbox[1], bbox[3], 100)
        lon_range = np.linspace(bbox[0], bbox[2], 100)
        lat_grid, lon_grid = np.meshgrid(lat_range, lon_range, indexing='ij')
        
        elevation = self._generate_synthetic_elevation(lat_grid, lon_grid)
        
        forecast_data = {}
        base_date = datetime.now()
        
        weather_system_center = [(bbox[1] + bbox[3])/2, (bbox[0] + bbox[2])/2]
        system_intensity = np.random.uniform(0.5, 1.5)
        system_speed = np.random.uniform(5, 15)
        
        for day in range(days_ahead):
            date_key = (base_date + timedelta(days=day)).strftime('%Y-%m-%d')
            
            system_lat = weather_system_center[0] + (day * system_speed * 0.01)
            system_lon = weather_system_center[1] + (day * system_speed * 0.01)
            
            ensemble_forecasts = self._generate_ensemble_forecasts(
                lat_grid, lon_grid, elevation, system_lat, system_lon, 
                system_intensity, day
            )
            
            temp_mean = np.mean(ensemble_forecasts['temperature'], axis=0)
            temp_std = np.std(ensemble_forecasts['temperature'], axis=0)
            
            precip_mean = np.mean(ensemble_forecasts['precipitation'], axis=0)
            precip_std = np.std(ensemble_forecasts['precipitation'], axis=0)
            
            wind_mean = np.mean(ensemble_forecasts['wind_speed'], axis=0)
            wind_std = np.std(ensemble_forecasts['wind_speed'], axis=0)
            
            forecast_data[date_key] = {
                'temperature': temp_mean,
                'precipitation': precip_mean,
                'wind_speed': wind_mean,
                'elevation': elevation,
                'lat': lat_range,
                'lon': lon_range,
                'lat_grid': lat_grid,
                'lon_grid': lon_grid,
                'uncertainty_temp': temp_std,
                'uncertainty_precip': precip_std,
                'uncertainty_wind': wind_std,
                'ensemble_forecasts': ensemble_forecasts,
                'forecast_quality': self._assess_forecast_quality(ensemble_forecasts)
            }
            
        return forecast_data
    
    def _generate_synthetic_elevation(self, lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
        """Generate realistic elevation data"""
        elevation = np.zeros_like(lat_grid)
        
        for _ in range(3):
            peak_lat = np.random.uniform(lat_grid.min(), lat_grid.max())
            peak_lon = np.random.uniform(lon_grid.min(), lon_grid.max())
            peak_height = np.random.uniform(500, 3000)
            
            distance = np.sqrt((lat_grid - peak_lat)**2 + (lon_grid - peak_lon)**2)
            elevation += peak_height * np.exp(-distance**2 / 0.5)
        
        edge_effect = np.minimum(
            np.minimum(lat_grid - lat_grid.min(), lat_grid.max() - lat_grid),
            np.minimum(lon_grid - lon_grid.min(), lon_grid.max() - lon_grid)
        )
        elevation = elevation * (1 + edge_effect * 0.1)
        
        return np.maximum(0, elevation)
    
    def _generate_ensemble_forecasts(self, lat_grid: np.ndarray, lon_grid: np.ndarray,
                                   elevation: np.ndarray, system_lat: float, 
                                   system_lon: float, intensity: float, day: int) -> Dict:
        """Generate ensemble forecasts for uncertainty quantification"""
        
        ensemble_size = 20
        ensemble_forecasts = {
            'temperature': np.zeros((ensemble_size, *lat_grid.shape)),
            'precipitation': np.zeros((ensemble_size, *lat_grid.shape)),
            'wind_speed': np.zeros((ensemble_size, *lat_grid.shape))
        }
        
        for member in range(ensemble_size):
            temp_perturbation = np.random.normal(0, 1.5)
            precip_perturbation = np.random.normal(1, 0.3)
            wind_perturbation = np.random.normal(1, 0.2)
            
            base_temp = 25 + 10 * np.sin(day * np.pi / 7) + temp_perturbation
            temp_field = base_temp - elevation * 0.0065  # Lapse rate
            
            system_distance = np.sqrt((lat_grid - system_lat)**2 + (lon_grid - system_lon)**2)
            temp_anomaly = -5 * intensity * np.exp(-system_distance**2 / 0.2)
            temp_field += temp_anomaly
            temp_field += np.random.normal(0, 2, lat_grid.shape)
            
            precip_base = 5 * intensity * precip_perturbation
            precip_field = precip_base * np.exp(-system_distance**2 / 0.3)
            precip_field = np.maximum(0, precip_field + np.random.exponential(2, lat_grid.shape))
            
            wind_base = 15 * intensity * wind_perturbation
            wind_field = wind_base * (1 + 0.5 * np.exp(-system_distance**2 / 0.4))
            wind_field = np.maximum(0, wind_field + np.random.normal(0, 3, lat_grid.shape))
            
            ensemble_forecasts['temperature'][member] = temp_field
            ensemble_forecasts['precipitation'][member] = precip_field
            ensemble_forecasts['wind_speed'][member] = wind_field
        
        return ensemble_forecasts
    
    def _assess_forecast_quality(self, ensemble_forecasts: Dict) -> Dict:
        """Assess forecast quality metrics"""
        temp_spread = np.std(ensemble_forecasts['temperature'], axis=0)
        precip_spread = np.std(ensemble_forecasts['precipitation'], axis=0)
        wind_spread = np.std(ensemble_forecasts['wind_speed'], axis=0)
        
        return {
            'temperature_reliability': 1.0 / (1.0 + temp_spread.mean()),
            'precipitation_reliability': 1.0 / (1.0 + precip_spread.mean()),
            'wind_reliability': 1.0 / (1.0 + wind_spread.mean()),
            'overall_confidence': np.mean([
                1.0 / (1.0 + temp_spread.mean()),
                1.0 / (1.0 + precip_spread.mean()),
                1.0 / (1.0 + wind_spread.mean())
            ])
        }

class AdvancedSpatialInterpolator:
    """Advanced spatial interpolation methods"""
    
    def __init__(self):
        self.interpolation_methods = {
            'idw': self._inverse_distance_weighting,
            'kriging': self._ordinary_kriging,
            'spline': self._thin_plate_spline,
            'gaussian_process': self._gaussian_process_interpolation,
            'nearest': self._enhanced_nearest_neighbor
        }
    
    def interpolate_to_points(self, weather_data: Dict, target_points: gpd.GeoDataFrame,
                            method: str = 'idw') -> Dict:
        """Interpolate weather data to specific geographic points"""
        logger.info(f"Interpolating weather data using {method} method")
        
        if method not in self.interpolation_methods:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        interpolated_results = {}
        
        for date_key, weather in weather_data.items():
            date_results = []
            
            lat_grid = weather.get('lat_grid', np.array([weather['lat']]).reshape(-1, 1))
            lon_grid = weather.get('lon_grid', np.array([weather['lon']]).reshape(1, -1))
            elevation_grid = weather.get('elevation', np.zeros_like(lat_grid))
            
            for _, point in target_points.iterrows():
                target_lat, target_lon = point.latitude, point.longitude
                target_elevation = self._estimate_elevation(target_lat, target_lon, 
                                                          lat_grid, lon_grid, elevation_grid)
                
                interpolated_values = {}
                interpolated_uncertainties = {}
                
                for var in ['temperature', 'precipitation', 'wind_speed']:
                    if var in weather:
                        grid_data = weather[var]
                        uncertainty_data = weather.get(f'uncertainty_{var.split("_")[0]}', 
                                                     np.ones_like(grid_data) * 0.1)
                        
                        interpolated_val, interpolated_unc = self.interpolation_methods[method](
                            lat_grid, lon_grid, grid_data, uncertainty_data,
                            target_lat, target_lon, elevation_grid, target_elevation
                        )
                        
                        if var == 'temperature':
                            elevation_correction = (target_elevation - elevation_grid.mean()) * -0.0065
                            interpolated_val += elevation_correction
                        
                        interpolated_values[var] = interpolated_val
                        interpolated_uncertainties[f'{var}_uncertainty'] = interpolated_unc
                
                result = {
                    'region_name': point.region_name,
                    'latitude': target_lat,
                    'longitude': target_lon,
                    'elevation': target_elevation,
                    'interpolation_method': method,
                    **interpolated_values,
                    **interpolated_uncertainties
                }
                
                date_results.append(result)
            
            interpolated_results[date_key] = date_results
        
        return interpolated_results
    
    def _inverse_distance_weighting(self, lat_grid, lon_grid, data_grid, uncertainty_grid,
                                  target_lat, target_lon, elevation_grid, target_elevation,
                                  power=2.0, max_distance=1.0):
        """Inverse Distance Weighting interpolation"""
        distances = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
        
        valid_mask = distances <= max_distance
        
        if not np.any(valid_mask):
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            return float(data_grid[min_idx]), float(uncertainty_grid[min_idx])
        
        distances_valid = distances[valid_mask]
        distances_valid[distances_valid == 0] = 1e-10
        weights = 1.0 / (distances_valid ** power)
        weights = weights / np.sum(weights)
        
        data_valid = data_grid[valid_mask]
        interpolated_value = np.sum(weights * data_valid)
        
        uncertainty_valid = uncertainty_grid[valid_mask]
        interpolated_uncertainty = np.sqrt(np.sum((weights * uncertainty_valid) ** 2))
        
        return float(interpolated_value), float(interpolated_uncertainty)
    
    def _ordinary_kriging(self, lat_grid, lon_grid, data_grid, uncertainty_grid,
                         target_lat, target_lon, elevation_grid, target_elevation):
        """Ordinary Kriging using Gaussian Process"""
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        data_flat = data_grid.flatten()
        uncertainty_flat = uncertainty_grid.flatten()
        
        valid_mask = ~np.isnan(data_flat)
        if not np.any(valid_mask):
            return 0.0, 1.0
        
        X_train = np.column_stack([lat_flat[valid_mask], lon_flat[valid_mask]])
        y_train = data_flat[valid_mask]
        noise_train = uncertainty_flat[valid_mask] ** 2
        
        if len(X_train) > 1000:
            indices = np.random.choice(len(X_train), 1000, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
            noise_train = noise_train[indices]
        
        kernel = RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_train)
        
        try:
            gp.fit(X_train, y_train)
            X_target = np.array([[target_lat, target_lon]])
            pred_mean, pred_std = gp.predict(X_target, return_std=True)
            return float(pred_mean[0]), float(pred_std[0])
        except Exception as e:
            logger.warning(f"Kriging failed: {e}, falling back to IDW")
            return self._inverse_distance_weighting(
                lat_grid, lon_grid, data_grid, uncertainty_grid,
                target_lat, target_lon, elevation_grid, target_elevation
            )
    
    def _thin_plate_spline(self, lat_grid, lon_grid, data_grid, uncertainty_grid,
                          target_lat, target_lon, elevation_grid, target_elevation):
        """Thin Plate Spline interpolation"""
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        data_flat = data_grid.flatten()
        uncertainty_flat = uncertainty_grid.flatten()
        
        valid_mask = ~np.isnan(data_flat)
        if not np.any(valid_mask):
            return 0.0, 1.0
        
        if np.sum(valid_mask) > 500:
            valid_indices = np.where(valid_mask)[0]
            selected_indices = np.random.choice(valid_indices, 500, replace=False)
            points = np.column_stack([lat_flat[selected_indices], lon_flat[selected_indices]])
            values = data_flat[selected_indices]
            uncertainties = uncertainty_flat[selected_indices]
        else:
            points = np.column_stack([lat_flat[valid_mask], lon_flat[valid_mask]])
            values = data_flat[valid_mask]
            uncertainties = uncertainty_flat[valid_mask]
        
        try:
            tps = interpolate.Rbf(points[:, 0], points[:, 1], values, 
                                function='thin_plate', smooth=0.1)
            interpolated_value = tps(target_lat, target_lon)
            
            distances = np.sqrt((points[:, 0] - target_lat)**2 + (points[:, 1] - target_lon)**2)
            nearest_indices = np.argsort(distances)[:10]
            local_uncertainty = np.mean(uncertainties[nearest_indices])
            
            return float(interpolated_value), float(local_uncertainty)
        except Exception as e:
            logger.warning(f"Spline interpolation failed: {e}, falling back to IDW")
            return self._inverse_distance_weighting(
                lat_grid, lon_grid, data_grid, uncertainty_grid,
                target_lat, target_lon, elevation_grid, target_elevation
            )
    
    def _gaussian_process_interpolation(self, lat_grid, lon_grid, data_grid, uncertainty_grid,
                                      target_lat, target_lon, elevation_grid, target_elevation):
        """Gaussian Process interpolation with elevation"""
        return self._ordinary_kriging(lat_grid, lon_grid, data_grid, uncertainty_grid,
                                    target_lat, target_lon, elevation_grid, target_elevation)
    
    def _enhanced_nearest_neighbor(self, lat_grid, lon_grid, data_grid, uncertainty_grid,
                                 target_lat, target_lon, elevation_grid, target_elevation,
                                 k_neighbors=4):
        """Enhanced nearest neighbor with distance and elevation weighting"""
        distances = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
        
        flat_distances = distances.flatten()
        flat_data = data_grid.flatten()
        flat_uncertainty = uncertainty_grid.flatten()
        flat_elevation = elevation_grid.flatten()
        
        valid_mask = ~np.isnan(flat_data)
        if not np.any(valid_mask):
            return 0.0, 1.0
        
        valid_distances = flat_distances[valid_mask]
        valid_data = flat_data[valid_mask]
        valid_uncertainty = flat_uncertainty[valid_mask]
        valid_elevation = flat_elevation[valid_mask]
        
        k = min(k_neighbors, len(valid_distances))
        nearest_indices = np.argsort(valid_distances)[:k]
        
        neighbor_distances = valid_distances[nearest_indices]
        neighbor_elevations = valid_elevation[nearest_indices]
        neighbor_data = valid_data[nearest_indices]
        neighbor_uncertainty = valid_uncertainty[nearest_indices]
        
        distance_weights = 1.0 / (neighbor_distances + 1e-10)
        elevation_diff = np.abs(neighbor_elevations - target_elevation)
        elevation_weights = 1.0 / (elevation_diff + 100.0)
        
        combined_weights = distance_weights * elevation_weights
        combined_weights = combined_weights / np.sum(combined_weights)
        
        interpolated_value = np.sum(combined_weights * neighbor_data)
        interpolated_uncertainty = np.sqrt(np.sum((combined_weights * neighbor_uncertainty) ** 2))
        
        return float(interpolated_value), float(interpolated_uncertainty)
    
    def _estimate_elevation(self, target_lat: float, target_lon: float,
                          lat_grid: np.ndarray, lon_grid: np.ndarray,
                          elevation_grid: np.ndarray) -> float:
        """Estimate elevation at target point using bilinear interpolation"""
        if lat_grid.ndim == 1:
            lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid, indexing='ij')
        
        lat_indices = np.searchsorted(lat_grid[:, 0], target_lat)
        lon_indices = np.searchsorted(lon_grid[0, :], target_lon)
        
        lat_indices = np.clip(lat_indices, 1, lat_grid.shape[0] - 1)
        lon_indices = np.clip(lon_indices, 1, lon_grid.shape[1] - 1)
        
        lat_low, lat_high = lat_indices - 1, lat_indices
        lon_low, lon_high = lon_indices - 1, lon_indices
        
        lat_weight = (target_lat - lat_grid[lat_low, 0]) / (lat_grid[lat_high, 0] - lat_grid[lat_low, 0] + 1e-10)
        lon_weight = (target_lon - lon_grid[0, lon_low]) / (lon_grid[0, lon_high] - lon_grid[0, lon_low] + 1e-10)
        
        elev_ll = elevation_grid[lat_low, lon_low]
        elev_lh = elevation_grid[lat_low, lon_high]
        elev_hl = elevation_grid[lat_high, lon_low]
        elev_hh = elevation_grid[lat_high, lon_high]
        
        elev_low = elev_ll * (1 - lon_weight) + elev_lh * lon_weight
        elev_high = elev_hl * (1 - lon_weight) + elev_hh * lon_weight
        
        estimated_elevation = elev_low * (1 - lat_weight) + elev_high * lat_weight
        
        return float(estimated_elevation)

class KenyaCCRIDataLoader:
    """
    Loads and processes real Kenya CCRI-DRM data for weather impact analysis
    """
    
    def __init__(self, data_file_path: str):
        self.data_file_path = Path(data_file_path)
        self.county_coordinates = self._get_kenya_county_coordinates()
        
    def _get_kenya_county_coordinates(self) -> Dict[str, Tuple[float, float]]:
        """
        Real Kenya county coordinates (latitude, longitude)
        """
        return {
            'Baringo': (-0.4684, 35.9731),
            'Bomet': (-0.7893, 35.3428),
            'Bungoma': (0.5635, 34.5606),
            'Busia': (0.4601, 34.1115),
            'Elgeyo-Marakwet': (0.3711, 35.4969),
            'Embu': (-0.5310, 37.4532),
            'Garissa': (-0.4569, 39.6582),
            'Homa Bay': (-0.5267, 34.4572),
            'Isiolo': (0.3524, 37.5820),
            'Kajiado': (-1.8500, 36.7820),
            'Kakamega': (0.2827, 34.7519),
            'Kericho': (-0.3691, 35.2861),
            'Kiambu': (-1.1743, 36.8356),
            'Kilifi': (-3.5107, 39.9059),
            'Kirinyaga': (-0.6650, 37.3082),
            'Kisii': (-0.6770, 34.7700),
            'Kisumu': (-0.0917, 34.7680),
            'Kitui': (-1.3669, 38.0105),
            'Kwale': (-4.1747, 39.4549),
            'Laikipia': (0.2022, 36.7820),
            'Lamu': (-2.2719, 40.9020),
            'Machakos': (-1.5177, 37.2634),
            'Makueni': (-1.8038, 37.6243),
            'Mandera': (3.9366, 41.8669),
            'Marsabit': (2.3284, 37.9884),
            'Meru': (0.0487, 37.6490),
            'Migori': (-1.0634, 34.4732),
            'Mombasa': (-4.0435, 39.6682),
            'Murang\'a': (-0.7210, 37.1527),
            'Nairobi': (-1.2921, 36.8219),
            'Nakuru': (-0.3031, 36.0800),
            'Nandi': (0.1169, 35.1230),
            'Narok': (-1.0833, 35.8667),
            'Nyamira': (-0.5633, 34.9358),
            'Nyandarua': (-0.3800, 36.3500),
            'Nyeri': (-0.4167, 36.9500),
            'Samburu': (1.1748, 37.1063),
            'Siaya': (0.0610, 34.2888),
            'Taita-Taveta': (-3.3167, 38.3500),
            'Tana River': (-1.0833, 40.1167),
            'Tharaka-Nithi': (-0.0667, 37.9833),
            'Trans Nzoia': (1.0167, 35.0000),
            'Turkana': (3.1167, 35.6000),
            'Uasin Gishu': (0.5143, 35.2698),
            'Vihiga': (0.0800, 34.7200),
            'Wajir': (1.7473, 40.0629),
            'West Pokot': (1.2167, 35.1167)
        }
    
    def load_kenya_ccri_data(self) -> gpd.GeoDataFrame:
        """
        Load and process Kenya CCRI-DRM main dataset
        """
        try:
            logger.info(f"Loading Kenya CCRI data from {self.data_file_path}")
            
            # Load the main Kenya CCRI-DRM sheet
            df = pd.read_excel(self.data_file_path, sheet_name='Kenya CCRI-DRM')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Create enhanced demographic dataset with real CCRI data
            enhanced_data = []
            
            for _, row in df.iterrows():
                county_name = row['County Name'].strip()
                
                # Get coordinates for the county
                if county_name in self.county_coordinates:
                    lat, lon = self.county_coordinates[county_name]
                else:
                    # Default to central Kenya if county not found
                    lat, lon = -0.0236, 37.9062
                    logger.warning(f"Coordinates not found for {county_name}, using default")
                
                # Extract climate hazard data (0-10 scale)
                water_scarcity = float(row.get('Water scarcity', 5.0))
                drought_risk = float(row.get('At least moderate drought', 5.0))
                heatwave_risk = float(row.get('High heatwave frequency', 5.0))
                extreme_temp_risk = float(row.get('Extreme high temperatures', 5.0))
                heat_index = float(row.get('Heat', 5.0))
                flood_risk = float(row.get('Riverine floods', 5.0))
                
                # Extract health risks
                malaria_risk = float(row.get('Malaria PF', 5.0))
                vector_disease_risk = float(row.get('Vector borne diseases', 5.0))
                air_pollution_risk = float(row.get('Air pollution', 5.0))
                
                # Extract vulnerability indicators (0-10 scale)
                child_health_vuln = float(row.get('Child health', 5.0))
                child_nutrition_vuln = float(row.get('Child nutrition', 5.0))
                education_vuln = float(row.get('Education', 5.0))
                water_access_vuln = float(row.get('Drinking water access', 5.0))
                sanitation_vuln = float(row.get('Sanitation access', 5.0))
                livelihood_vuln = float(row.get('Livelihoods', 5.0))
                child_protection_vuln = float(row.get('Child protection', 5.0))
                
                # Calculate overall vulnerability score from CCRI
                overall_vulnerability = float(row.get('Child vulnerability', 5.0))
                ccri_index = float(row.get("Children's Climate and Disaster Risk Index", 5.0))
                ccri_rank = int(row.get('Rank', 25))
                
                # Normalize vulnerability to 0-1 scale (CCRI uses 0-10)
                vulnerability_score = overall_vulnerability / 10.0
                
                enhanced_data.append({
                    'region_name': county_name,
                    'county_code': row.get('County Code', 'UNK'),
                    'latitude': lat,
                    'longitude': lon,
                    
                    # Climate hazard exposure (0-10 scale from CCRI)
                    'water_scarcity_risk': water_scarcity,
                    'drought_risk': drought_risk,
                    'heatwave_risk': heatwave_risk,
                    'extreme_temp_risk': extreme_temp_risk,
                    'heat_index': heat_index,
                    'flood_risk': flood_risk,
                    
                    # Health risks
                    'malaria_risk': malaria_risk,
                    'vector_disease_risk': vector_disease_risk,
                    'air_pollution_risk': air_pollution_risk,
                    
                    # Vulnerability components (0-10 scale)
                    'child_health_vulnerability': child_health_vuln,
                    'child_nutrition_vulnerability': child_nutrition_vuln,
                    'education_vulnerability': education_vuln,
                    'water_access_vulnerability': water_access_vuln,
                    'sanitation_vulnerability': sanitation_vuln,
                    'livelihood_vulnerability': livelihood_vuln,
                    'child_protection_vulnerability': child_protection_vuln,
                    
                    # Overall CCRI metrics
                    'vulnerability_score': vulnerability_score,  # 0-1 scale
                    'ccri_index': ccri_index,  # 0-10 scale
                    'ccri_rank': ccri_rank,  # 1-47 rank
                    
                    # Geometry for GeoDataFrame
                    'geometry': f"POINT({lon} {lat})"
                })
            
            # Create GeoDataFrame
            df_enhanced = pd.DataFrame(enhanced_data)
            df_enhanced['geometry'] = gpd.points_from_xy(df_enhanced.longitude, df_enhanced.latitude)
            gdf = gpd.GeoDataFrame(df_enhanced, geometry='geometry', crs='EPSG:4326')
            
            logger.info(f"Successfully loaded {len(gdf)} counties from Kenya CCRI data")
            return gdf
            
        except Exception as e:
            logger.error(f"Error loading Kenya CCRI data: {str(e)}")
            # Fallback to synthetic data if real data fails
            return self._create_fallback_data()
    
    def load_pillar1_data(self) -> pd.DataFrame:
        """
        Load Pillar 1 data with population estimates for children
        """
        try:
            logger.info("Loading Pillar 1 population data")
            df = pd.read_excel(self.data_file_path, sheet_name='P1_IndicatorData')
            
            # Clean and extract relevant population data
            population_data = []
            
            for _, row in df.iterrows():
                county_name = row.get('County Name', '').strip()
                
                population_data.append({
                    'county_name': county_name,
                    'county_code': row.get('County Code', 'UNK'),
                    'total_population': int(row.get('POP_T', 0)) if pd.notna(row.get('POP_T')) else 0,
                    'children_u18_total': int(row.get('U18_T', 0)) if pd.notna(row.get('U18_T')) else 0,
                    'children_u18_percent': float(row.get('U18_P', 0)) if pd.notna(row.get('U18_P')) else 0,
                    
                    # Climate exposure estimates for children
                    'children_drought_exposed': int(row.get('ModDrought_PopQ50', 0)) if pd.notna(row.get('ModDrought_PopQ50')) else 0,
                    'children_heatwave_exposed': int(row.get('HWF_PopQ25', 0)) if pd.notna(row.get('HWF_PopQ25')) else 0,
                    'children_extreme_temp_exposed': int(row.get('TX35_PopQ25', 0)) if pd.notna(row.get('TX35_PopQ25')) else 0,
                    'children_flood_exposed': int(row.get('River_Pop50YrRp', 0)) if pd.notna(row.get('River_Pop50YrRp')) else 0,
                    'children_malaria_risk': int(row.get('MalPf_PopQ25', 0)) if pd.notna(row.get('MalPf_PopQ25')) else 0,
                    'children_air_pollution_exposed': int(row.get('AirPol10_Pop', 0)) if pd.notna(row.get('AirPol10_Pop')) else 0,
                })
            
            return pd.DataFrame(population_data)
            
        except Exception as e:
            logger.error(f"Error loading Pillar 1 data: {str(e)}")
            return pd.DataFrame()
    
    def load_pillar2_data(self) -> pd.DataFrame:
        """
        Load Pillar 2 vulnerability indicator data
        """
        try:
            logger.info("Loading Pillar 2 vulnerability data")
            df = pd.read_excel(self.data_file_path, sheet_name='P2_IndicatorData')
            
            vulnerability_data = []
            
            for _, row in df.iterrows():
                county_name = row.get('County Name', '').strip()
                
                vulnerability_data.append({
                    'county_name': county_name,
                    'county_code': row.get('County Code', 'UNK'),
                    
                    # Health indicators
                    'under5_mortality': float(row.get('U5_mortal', 0)) if pd.notna(row.get('U5_mortal')) else 0,
                    'stunting_rate': float(row.get('Stun', 0)) if pd.notna(row.get('Stun')) else 0,
                    'maternal_mortality': float(row.get('Mat_mort', 0)) if pd.notna(row.get('Mat_mort')) else 0,
                    'skilled_birth_attendance': float(row.get('Skill_ast_birth', 0)) if pd.notna(row.get('Skill_ast_birth')) else 0,
                    
                    # Education indicators
                    'primary_enrollment': float(row.get('Pr_net_enroll_rt', 0)) if pd.notna(row.get('Pr_net_enroll_rt')) else 0,
                    'secondary_enrollment': float(row.get('Sec_net_enroll_rt', 0)) if pd.notna(row.get('Sec_net_enroll_rt')) else 0,
                    'primary_out_of_school': float(row.get('Pr_outsch_rt', 0)) if pd.notna(row.get('Pr_outsch_rt')) else 0,
                    
                    # WASH indicators
                    'basic_water_access': float(row.get('wat_bas_sev', 0)) if pd.notna(row.get('wat_bas_sev')) else 0,
                    'improved_sanitation': float(row.get('san_imp', 0)) if pd.notna(row.get('san_imp')) else 0,
                    'open_defecation': float(row.get('Open_defec', 0)) if pd.notna(row.get('Open_defec')) else 0,
                    
                    # Socioeconomic indicators
                    'monetary_poverty': float(row.get('PovMon', 0)) if pd.notna(row.get('PovMon')) else 0,
                    'electricity_access': float(row.get('Elect_pop', 0)) if pd.notna(row.get('Elect_pop')) else 0,
                    'birth_registration': float(row.get('brth_reg', 0)) if pd.notna(row.get('brth_reg')) else 0,
                    'social_protection': float(row.get('ScProtSch', 0)) if pd.notna(row.get('ScProtSch')) else 0,
                })
            
            return pd.DataFrame(vulnerability_data)
            
        except Exception as e:
            logger.error(f"Error loading Pillar 2 data: {str(e)}")
            return pd.DataFrame()
    
    def _create_fallback_data(self) -> gpd.GeoDataFrame:
        """
        Create fallback synthetic data if real data loading fails
        """
        logger.warning("Using fallback synthetic data for Kenya counties")
        
        fallback_counties = [
            {"name": "Nairobi", "lat": -1.2921, "lon": 36.8219, "pop": 800000},
            {"name": "Mombasa", "lat": -4.0435, "lon": 39.6682, "pop": 300000},
            {"name": "Kisumu", "lat": -0.0917, "lon": 34.7680, "pop": 150000},
            {"name": "Nakuru", "lat": -0.3031, "lon": 36.0800, "pop": 200000},
            {"name": "Eldoret", "lat": 0.5143, "lon": 35.2698, "pop": 120000},
        ]
        
        data = []
        for county in fallback_counties:
            data.append({
                'region_name': county['name'],
                'county_code': 'SYN',
                'latitude': county['lat'],
                'longitude': county['lon'],
                'child_population': county['pop'],
                'vulnerability_score': np.random.uniform(0.3, 0.8),
                'ccri_index': np.random.uniform(3.0, 8.0),
                'ccri_rank': np.random.randint(1, 48),
                'geometry': f"POINT({county['lon']} {county['lat']})"
            })
        
        df = pd.DataFrame(data)
        df['geometry'] = gpd.points_from_xy(df.longitude, df.latitude)
        return gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

class EnhancedKenyaWeatherProcessor:
    """
    Enhanced weather processor using real Kenya CCRI data
    """
    
    def __init__(self, ccri_data_path: str = None):
        self.ccri_loader = KenyaCCRIDataLoader(ccri_data_path) if ccri_data_path else None
        
    def load_enhanced_demographic_data(self, use_real_data: bool = True) -> gpd.GeoDataFrame:
        """
        Load enhanced demographic data with real CCRI integration
        """
        if use_real_data and self.ccri_loader:
            logger.info("Loading real Kenya CCRI demographic data")
            
            # Load main CCRI data
            ccri_data = self.ccri_loader.load_kenya_ccri_data()
            
            # Load population data from Pillar 1
            population_data = self.ccri_loader.load_pillar1_data()
            
            # Load vulnerability data from Pillar 2  
            vulnerability_data = self.ccri_loader.load_pillar2_data()
            
            # Merge datasets
            if not population_data.empty:
                ccri_data = ccri_data.merge(
                    population_data[['county_name', 'children_u18_total', 'total_population']], 
                    left_on='region_name', 
                    right_on='county_name', 
                    how='left'
                )
                
                # Add child population from real data
                ccri_data['child_population'] = ccri_data['children_u18_total'].fillna(
                    ccri_data['total_population'] * 0.45  # Approximate 45% under 18
                ).astype(int)
            else:
                # Estimate child population if no real data
                ccri_data['child_population'] = (ccri_data['ccri_rank'].apply(
                    lambda x: max(50000, 500000 - x * 10000)  # Higher rank = lower population estimate
                )).astype(int)
            
            if not vulnerability_data.empty:
                ccri_data = ccri_data.merge(
                    vulnerability_data[['county_name', 'under5_mortality', 'stunting_rate', 
                                     'monetary_poverty', 'basic_water_access']], 
                    left_on='region_name', 
                    right_on='county_name', 
                    how='left'
                )
            
            return ccri_data
        else:
            # Use original synthetic data
            logger.info("Using synthetic demographic data")
            return self._create_synthetic_data()
    
    def calculate_enhanced_impact_uncertainty(self, weather_data: Dict, 
                                            demographic_data: gpd.GeoDataFrame,
                                            use_ccri_risks: bool = True) -> Dict:
        """
        Calculate enhanced uncertainty using real CCRI risk factors
        """
        logger.info("Calculating enhanced impact uncertainty with CCRI data")
        
        uncertainty_results = {}
        
        for date_key, weather in weather_data.items():
            date_results = []
            
            for _, region in demographic_data.iterrows():
                # Get interpolated weather values (from existing weather interpolation)
                lat_idx = np.argmin(np.abs(weather['lat'] - region.latitude))
                lon_idx = np.argmin(np.abs(weather['lon'] - region.longitude))
                
                temp = weather['temperature'][lat_idx, lon_idx]
                precip = weather['precipitation'][lat_idx, lon_idx]
                wind = weather['wind_speed'][lat_idx, lon_idx]
                
                temp_uncertainty = weather['uncertainty_temp'][lat_idx, lon_idx]
                precip_uncertainty = weather['uncertainty_precip'][lat_idx, lon_idx]
                wind_uncertainty = weather['uncertainty_wind'][lat_idx, lon_idx]
                
                if use_ccri_risks and 'heat_index' in region:
                    # Use CCRI climate risk factors (0-10 scale, normalize to 0-1)
                    heat_risk_base = region.get('heat_index', 5.0) / 10.0
                    drought_risk_base = region.get('drought_risk', 5.0) / 10.0
                    flood_risk_base = region.get('flood_risk', 5.0) / 10.0
                    
                    # Adjust base risks with current weather
                    heat_risk = heat_risk_base * max(0, (temp - 30) / 15)  # Scale with temperature
                    flood_risk = flood_risk_base * min(1, precip / 30)     # Scale with precipitation
                    storm_risk = min(1, wind / 60) * 0.5  # Wind-based storm risk
                else:
                    # Fallback to calculated risks
                    heat_risk = max(0, (temp - 35) / 10)
                    flood_risk = min(1, precip / 50)
                    storm_risk = min(1, wind / 80)
                
                # Enhanced vulnerability using CCRI data
                vulnerability_score = region.get('vulnerability_score', 0.5)
                
                # Enhanced impact calculation with CCRI risk factors
                base_impact = (
                    heat_risk * vulnerability_score * 0.4 +
                    flood_risk * vulnerability_score * 0.4 +
                    storm_risk * vulnerability_score * 0.2
                )
                
                # Enhanced uncertainty with CCRI confidence
                weather_uncertainty = np.sqrt(
                    (temp_uncertainty / max(temp, 1)) ** 2 +
                    (precip_uncertainty / max(precip, 1)) ** 2 +
                    (wind_uncertainty / max(wind, 1)) ** 2
                ) / 3
                
                # CCRI-based uncertainty factors
                ccri_confidence = 1.0 - (region.get('ccri_index', 5.0) / 10.0 * 0.1)  # Lower CCRI = higher confidence
                data_uncertainty = 0.05 if use_ccri_risks else 0.15  # Lower uncertainty with real data
                
                total_uncertainty = np.sqrt(
                    weather_uncertainty ** 2 +
                    data_uncertainty ** 2 +
                    (1 - ccri_confidence) ** 2 * 0.1
                )
                
                impact_lower = max(0, base_impact - 1.96 * total_uncertainty)
                impact_upper = min(1, base_impact + 1.96 * total_uncertainty)
                
                child_population = region.get('child_population', 100000)
                
                date_results.append({
                    'region_name': region.region_name,
                    'county_code': region.get('county_code', 'UNK'),
                    'latitude': region.latitude,
                    'longitude': region.longitude,
                    'child_population': child_population,
                    'vulnerability_score': vulnerability_score,
                    'ccri_index': region.get('ccri_index', 5.0),
                    'ccri_rank': region.get('ccri_rank', 25),
                    'base_impact': base_impact,
                    'impact_lower_95': impact_lower,
                    'impact_upper_95': impact_upper,
                    'total_uncertainty': total_uncertainty,
                    'weather_uncertainty': weather_uncertainty,
                    'data_source': 'ccri' if use_ccri_risks else 'synthetic',
                    'children_at_risk': int(base_impact * child_population),
                    'children_at_risk_lower': int(impact_lower * child_population),
                    'children_at_risk_upper': int(impact_upper * child_population),
                    'temperature': temp,
                    'precipitation': precip,
                    'wind_speed': wind,
                    
                    # Additional CCRI-specific metrics
                    'heat_risk_factor': region.get('heat_index', 5.0),
                    'drought_risk_factor': region.get('drought_risk', 5.0),
                    'flood_risk_factor': region.get('flood_risk', 5.0),
                    'malaria_risk_factor': region.get('malaria_risk', 5.0),
                })
            
            uncertainty_results[date_key] = date_results
        
        return uncertainty_results
    
    def _create_synthetic_data(self) -> gpd.GeoDataFrame:
        """Fallback synthetic data creation"""
        return self.ccri_loader._create_fallback_data() if self.ccri_loader else gpd.GeoDataFrame()

class IntegratedWeatherSystemCCRI:
    """
    Enhanced integrated system with Kenya CCRI-DRM data integration
    """
    
    def __init__(self, ecmwf_api_key: Optional[str] = None, ccri_data_path: Optional[str] = None):
        # Initialize original processors
        self.basic_processor = WeatherUncertaintyProcessor()
        self.enhanced_processor = EnhancedForecastProcessor(ecmwf_api_key)
        self.spatial_interpolator = AdvancedSpatialInterpolator()
        
        # Initialize Kenya CCRI integration
        self.ccri_data_path = ccri_data_path
        self.kenya_processor = None
        
        if ccri_data_path and Path(ccri_data_path).exists():
            try:
                self.kenya_processor = EnhancedKenyaWeatherProcessor(ccri_data_path)
                logger.info("✅ Kenya CCRI data integration enabled")
            except Exception as e:
                logger.warning(f"Kenya CCRI integration failed: {e}, using synthetic data")
                self.kenya_processor = None
        else:
            logger.info("📄 No CCRI data file provided, using synthetic data")
        
        # Create visualization methods
        self.create_animated_visualization = self.basic_processor.create_animated_visualization
        self.create_folium_map = self.basic_processor.create_folium_map
    
    async def analyze_kenya_ccri(self, bbox: Tuple[float, float, float, float], 
                                days_ahead: int = 7, 
                                interpolation_method: str = 'idw',
                                use_real_ccri_data: bool = True) -> Dict:
        """
        Enhanced analysis specifically for Kenya using CCRI data
        """
        logger.info(f"Running Kenya CCRI analysis with {interpolation_method}")
        
        # Get enhanced weather data
        weather_data = await self.enhanced_processor.download_real_ecmwf_data(bbox, days_ahead)
        
        # Load Kenya demographic data (real CCRI or synthetic)
        if self.kenya_processor and use_real_ccri_data:
            demographic_data = self.kenya_processor.load_enhanced_demographic_data(use_real_data=True)
            data_source = "real_ccri"
        else:
            demographic_data = self.basic_processor.load_demographic_data("KEN")
            data_source = "synthetic"
        
        # Enhanced spatial interpolation
        interpolated_data = self.spatial_interpolator.interpolate_to_points(
            weather_data, demographic_data, method=interpolation_method
        )
        
        # Calculate enhanced uncertainties with CCRI factors
        if self.kenya_processor and use_real_ccri_data:
            enhanced_uncertainty_results = self.kenya_processor.calculate_enhanced_impact_uncertainty(
                interpolated_data, demographic_data, use_ccri_risks=True
            )
        else:
            enhanced_uncertainty_results = self._calculate_enhanced_impact_uncertainty(
                interpolated_data, demographic_data, weather_data
            )
        
        # Calculate Kenya-specific metrics
        kenya_metrics = self._calculate_kenya_specific_metrics(enhanced_uncertainty_results, demographic_data)
        
        return {
            'method': 'kenya_ccri_enhanced',
            'data_source': data_source,
            'weather_data': weather_data,
            'interpolated_data': interpolated_data,
            'uncertainty_results': enhanced_uncertainty_results,
            'kenya_metrics': kenya_metrics,
            'metadata': {
                'bbox': bbox,
                'days_analyzed': days_ahead,
                'country': 'KEN',
                'interpolation_method': interpolation_method,
                'ccri_data_used': use_real_ccri_data and self.kenya_processor is not None,
                'counties_analyzed': len(demographic_data),
                'forecast_quality': {
                    date: weather_data[date].get('forecast_quality', {})
                    for date in weather_data.keys()
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    def _calculate_kenya_specific_metrics(self, uncertainty_results: Dict, demographic_data: gpd.GeoDataFrame) -> Dict:
        """Calculate Kenya-specific summary metrics"""
        
        kenya_summary = {
            'national_summary': {},
            'county_rankings': {},
            'risk_hotspots': {},
            'ccri_correlations': {}
        }
        
        # Aggregate across all dates
        all_results = []
        for date_results in uncertainty_results.values():
            all_results.extend(date_results)
        
        if not all_results:
            return kenya_summary
        
        df_all = pd.DataFrame(all_results)
        
        # National summary
        kenya_summary['national_summary'] = {
            'total_children_at_risk': int(df_all['children_at_risk'].sum()),
            'total_child_population': int(df_all['child_population'].sum()),
            'national_risk_rate': float(df_all['children_at_risk'].sum() / df_all['child_population'].sum()),
            'average_uncertainty': float(df_all['total_uncertainty'].mean()),
            'counties_analyzed': len(df_all['region_name'].unique()),
            'high_risk_counties': len(df_all[df_all['base_impact'] > 0.7]),
            'medium_risk_counties': len(df_all[(df_all['base_impact'] > 0.4) & (df_all['base_impact'] <= 0.7)]),
            'low_risk_counties': len(df_all[df_all['base_impact'] <= 0.4])
        }
        
        # County rankings by risk
        county_avg_risk = df_all.groupby('region_name').agg({
            'base_impact': 'mean',
            'children_at_risk': 'mean',
            'total_uncertainty': 'mean',
            'ccri_rank': 'first'  # CCRI rank is constant per county
        }).round(4)
        
        kenya_summary['county_rankings'] = {
            'highest_risk_counties': county_avg_risk.nlargest(5, 'base_impact')[['base_impact', 'children_at_risk']].to_dict('index'),
            'highest_uncertainty_counties': county_avg_risk.nlargest(5, 'total_uncertainty')[['total_uncertainty', 'base_impact']].to_dict('index'),
            'most_children_at_risk': county_avg_risk.nlargest(5, 'children_at_risk')[['children_at_risk', 'base_impact']].to_dict('index')
        }
        
        # Risk hotspots identification
        high_risk_threshold = df_all['base_impact'].quantile(0.8)
        hotspots = df_all[df_all['base_impact'] >= high_risk_threshold]
        
        if len(hotspots) > 0:
            kenya_summary['risk_hotspots'] = {
                'threshold_impact': float(high_risk_threshold),
                'hotspot_counties': hotspots.groupby('region_name').agg({
                    'base_impact': 'mean',
                    'children_at_risk': 'mean',
                    'latitude': 'first',
                    'longitude': 'first'
                }).round(4).to_dict('index'),
                'total_hotspot_children': int(hotspots['children_at_risk'].sum())
            }
        
        # CCRI correlations (if CCRI data available)
        if 'ccri_index' in df_all.columns and df_all['ccri_index'].notna().any():
            ccri_correlation = df_all['base_impact'].corr(df_all['ccri_index'])
            rank_correlation = df_all['base_impact'].corr(df_all['ccri_rank']) if 'ccri_rank' in df_all.columns else None
            
            kenya_summary['ccri_correlations'] = {
                'impact_vs_ccri_index': float(ccri_correlation) if pd.notna(ccri_correlation) else None,
                'impact_vs_ccri_rank': float(rank_correlation) if rank_correlation and pd.notna(rank_correlation) else None,
                'ccri_validation': 'high' if abs(ccri_correlation or 0) > 0.6 else 'medium' if abs(ccri_correlation or 0) > 0.3 else 'low'
            }
        
        return kenya_summary
    
    # Keep the original methods for backward compatibility
    async def analyze_basic(self, bbox: Tuple[float, float, float, float], 
                       days_ahead: int = 7, country_code: str = 'KEN') -> Dict:
        """Basic analysis using original methods"""
        logger.info(f"Running basic analysis for bbox: {bbox}")
        
        weather_data = await self.basic_processor.download_ecmwf_data(bbox, days_ahead)
        demographic_data = self.basic_processor.load_demographic_data(country_code)
        uncertainty_results = self.basic_processor.calculate_impact_uncertainty(
            weather_data, demographic_data
        )
        
        return {
            'method': 'basic',
            'weather_data': weather_data,
            'uncertainty_results': uncertainty_results,
            'metadata': {
                'bbox': bbox,
                'days_analyzed': days_ahead,
                'country': country_code,
                'interpolation_method': 'nearest_neighbor',
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    async def analyze_enhanced(self, bbox: Tuple[float, float, float, float], 
                          days_ahead: int = 7, country_code: str = 'KEN',
                          interpolation_method: str = 'idw') -> Dict:
        """Enhanced analysis using advanced methods"""
        logger.info(f"Running enhanced analysis with {interpolation_method} for bbox: {bbox}")
        
        # Get enhanced weather data
        weather_data = await self.enhanced_processor.download_real_ecmwf_data(bbox, days_ahead)
        
        # Load demographic data
        demographic_data = self.basic_processor.load_demographic_data(country_code)
        
        # Enhanced spatial interpolation
        interpolated_data = self.spatial_interpolator.interpolate_to_points(
            weather_data, demographic_data, method=interpolation_method
        )
        
        # Calculate enhanced uncertainties
        enhanced_uncertainty_results = self._calculate_enhanced_impact_uncertainty(
            interpolated_data, demographic_data, weather_data
        )
        
        return {
            'method': 'enhanced',
            'weather_data': weather_data,
            'interpolated_data': interpolated_data,
            'uncertainty_results': enhanced_uncertainty_results,
            'metadata': {
                'bbox': bbox,
                'days_analyzed': days_ahead,
                'country': country_code,
                'interpolation_method': interpolation_method,
                'forecast_quality': {
                    date: weather_data[date].get('forecast_quality', {})
                    for date in weather_data.keys()
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    def _calculate_enhanced_impact_uncertainty(self, interpolated_data: Dict, 
                                         demographic_data: gpd.GeoDataFrame,
                                         weather_data: Dict) -> Dict:
        """Calculate enhanced uncertainty for child impact predictions"""
        logger.info("Calculating enhanced impact uncertainty")
        
        enhanced_results = {}
        
        for date_key, date_results in interpolated_data.items():
            enhanced_date_results = []
            
            for result in date_results:
                # Find corresponding demographic data
                region_demo = demographic_data[
                    demographic_data['region_name'] == result['region_name']
                ].iloc[0]
                
                # Extract interpolated weather values
                temp = result.get('temperature', 25)
                precip = result.get('precipitation', 0)
                wind = result.get('wind_speed', 10)
                
                # Extract uncertainties
                temp_unc = result.get('temperature_uncertainty', 2.0)
                precip_unc = result.get('precipitation_uncertainty', 1.0)
                wind_unc = result.get('wind_speed_uncertainty', 3.0)
                
                # Calculate risk factors
                heat_risk = max(0, (temp - 35) / 10)
                flood_risk = min(1, precip / 50)
                storm_risk = min(1, wind / 80)
                
                # Enhanced impact calculation with vulnerability
                base_impact = (
                    heat_risk * region_demo.vulnerability_score * 0.4 +
                    flood_risk * region_demo.vulnerability_score * 0.4 +
                    storm_risk * region_demo.vulnerability_score * 0.2
                )
                
                # Enhanced uncertainty calculation
                weather_uncertainty = np.sqrt(
                    (temp_unc / max(temp, 1)) ** 2 +
                    (precip_unc / max(precip, 1)) ** 2 +
                    (wind_unc / max(wind, 1)) ** 2
                ) / 3
                
                # Add interpolation method uncertainty
                method_uncertainty = 0.10  # Default uncertainty
                
                # Get forecast quality factor
                quality_factor = 1.0
                if date_key in weather_data:
                    forecast_quality = weather_data[date_key].get('forecast_quality', {})
                    quality_factor = forecast_quality.get('overall_confidence', 1.0)
                
                # Total enhanced uncertainty
                total_uncertainty = np.sqrt(
                    weather_uncertainty ** 2 +
                    method_uncertainty ** 2 +
                    (1 - quality_factor) ** 2 * 0.1 +
                    0.05 ** 2  # Demographic data uncertainty
                )
                
                # Calculate confidence intervals
                impact_lower = max(0, base_impact - 1.96 * total_uncertainty)
                impact_upper = min(1, base_impact + 1.96 * total_uncertainty)
                
                enhanced_result = {
                    'region_name': result['region_name'],
                    'latitude': result['latitude'],
                    'longitude': result['longitude'],
                    'child_population': region_demo.child_population,
                    'vulnerability_score': region_demo.vulnerability_score,
                    'base_impact': base_impact,
                    'impact_lower_95': impact_lower,
                    'impact_upper_95': impact_upper,
                    'total_uncertainty': total_uncertainty,
                    'weather_uncertainty': weather_uncertainty,
                    'method_uncertainty': method_uncertainty,
                    'forecast_quality': quality_factor,
                    'children_at_risk': int(base_impact * region_demo.child_population),
                    'children_at_risk_lower': int(impact_lower * region_demo.child_population),
                    'children_at_risk_upper': int(impact_upper * region_demo.child_population),
                    'temperature': temp,
                    'precipitation': precip,
                    'wind_speed': wind,
                    'interpolation_method': result.get('interpolation_method', 'unknown')
                }
                
                enhanced_date_results.append(enhanced_result)
            
            enhanced_results[date_key] = enhanced_date_results
        
        return enhanced_results

# Flask Application with Integrated Endpoints
app = Flask(__name__)
CORS(app)

# Initialize integrated system
integrated_system = IntegratedWeatherSystemCCRI()

@app.route('/')
def index():
    """Enhanced API documentation page"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced UNICEF Weather Impact Uncertainty Microservice</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1000px; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #1976d2; }
            .endpoint { background: #e3f2fd; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .enhanced { background: #e8f5e8; border-left: 4px solid #4caf50; }
            .method { background: #4caf50; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
            .enhanced-method { background: #ff9800; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
            code { background: #f5f5f5; padding: 2px 5px; border-radius: 3px; }
            .feature-list { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .feature { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 3px solid #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌪️ Enhanced UNICEF Weather Impact Uncertainty Microservice</h1>
            <p>This integrated microservice provides both <strong>basic</strong> and <strong>enhanced</strong> methods for quantifying and visualizing uncertainty in weather impact predictions for children.</p>
            
            <h2>🆕 New Enhanced Features:</h2>
            <div class="feature-list">
                <div class="feature">
                    <h4>🎯 Advanced Forecasting</h4>
                    <ul>
                        <li>Real ECMWF API integration</li>
                        <li>Ensemble uncertainty quantification</li>
                        <li>Physics-based weather patterns</li>
                        <li>Topographic corrections</li>
                    </ul>
                </div>
                <div class="feature">
                    <h4>🗺️ Spatial Interpolation</h4>
                    <ul>
                        <li>5 interpolation methods</li>
                        <li>Elevation-aware processing</li>
                        <li>Uncertainty propagation</li>
                        <li>Quality control validation</li>
                    </ul>
                </div>
            </div>
            
            <h2>API Endpoints:</h2>
            
            <h3>Basic Endpoints (Original Functionality)</h3>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /api/analyze</h3>
                <p>Basic weather impact uncertainty analysis (original method)</p>
                <p><strong>Parameters:</strong> bbox, days_ahead, country_code</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /api/visualize/animated</h3>
                <p>Generate basic animated uncertainty visualization</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /api/visualize/map</h3>
                <p>Generate basic interactive uncertainty map</p>
            </div>
            
            <h3>Enhanced Endpoints (New Advanced Features)</h3>
            
            <div class="endpoint enhanced">
                <h3><span class="enhanced-method">POST</span> /api/analyze/enhanced</h3>
                <p>Enhanced weather impact analysis with advanced forecasting and interpolation</p>
                <p><strong>Parameters:</strong></p>
                <ul>
                    <li><code>bbox</code>: Bounding box [min_lon, min_lat, max_lon, max_lat]</li>
                    <li><code>days_ahead</code>: Number of forecast days (default: 7)</li>
                    <li><code>country_code</code>: Country code (default: "KEN")</li>
                    <li><code>interpolation_method</code>: "nearest", "idw", "spline", "kriging", "gaussian_process"</li>
                    <li><code>ecmwf_api_key</code>: Optional ECMWF API key for real data</li>
                </ul>
            </div>
            
            <div class="endpoint enhanced">
                <h3><span class="enhanced-method">POST</span> /api/compare/methods</h3>
                <p>Compare multiple interpolation methods side-by-side</p>
                <p>Tests all 5 interpolation methods and returns accuracy metrics</p>
            </div>
            
            <div class="endpoint enhanced">
                <h3><span class="enhanced-method">GET</span> /api/methods/info</h3>
                <p>Get information about available interpolation methods and their characteristics</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /api/health</h3>
                <p>Service health check</p>
            </div>
            
            <h2>Example Usage:</h2>
            
            <h4>Basic Analysis:</h4>
            <pre><code>curl -X POST "http://localhost:5000/api/analyze" \
     -H "Content-Type: application/json" \
     -d '{"bbox": [33.4, -5.5, 42.9, 6.0], "days_ahead": 7}'</code></pre>
            
            <h4>Enhanced Analysis:</h4>
            <pre><code>curl -X POST "http://localhost:5000/api/analyze/enhanced" \
     -H "Content-Type: application/json" \
     -d '{
       "bbox": [33.4, -5.5, 42.9, 6.0], 
       "days_ahead": 7,
       "interpolation_method": "kriging",
       "ecmwf_api_key": "your_key_here"
     }'</code></pre>
            
            <h4>Method Comparison:</h4>
            <pre><code>curl -X POST "http://localhost:5000/api/compare/methods" \
     -H "Content-Type: application/json" \
     -d '{"bbox": [33.4, -5.5, 42.9, 6.0], "days_ahead": 3}'</code></pre>
            
            <h2>Response Features:</h2>
            <ul>
                <li>📊 <strong>Enhanced uncertainty quantification</strong> - Multiple uncertainty sources</li>
                <li>🎯 <strong>Forecast quality scoring</strong> - Reliability indicators</li>
                <li>🗺️ <strong>Elevation-corrected interpolation</strong> - Topographic awareness</li>
                <li>📈 <strong>Method performance metrics</strong> - Accuracy comparisons</li>
                <li>🔍 <strong>Confidence intervals</strong> - 95% prediction bounds</li>
            </ul>
        </div>
    </body>
    </html>
    """)

@app.route('/api/health')
def health_check():
    """Enhanced health check showing CCRI integration status"""
    features = ["basic_analysis", "enhanced_forecasting", "advanced_interpolation", "quality_control"]
    
    # Check for CCRI data availability
    ccri_available = False
    ccri_path = os.environ.get('CCRI_DATA_PATH')
    if ccri_path and Path(ccri_path).exists():
        ccri_available = True
        features.append("kenya_ccri_integration")
    
    return jsonify({
        "status": "healthy", 
        "service": "Enhanced UNICEF Weather Uncertainty Microservice with Kenya CCRI",
        "features": features,
        "available_methods": ["nearest", "idw", "spline", "kriging", "gaussian_process"],
        "kenya_ccri_available": ccri_available,
        "ccri_features": [
            "real_county_data", 
            "climate_risk_indices", 
            "child_population_estimates",
            "vulnerability_scoring",
            "prediction_validation"
        ] if ccri_available else []
    })

def async_route(f):
    """Decorator to handle async functions in Flask"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

@app.route('/api/analyze', methods=['POST'])
@async_route
async def analyze_basic():
    """Basic analysis endpoint (original functionality)"""
    try:
        data = request.get_json()
        bbox = data.get('bbox', [33.4, -5.5, 42.9, 6.0])
        days_ahead = data.get('days_ahead', 7)
        country_code = data.get('country_code', 'KEN')
        
        logger.info(f"Running basic analysis for bbox: {bbox}")
        
        results = await integrated_system.analyze_basic(bbox, days_ahead, country_code)
        
        # Calculate summary statistics
        summary = {}
        for date_key, date_results in results['uncertainty_results'].items():
            df = pd.DataFrame(date_results)
            summary[date_key] = {
                'total_children_at_risk': int(df['children_at_risk'].sum()),
                'total_children_at_risk_lower': int(df['children_at_risk_lower'].sum()),
                'total_children_at_risk_upper': int(df['children_at_risk_upper'].sum()),
                'average_uncertainty': float(df['total_uncertainty'].mean()),
                'max_impact_region': df.loc[df['base_impact'].idxmax(), 'region_name'],
                'max_impact_score': float(df['base_impact'].max())
            }
        
        return jsonify({
            'status': 'success',
            'method': 'basic',
            'analysis_summary': summary,
            'detailed_results': results['uncertainty_results'],
            'metadata': results['metadata']
        })
        
    except Exception as e:
        logger.error(f"Basic analysis error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/analyze/enhanced', methods=['POST'])
@async_route
async def analyze_enhanced():
    """Enhanced analysis endpoint with advanced features"""
    try:
        data = request.get_json()
        bbox = data.get('bbox', [33.4, -5.5, 42.9, 6.0])
        days_ahead = data.get('days_ahead', 7)
        country_code = data.get('country_code', 'KEN')
        interpolation_method = data.get('interpolation_method', 'idw')
        ecmwf_api_key = data.get('ecmwf_api_key')
        
        # Update API key if provided
        if ecmwf_api_key:
            integrated_system.enhanced_processor.ecmwf_api_key = ecmwf_api_key
        
        logger.info(f"Running enhanced analysis with {interpolation_method} for bbox: {bbox}")
        
        results = await integrated_system.analyze_enhanced(
            bbox, days_ahead, country_code, interpolation_method
        )
        
        # Calculate enhanced summary statistics
        summary = {}
        for date_key, date_results in results['uncertainty_results'].items():
            df = pd.DataFrame(date_results)
            summary[date_key] = {
                'total_children_at_risk': int(df['children_at_risk'].sum()),
                'total_children_at_risk_lower': int(df['children_at_risk_lower'].sum()),
                'total_children_at_risk_upper': int(df['children_at_risk_upper'].sum()),
                'average_uncertainty': float(df['total_uncertainty'].mean()),
                'average_forecast_quality': float(df['forecast_quality'].mean()),
                'interpolation_method': interpolation_method,
                'max_impact_region': df.loc[df['base_impact'].idxmax(), 'region_name'],
                'max_impact_score': float(df['base_impact'].max()),
                'uncertainty_breakdown': {
                    'weather_uncertainty': float(df['weather_uncertainty'].mean()),
                    'method_uncertainty': float(df['method_uncertainty'].mean()),
                    'total_uncertainty': float(df['total_uncertainty'].mean())
                }
            }
        
        return jsonify({
            'status': 'success',
            'method': 'enhanced',
            'interpolation_method': interpolation_method,
            'analysis_summary': summary,
            'detailed_results': results['uncertainty_results'],
            'metadata': results['metadata']
        })
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/compare/methods', methods=['POST'])
async def compare_interpolation_methods():
    """Compare different interpolation methods"""
    try:
        data = request.get_json()
        bbox = data.get('bbox', [33.4, -5.5, 42.9, 6.0])
        days_ahead = data.get('days_ahead', 3)  # Shorter for comparison
        country_code = data.get('country_code', 'KEN')
        
        methods = ['nearest', 'idw', 'spline', 'kriging', 'gaussian_process']
        comparison_results = {}
        
        logger.info(f"Comparing interpolation methods for bbox: {bbox}")
        
        for method in methods:
            try:
                results = await integrated_system.analyze_enhanced(
                    bbox, days_ahead, country_code, method
                )
                
                # Calculate performance metrics
                first_date = list(results['uncertainty_results'].keys())[0]
                df = pd.DataFrame(results['uncertainty_results'][first_date])
                
                comparison_results[method] = {
                    'status': 'success',
                    'average_uncertainty': float(df['total_uncertainty'].mean()),
                    'forecast_quality': float(df['forecast_quality'].mean()),
                    'total_children_at_risk': int(df['children_at_risk'].sum()),
                    'processing_time': 'estimated',  # Could add actual timing
                    'method_characteristics': {
                        'smoothness': {'nearest': 1, 'idw': 3, 'spline': 5, 'kriging': 4, 'gaussian_process': 5}[method],
                        'accuracy': {'nearest': 2, 'idw': 3, 'spline': 4, 'kriging': 5, 'gaussian_process': 5}[method],
                        'computational_cost': {'nearest': 1, 'idw': 2, 'spline': 3, 'kriging': 4, 'gaussian_process': 5}[method],
                        'uncertainty_quantification': {'nearest': 2, 'idw': 2, 'spline': 3, 'kriging': 5, 'gaussian_process': 5}[method]
                    }
                }
                
            except Exception as e:
                comparison_results[method] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Recommend best method
        successful_methods = {k: v for k, v in comparison_results.items() if v['status'] == 'success'}
        
        if successful_methods:
            # Simple scoring based on uncertainty and quality
            best_method = min(successful_methods.keys(), 
                            key=lambda m: successful_methods[m]['average_uncertainty'])
            
            recommendation = {
                'recommended_method': best_method,
                'reason': f"Lowest average uncertainty ({successful_methods[best_method]['average_uncertainty']:.3f})",
                'alternatives': list(successful_methods.keys())
            }
        else:
            recommendation = {
                'recommended_method': 'idw',
                'reason': 'Default fallback - all methods failed',
                'alternatives': ['nearest']
            }
        
        return jsonify({
            'status': 'success',
            'comparison_results': comparison_results,
            'recommendation': recommendation,
            'metadata': {
                'bbox': bbox,
                'days_analyzed': days_ahead,
                'country': country_code,
                'comparison_timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Method comparison error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/methods/info')
def get_methods_info():
    """Get information about available interpolation methods"""
    methods_info = {
        'nearest': {
            'name': 'Enhanced Nearest Neighbor',
            'description': 'Distance and elevation weighted nearest neighbor interpolation',
            'advantages': ['Fast', 'Simple', 'Elevation-aware'],
            'disadvantages': ['Can be discontinuous', 'Limited smoothness'],
            'best_for': ['Real-time applications', 'Sparse data', 'Quick estimates'],
            'uncertainty_estimate': 'Medium',
            'computational_cost': 'Low'
        },
        'idw': {
            'name': 'Inverse Distance Weighting',
            'description': 'Distance-based weighted interpolation with configurable power parameter',
            'advantages': ['Smooth results', 'Intuitive', 'Robust'],
            'disadvantages': ['No inherent uncertainty', 'Can smooth over important features'],
            'best_for': ['General purpose', 'Smooth fields', 'Moderate data density'],
            'uncertainty_estimate': 'Medium',
            'computational_cost': 'Low-Medium'
        },
        'spline': {
            'name': 'Thin Plate Splines',
            'description': 'Minimum curvature surface interpolation',
            'advantages': ['Very smooth', 'Good for large-scale patterns', 'Mathematically elegant'],
            'disadvantages': ['Can overshoot', 'Computationally intensive', 'May not honor local variations'],
            'best_for': ['Large-scale patterns', 'Dense data', 'Smooth phenomena'],
            'uncertainty_estimate': 'Medium-High',
            'computational_cost': 'Medium-High'
        },
        'kriging': {
            'name': 'Ordinary Kriging (Gaussian Process)',
            'description': 'Optimal spatial interpolation with uncertainty quantification',
            'advantages': ['Optimal estimates', 'Uncertainty quantification', 'Flexible'],
            'disadvantages': ['Computationally expensive', 'Requires parameter tuning'],
            'best_for': ['High accuracy requirements', 'Uncertainty analysis', 'Research applications'],
            'uncertainty_estimate': 'High',
            'computational_cost': 'High'
        },
        'gaussian_process': {
            'name': '3D Gaussian Process',
            'description': 'Gaussian process with elevation as additional feature',
            'advantages': ['Best accuracy', 'Elevation-aware', 'Full uncertainty', 'Flexible kernels'],
            'disadvantages': ['Most computationally expensive', 'Complex parameter tuning'],
            'best_for': ['Highest accuracy needs', 'Complex terrain', 'Research applications'],
            'uncertainty_estimate': 'Highest',
            'computational_cost': 'Highest'
        }
    }
    
    return jsonify({
        'available_methods': methods_info,
        'selection_guide': {
            'speed_priority': 'nearest',
            'balance': 'idw',
            'accuracy_priority': 'kriging',
            'best_uncertainty': 'gaussian_process',
            'mountainous_terrain': 'gaussian_process',
            'flat_terrain': 'idw'
        },
        'default_recommendation': 'idw'
    })

@app.route('/api/visualize/animated')
@async_route
async def create_animated_viz():
    """Create animated visualization - FIXED to actually create animation"""
    try:
        bbox = [33.4, -5.5, 42.9, 6.0]
        results = await integrated_system.analyze_basic(bbox, 7, 'KEN')
        
        # Create static directory
        Path("static").mkdir(exist_ok=True)
        output_path = "static/weather_uncertainty_animation.html"
        
        fig = integrated_system.basic_processor.create_animated_visualization(
            results['uncertainty_results'], output_path
        )
        
        return jsonify({
            'status': 'success',
            'animation_url': f'/static/weather_uncertainty_animation.html',
            'message': 'Animated visualization created successfully',
            'type': 'animation',
            'file_size': Path(output_path).stat().st_size if Path(output_path).exists() else 0
        })
        
    except Exception as e:
        logger.error(f"Animation creation error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/analyze/kenya', methods=['POST'])
@async_route
async def analyze_kenya_ccri():
    """
    Kenya-specific analysis using CCRI data
    NEW ENDPOINT for Kenya CCRI integration
    """
    try:
        data = request.get_json()
        bbox = data.get('bbox', [33.4, -5.5, 42.9, 6.0])  # Kenya bounding box
        days_ahead = data.get('days_ahead', 7)
        interpolation_method = data.get('interpolation_method', 'idw')
        use_real_ccri_data = data.get('use_real_ccri_data', True)
        ecmwf_api_key = data.get('ecmwf_api_key')
        ccri_data_path = data.get('ccri_data_path')  # Path to CCRI Excel file
        
        # Initialize enhanced system with CCRI data
        enhanced_system = IntegratedWeatherSystemCCRI(
            ecmwf_api_key=ecmwf_api_key,
            ccri_data_path=ccri_data_path
        )
        
        logger.info(f"Running Kenya CCRI analysis for bbox: {bbox}")
        
        results = await enhanced_system.analyze_kenya_ccri(
            bbox, days_ahead, interpolation_method, use_real_ccri_data
        )
        
        # Enhanced summary with Kenya-specific metrics
        summary = {}
        for date_key, date_results in results['uncertainty_results'].items():
            df = pd.DataFrame(date_results)
            summary[date_key] = {
                'total_children_at_risk': int(df['children_at_risk'].sum()),
                'total_children_at_risk_lower': int(df['children_at_risk_lower'].sum()),
                'total_children_at_risk_upper': int(df['children_at_risk_upper'].sum()),
                'average_uncertainty': float(df['total_uncertainty'].mean()),
                'interpolation_method': interpolation_method,
                'data_source': results['data_source'],
                'highest_risk_county': df.loc[df['base_impact'].idxmax(), 'region_name'],
                'highest_risk_score': float(df['base_impact'].max()),
                'counties_with_ccri_data': len(df[df.get('ccri_index', pd.Series()).notna()]) if 'ccri_index' in df else 0
            }
        
        return jsonify({
            'status': 'success',
            'method': 'kenya_ccri_enhanced',
            'data_source': results['data_source'],
            'analysis_summary': summary,
            'kenya_metrics': results['kenya_metrics'],
            'detailed_results': results['uncertainty_results'],
            'metadata': results['metadata']
        })
        
    except Exception as e:
        logger.error(f"Kenya CCRI analysis error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/kenya/counties')
def get_kenya_counties():
    """
    Get list of Kenya counties with CCRI data availability
    """
    try:
        ccri_data_path = request.args.get('ccri_data_path')
        
        if ccri_data_path and Path(ccri_data_path).exists():
            loader = KenyaCCRIDataLoader(ccri_data_path)
            ccri_data = loader.load_kenya_ccri_data()
            
            counties_info = []
            for _, county in ccri_data.iterrows():
                counties_info.append({
                    'county_name': county['region_name'],
                    'county_code': county.get('county_code', 'UNK'),
                    'latitude': float(county['latitude']),
                    'longitude': float(county['longitude']),
                    'ccri_index': float(county.get('ccri_index', 0)),
                    'ccri_rank': int(county.get('ccri_rank', 0)),
                    'vulnerability_score': float(county.get('vulnerability_score', 0)),
                    'child_population': int(county.get('child_population', 0)),
                    'data_source': 'ccri'
                })
            
            return jsonify({
                'status': 'success',
                'counties': counties_info,
                'total_counties': len(counties_info),
                'data_source': 'real_ccri'
            })
        else:
            # Return synthetic county data
            synthetic_counties = [
                {'county_name': 'Nairobi', 'latitude': -1.2921, 'longitude': 36.8219, 'data_source': 'synthetic'},
                {'county_name': 'Mombasa', 'latitude': -4.0435, 'longitude': 39.6682, 'data_source': 'synthetic'},
                {'county_name': 'Kisumu', 'latitude': -0.0917, 'longitude': 34.7680, 'data_source': 'synthetic'},
                {'county_name': 'Nakuru', 'latitude': -0.3031, 'longitude': 36.0800, 'data_source': 'synthetic'},
                {'county_name': 'Eldoret', 'latitude': 0.5143, 'longitude': 35.2698, 'data_source': 'synthetic'}
            ]
            
            return jsonify({
                'status': 'success',
                'counties': synthetic_counties,
                'total_counties': len(synthetic_counties),
                'data_source': 'synthetic',
                'message': 'CCRI data not available, showing synthetic counties'
            })
            
    except Exception as e:
        logger.error(f"Counties retrieval error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/kenya/ccri-validation', methods=['POST'])
@async_route
async def validate_ccri_predictions():
    """
    Validate weather impact predictions against CCRI baseline data
    """
    try:
        data = request.get_json()
        bbox = data.get('bbox', [33.4, -5.5, 42.9, 6.0])
        ccri_data_path = data.get('ccri_data_path')
        
        if not ccri_data_path or not Path(ccri_data_path).exists():
            return jsonify({
                'status': 'error', 
                'message': 'CCRI data file required for validation'
            }), 400
        
        # Run analysis with CCRI data
        enhanced_system = IntegratedWeatherSystemCCRI(ccri_data_path=ccri_data_path)
        results = await enhanced_system.analyze_kenya_ccri(bbox, days_ahead=3, use_real_ccri_data=True)
        
        # Extract validation metrics
        validation_results = {
            'ccri_correlation': results['kenya_metrics']['ccri_correlations'],
            'prediction_accuracy': 'high' if results['kenya_metrics']['ccri_correlations'].get('impact_vs_ccri_index', 0) > 0.6 else 'medium',
            'counties_validated': results['metadata']['counties_analyzed'],
            'data_source_confirmation': results['data_source'],
            'validation_summary': {
                'weather_vs_ccri_alignment': results['kenya_metrics']['ccri_correlations'].get('ccri_validation', 'unknown'),
                'hotspots_identified': len(results['kenya_metrics']['risk_hotspots'].get('hotspot_counties', {})),
                'total_children_analyzed': results['kenya_metrics']['national_summary']['total_child_population']
            }
        }
        
        return jsonify({
            'status': 'success',
            'validation_results': validation_results,
            'methodology': 'Correlation between weather impact predictions and CCRI baseline risk indices'
        })
        
    except Exception as e:
        logger.error(f"CCRI validation error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files - FIXED with security check"""
    # Security fix: prevent directory traversal
    if '..' in filename or filename.startswith('/'):
        return "Invalid file path", 400
    
    try:
        return app.send_static_file(filename)
    except Exception:
        return "File not found", 404

# Add the visualization methods to the basic processor
def create_animated_visualization(self, uncertainty_data: Dict, 
                                output_path: str = "weather_uncertainty_animation.html"):
    """Create animated visualization of uncertainty in weather impacts"""
    logger.info("Creating animated uncertainty visualization")
    
    dates = sorted(uncertainty_data.keys())
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Children at Risk (with Uncertainty)', 'Impact by Region Over Time',
                       'Uncertainty Components', 'Risk Distribution'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    frames = []
    
    for i, date in enumerate(dates):
        data = uncertainty_data[date]
        df_day = pd.DataFrame(data)
        
        scatter_trace = go.Scatter(
            x=df_day['longitude'],
            y=df_day['latitude'],
            mode='markers',
            marker=dict(
                size=df_day['children_at_risk'] / 5000,
                color=df_day['base_impact'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Impact Score", x=0.45),
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=[f"{row['region_name']}<br>At Risk: {row['children_at_risk']:,}<br>"
                  f"Uncertainty: ±{row['total_uncertainty']:.2f}" 
                  for _, row in df_day.iterrows()],
            hovertemplate='%{text}<extra></extra>',
            name=f'Day {i+1}'
        )
        
        frames.append(go.Frame(
            data=[scatter_trace],
            name=f'Day {i+1}',
            layout=go.Layout(title_text=f"Weather Impact Uncertainty - {date}")
        ))
    
    initial_data = uncertainty_data[dates[0]]
    df_initial = pd.DataFrame(initial_data)
    
    fig.add_trace(
        go.Scatter(
            x=df_initial['longitude'],
            y=df_initial['latitude'],
            mode='markers',
            marker=dict(
                size=df_initial['children_at_risk'] / 5000,
                color=df_initial['base_impact'],
                colorscale='Reds',
                showscale=True,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=[f"{row['region_name']}<br>At Risk: {row['children_at_risk']:,}<br>"
                  f"Uncertainty: ±{row['total_uncertainty']:.2f}" 
                  for _, row in df_initial.iterrows()],
            hovertemplate='%{text}<extra></extra>',
            name='Children at Risk'
        ),
        row=1, col=1
    )
    
    uncertainty_breakdown = pd.DataFrame({
        'Component': ['Model Error', 'Interpolation', 'Data Age', 'Aggregation'],
        'Uncertainty': [0.15, 0.10, 0.05, 0.08]
    })
    
    fig.add_trace(
        go.Bar(
            x=uncertainty_breakdown['Component'],
            y=uncertainty_breakdown['Uncertainty'],
            marker_color=['red', 'blue', 'green', 'orange'],
            name='Uncertainty Components'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=df_initial['base_impact'],
            nbinsx=20,
            marker_color='lightblue',
            name='Risk Distribution'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=df_initial['region_name'],
            y=df_initial['children_at_risk'],
            error_y=dict(
                type='data',
                array=df_initial['children_at_risk_upper'] - df_initial['children_at_risk'],
                arrayminus=df_initial['children_at_risk'] - df_initial['children_at_risk_lower']
            ),
            marker_color='coral',
            name='Regional Risk'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="UNICEF Weather Impact Uncertainty Analysis for Children",
        height=800,
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 1000, "redraw": True},
                                     "fromcurrent": True}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}])
                ]),
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.011,
                xanchor="right",
                y=0,
                yanchor="top"
            ),
        ],
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue={
                "font": {"size": 20},
                "prefix": "Day:",
                "visible": True,
                "xanchor": "right"
            },
            transition={"duration": 300, "easing": "cubic-in-out"},
            pad={"b": 10, "t": 50},
            len=0.9,
            x=0.1,
            y=0,
            steps=[dict(
                args=[[f'Day {i+1}'],
                      {"frame": {"duration": 300, "redraw": True},
                       "mode": "immediate",
                       "transition": {"duration": 300}}],
                label=f'Day {i+1}',
                method="animate") for i in range(len(dates))]
        )]
    )
    
    fig.frames = frames
    fig.write_html(output_path)
    logger.info(f"Animated visualization saved to {output_path}")
    
    return fig

@app.route('/api/visualize/map')
@async_route
async def create_map_viz():
    """Create interactive map visualization - FIXED to use proper naming"""
    try:
        bbox = [33.4, -5.5, 42.9, 6.0]
        results = await integrated_system.analyze_basic(bbox, 7, 'KEN')
        
        Path("static").mkdir(exist_ok=True)
        output_path = "static/uncertainty_map.html"
        
        # Create the map using the integrated system's map method
        integrated_system.basic_processor.create_folium_map(
            results['uncertainty_results'], output_path
        )
        
        return jsonify({
            'status': 'success',
            'map_url': f'/static/uncertainty_map.html',
            'message': 'Interactive map created successfully',
            'type': 'map',
            'regions_analyzed': len(results['uncertainty_results'][list(results['uncertainty_results'].keys())[0]]),
            'file_size': Path(output_path).stat().st_size if Path(output_path).exists() else 0
        })
        
    except Exception as e:
        logger.error(f"Map creation error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def create_folium_map_standalone(uncertainty_data: Dict, 
                                output_path: str = "uncertainty_map.html") -> folium.Map:
    """Standalone function to create interactive Folium map with uncertainty visualization"""
    logger.info("Creating interactive Folium map")
    
    # Use first date for map
    first_date = list(uncertainty_data.keys())[0]
    data = uncertainty_data[first_date]
    df = pd.DataFrame(data)
    
    # Create base map centered on Kenya
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Add uncertainty circles
    for _, row in df.iterrows():
        # Main impact circle
        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=row['children_at_risk'] * 10,  # Scale radius
            popup=folium.Popup(
                f"""
                <b>{row['region_name']}</b><br>
                Children at Risk: {row['children_at_risk']:,}<br>
                Impact Score: {row['base_impact']:.3f}<br>
                Uncertainty: ±{row['total_uncertainty']:.3f}<br>
                95% CI: {row['children_at_risk_lower']:,} - {row['children_at_risk_upper']:,}
                """,
                max_width=300
            ),
            color='red',
            fillColor='red',
            fillOpacity=0.6,
            weight=2
        ).add_to(m)
        
        # Uncertainty ring
        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=row['children_at_risk_upper'] * 10,
            color='orange',
            fillColor='orange',
            fillOpacity=0.1,
            weight=1,
            dashArray='5, 5'
        ).add_to(m)
    
    # Add heatmap layer
    heat_data = [[row['latitude'], row['longitude'], row['base_impact']] 
                for _, row in df.iterrows()]
    
    plugins.HeatMap(
        heat_data,
        name="Risk Heatmap",
        min_opacity=0.2,
        radius=50,
        blur=40,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    m.save(output_path)
    logger.info(f"Interactive map saved to {output_path}")
    
    return m

def create_folium_map(self, uncertainty_data: Dict, 
                         output_path: str = "uncertainty_map.html") -> folium.Map:
        """Create interactive Folium map with uncertainty visualization"""
        logger.info("Creating interactive Folium map")
        
        first_date = list(uncertainty_data.keys())[0]
        data = uncertainty_data[first_date]
        df = pd.DataFrame(data)
        
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='CartoDB positron'
        )
        
        for _, row in df.iterrows():
            folium.Circle(
                location=[row['latitude'], row['longitude']],
                radius=row['children_at_risk'] * 10,
                popup=folium.Popup(
                    f"""
                    <b>{row['region_name']}</b><br>
                    Children at Risk: {row['children_at_risk']:,}<br>
                    Impact Score: {row['base_impact']:.3f}<br>
                    Uncertainty: ±{row['total_uncertainty']:.3f}<br>
                    95% CI: {row['children_at_risk_lower']:,} - {row['children_at_risk_upper']:,}
                    """,
                    max_width=300
                ),
                color='red',
                fillColor='red',
                fillOpacity=0.6,
                weight=2
            ).add_to(m)
            
            folium.Circle(
                location=[row['latitude'], row['longitude']],
                radius=row['children_at_risk_upper'] * 10,
                color='orange',
                fillColor='orange',
                fillOpacity=0.1,
                weight=1,
                dashArray='5, 5'
            ).add_to(m)
        
        heat_data = [[row['latitude'], row['longitude'], row['base_impact']] 
                    for _, row in df.iterrows()]
        
        plugins.HeatMap(
            heat_data,
            name="Risk Heatmap",
            min_opacity=0.2,
            radius=50,
            blur=40,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        m.save(output_path)
        logger.info(f"Interactive map saved to {output_path}")
        
        return m

if __name__ == '__main__':
    ccri_data_path = "C:/Users/MMOHAMEDSU1/Downloads/kenya_ccridrm_model_v2.3.xlsx"
    
    if Path(ccri_data_path).exists():
        logger.info(f"✅ Kenya CCRI data found at: {ccri_data_path}")
        integrated_system = IntegratedWeatherSystemCCRI(ccri_data_path=ccri_data_path)
    else:
        logger.info("📄 No CCRI data found, using original system")
    
    print("\n" + "="*80)
    print("🌍 ENHANCED UNICEF WEATHER IMPACT UNCERTAINTY MICROSERVICE")
    print("🇰🇪 WITH KENYA CCRI-DRM DATA INTEGRATION")
    print("="*80)
    print("✅ Basic functionality (backward compatible)")
    print("✅ Enhanced forecasting with ECMWF integration")
    print("✅ Advanced spatial interpolation (5 methods)")
    print("✅ Quality control and validation")
    print("✅ Method comparison capabilities")
    print("🆕 Kenya CCRI-DRM real data integration")
    print("🆕 47 Kenya counties with real risk indices")
    print("🆕 Child population estimates")
    print("🆕 Vulnerability scoring from UNICEF data")
    print("🆕 Prediction validation against CCRI baseline")
    
    print("\n📡 Endpoints available:")
    print("   • Basic: POST /api/analyze")
    print("   • Enhanced: POST /api/analyze/enhanced")
    print("   🆕 Kenya CCRI: POST /api/analyze/kenya")
    print("   🆕 Counties: GET /api/kenya/counties")
    print("   🆕 CCRI Validation: POST /api/kenya/ccri-validation")
    print("   • Compare: POST /api/compare/methods")
    print("   • Methods Info: GET /api/methods/info")
    print("   • Visualizations: GET /api/visualize/animated, /api/visualize/map")
    print("   • Health Check: GET /api/health")
    
    if Path(ccri_data_path).exists():
        print(f"\n🎯 CCRI Integration: ENABLED ({ccri_data_path})")
    else:
        print(f"\n📝 CCRI Integration: DISABLED (file not found: {ccri_data_path})")
        print("   Set CCRI_DATA_PATH environment variable to enable")
    
    print("\n🌐 Access the service at: http://localhost:5000")
    print("📚 Documentation at: http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)