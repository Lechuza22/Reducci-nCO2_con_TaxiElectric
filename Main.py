import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar los datasets
@st.cache_data
def load_data():
    locacion_y_coordenadas = pd.read_csv('LocacionYcoordenadas.csv')
    yellow_trip_data = pd.read_csv('Yellow_Tripdata_2024-10_r_24mb.csv')
    vehicle_fuel_economy = pd.read_csv('transformed_Vehicle Fuel Economy Data.csv')
    return locacion_y_coordenadas, yellow_trip_data, vehicle_fuel_economy

locacion_y_coordenadas, yellow_trip_data, vehicle_fuel_economy = load_data()

# Calcular el promedio de emisiones de CO2 por milla
co2_emission_columns = [col for col in vehicle_fuel_economy.columns if 'co2' in col.lower() and 'gpm' in col.lower()]
average_co2_per_mile = vehicle_fuel_economy[co2_emission_columns].mean(axis=0).mean()

# Preparar los datos de los viajes diarios
yellow_trip_data['tpep_pickup_datetime'] = pd.to_datetime(yellow_trip_data['tpep_pickup_datetime'])
yellow_trip_data['pickup_date'] = yellow_trip_data['tpep_pickup_datetime'].dt.date
daily_trip_data = yellow_trip_data.groupby('pickup_date').agg(
    total_distance=('trip_distance', 'sum'),
    total_trips=('trip_distance', 'count')
).reset_index()
daily_trip_data['daily_emissions'] = daily_trip_data['total_distance'] * average_co2_per_mile / 1000

# Modelo de predicción de emisiones con Machine Learning
def train_prediction_model(daily_trip_data):
    X = daily_trip_data[['total_trips']]
    y = daily_trip_data['daily_emissions']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_prediction_model(daily_trip_data)

# Predicción para los próximos 31 días
future_trips = np.linspace(daily_trip_data['total_trips'].min(), daily_trip_data['total_trips'].max(), 31).reshape(-1, 1)
future_emissions = model.predict(future_trips)

# Configuración de la app de Streamlit
st.title("Análisis y Predicción de Emisiones con Autos Eléctricos")

menu = st.sidebar.radio("Menú", ["Predicción de Emisiones", "Integración de Autos Eléctricos"])

if menu == "Predicción de Emisiones":
    st.header("Predicción de Emisiones de CO2")
    
    # Graficar predicción
    plt.figure(figsize=(10, 6))
    plt.plot(daily_trip_data['pickup_date'], daily_trip_data['daily_emissions'], label="Emisiones Históricas", marker='o')
    plt.plot(pd.date_range(start=daily_trip_data['pickup_date'].iloc[-1], periods=31, freq='D'), 
             future_emissions, label="Predicción", linestyle='--', color='orange')
    plt.title("Predicción de Emisiones de CO2", fontsize=14)
    plt.xlabel("Fecha")
    plt.ylabel("Emisiones de CO2 (kg)", fontsize=12)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

elif menu == "Integración de Autos Eléctricos":
    st.header("Integración de Autos Eléctricos")
    
    selected_reduction = st.slider(
        "Selecciona el porcentaje de reducción de emisiones de CO2:",
        min_value=5,
        max_value=70,
        value=20,
        step=5
    )

    # Cálculo basado en el porcentaje seleccionado
    average_trips_per_day_per_car = 9

    def calculate_electric_vehicles_needed(reduction_target_percentage, daily_trip_data, average_trips_per_day_per_car):
        current_emissions = daily_trip_data['daily_emissions'].sum()
        target_emissions = current_emissions * (1 - reduction_target_percentage / 100)
        emissions_to_reduce = current_emissions - target_emissions

        total_trips = daily_trip_data['total_trips'].sum()
        emissions_per_trip = current_emissions / total_trips

        trips_needed_to_replace = emissions_to_reduce / emissions_per_trip
        electric_cars_needed = trips_needed_to_replace / (average_trips_per_day_per_car * len(daily_trip_data))

        return electric_cars_needed, target_emissions

    electric_cars_needed, target_emissions = calculate_electric_vehicles_needed(
        selected_reduction, daily_trip_data, average_trips_per_day_per_car
    )

    st.write(f"Para reducir las emisiones de CO2 en un {selected_reduction}% se necesitan aproximadamente {electric_cars_needed:.0f} autos eléctricos.")
    st.write(f"Las emisiones objetivo serán de aproximadamente {target_emissions:.2f} kg de CO2 durante el próximo mes.")
