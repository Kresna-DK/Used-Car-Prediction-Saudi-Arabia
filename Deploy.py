# Deploy Used Car Predictor

# ======================================================
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import pickle
# ======================================================

# Title
st.write('''
# USED CAR RECOMMENDATION
''')

# ======================================================
car_make = ['GMC', 'Land Rover', 'Kia', 'Mazda', 'Porsche', 'Hyundai',
       'Toyota', 'Chrysler', 'Lexus', 'Nissan', 'Mitsubishi', 'Ford',
       'MG', 'Chevrolet', 'Mercedes', 'Jeep', 'BMW', 'Lincoln', 'Genesis',
       'Honda', 'Dodge', 'HAVAL', 'Zhengzhou', 'Cadillac', 'Changan',
       'Aston Martin', 'Renault', 'Suzuki', 'Mercury', 'INFINITI', 'Audi',
       'Rolls-Royce', 'MINI', 'Other', 'BYD', 'Volkswagen',
       'Victory Auto', 'Geely', 'Classic', 'Isuzu', 'Daihatsu',
       'Maserati', 'Hummer', 'GAC', 'Lifan', 'Bentley', 'Chery', 'Jaguar',
       'Peugeot', 'Foton', 'Škoda', 'Fiat', 'Iveco', 'FAW', 'Great Wall',
       'Ferrari']


car_type = ['Yukon', 'Range Rover', 'Optima', 'CX3', 'Cayenne S', 'Sonata',
       'Avalon', 'C300', 'Land Cruiser', 'LS', 'FJ', 'Tucson', 'Sunny',
       'Pajero', 'Azera', 'Focus', '5', 'Spark', 'Pathfinder', 'Accent',
       'ML', 'Corolla', 'Tahoe', 'A', 'Altima', 'Expedition', 'Senta fe',
       'Liberty', 'X', 'Land Cruiser Pickup', 'VTC', 'Malibu', 'The 5',
       'Patrol', 'Grand Cherokee', 'Camry', 'SL', 'Previa', 'SEL', 'MKZ',
       'Datsun', 'Hilux', 'GLC', 'Edge', '6', 'Innova', 'Navara', 'G80',
       'Carnival', 'Suburban', 'Camaro', 'Accord', 'Taurus', 'Optra',
       'Elantra', 'Flex', 'S', 'Cerato', 'Furniture', 'Murano',
       'Land Cruiser 70', '3', 'Charger', 'H6', 'Hiace', 'Fusion', 'Aveo',
       'CX9', 'Yaris', 'Sierra', 'Durango', 'Pick up', 'CT-S',
       'Sylvian Bus', 'ES', 'Navigator', 'Opirus', 'The 7', 'Creta',
       'CS35', 'The 3', 'GLE', 'Sedona', 'Victoria', 'Prestige', 'CLA',
       'Vanquish', 'Safrane', 'Cores', 'Cadenza', "D'max", 'Silverado',
       'Rio', 'Maxima', 'X-Trail', 'Cruze', 'C', 'Seven', 'Prado',
       'Caprice', 'Grand Marquis', 'LX', 'Impala', 'QX', 'Blazer', 'H1',
       'Rav4', 'The M', 'Genesis', 'Traverse', 'Civic', 'Echo Sport',
       'Challenger', 'CL', 'Wrangler', 'A6', 'Dokker', 'CX5', 'Mohave',
       'Explorer', 'Ghost', 'Rush', 'Sentra', 'Cherokee', 'Copper',
       'Veloster', 'E', 'G', 'IS', 'Fluence', 'Vego', 'Ciocca', 'Other',
       'Marquis', 'Q', 'F3', 'Kona', 'UX', 'Beetle', 'F150', 'Lancer',
       'Van R', 'Mustang', 'CS35 Plus', 'DB9', 'Sorento', 'APV', 'Viano',
       'EC7', 'Safari', 'Cadillac', 'Duster', 'RX', 'Platinum', 'Carenz',
       'Avanza', 'Emgrand', 'D-MAX', 'Dyna', 'Z', 'Coupe S', 'Odyssey',
       'Panamera', 'Juke', 'Sportage', 'C200', 'Attrage', 'GS', 'X-Terra',
       'Picanto', 'CT5', 'KICKS', 'Gran Max', 'Cayman', 'A8', 'Levante',
       'Montero', '300', 'A3', 'Touareg', 'Passat', 'Delta', 'Acadia',
       'H3', 'GS3', 'Coupe', 'Cayenne Turbo', 'Colorado', 'Trailblazer',
       'Vitara', 'Kaptiva', 'Nativa', 'CLS', 'LF X60', 'Aurion', 'Koleos',
       'Abeka', 'Flying Spur', 'Pilot', 'L200', 'Ranger', 'Escalade',
       'A7', 'Quattroporte', 'Compass', 'Bus Urvan', 'Macan', 'Azkarra',
       'GL', 'City', 'Symbol', 'Ertiga', 'RX5', 'Envoy', 'CT6',
       'Fleetwood', 'Tiggo', 'Q5', 'A4', 'XJ', 'H2', 'HS', 'Seltos',
       'RX8', '301', 'EC8', '3008', 'Suvana', 'Prius', 'Cayenne', 'Eado',
       'Royal', 'NX', 'CS75', 'F-Pace', 'Coolray', 'CS85', 'Jimny', 'GC7',
       '360', 'A5', 'S300', 'Superb', 'Ram', 'Terrain', 'Cressida', '500',
       'Armada', 'Logan', '5008', 'Tiguan', 'Golf', 'CS95', 'S5', '911',
       'Camargue', 'M', 'Defender', 'Daily', 'Nitro', 'Mini Van', 'Pegas',
       'Grand Vitara', 'FX', 'L300', 'Coaster', 'Discovery', 'Montero2',
       'Bentayga', 'Z370', 'Bus County', 'Stinger', 'SRT', 'K5', 'CT4',
       'F Type', 'CC', 'ASX', 'Carens', 'XT5', 'Tuscani', '4Runner',
       'ATS', 'CRV', 'The 4', 'HRV', 'X7', 'GX', 'X40', 'Q7', 'ZS', 'G70',
       'Megane', 'Power', 'B50', 'Town Car', 'Van', '2', 'i40', 'XF',
       'RC', 'Doblo', 'MKX', 'The 6', 'Jetta', 'Soul', 'Lumina', 'Dzire',
       'Avante', 'Z350', 'CX7', 'Countryman', 'GTB 599 Fiorano',
       'Prestige Plus', 'MKS', 'Milan', 'Savana', 'S8']

car_region = ['Riyadh', 'Hafar Al-Batin', 'Abha', 'Makkah', 'Dammam', 'Jeddah',
       'Khobar', 'Al-Baha', 'Al-Ahsa', 'Jazan', 'Aseer', 'Al-Medina',
       'Al-Namas', 'Taef', 'Tabouk', 'Qassim', 'Al-Jouf', 'Yanbu',
       'Najran', 'Hail', 'Jubail', 'Wadi Dawasir', 'Arar', 'Besha',
       'Qurayyat', 'Sakaka']

car_origin = ['Saudi', 'Gulf Arabic', 'Other', 'Unknown']

car_gear = ['Automatic', 'Manual']

car_options = ['Full', 'Semi Full', 'Standard']

# ======================================================

# Sidebar
st.sidebar.header("Please input the car's feature")

def car_input():

    ## Numerical feature
    Year = st.sidebar.number_input(label= "Year", min_value= 1900, max_value= 2024, value= 2010)
    Engine_Size = st.sidebar.number_input(label= "Engine_Size", min_value= 1000, max_value= 9000, value= 2000, step= 100)
    Mileage = st.sidebar.number_input(label= "Mileage", min_value= 0, max_value= 3_000_000, value= 0)

    ## Categorical feature
    Make = st.sidebar.selectbox(label= "Make", options= (car_make))
    Type = st.sidebar.selectbox(label= "Type", options= (car_type))
    Region = st.sidebar.selectbox(label= "Region", options= (car_region))
    Origin = st.sidebar.selectbox(label= "Origin", options= (car_origin))
    Gear_Type = st.sidebar.selectbox(label= "Gear_Type", options= (car_gear))
    Options = st.sidebar.selectbox(label= "Options", options= (car_options))
        
    st.subheader("Car Features")
    df = pd.DataFrame()
    df['Year'] = [Year]
    df["Engine_Size"] = [Engine_Size]
    df["Mileage"] = [Mileage]
    df["Make"] = [Make]
    df["Type"] = [Type]
    df["Region"] = [Region]
    df["Origin"] = [Origin]
    df["Gear_Type"] = [Gear_Type]
    df["Options"] = [Options]
    
    return df

df_car = car_input()
df_car.index = ['value']

# Car price recommendation

# Change into the correct directory with the SAV file (Model_Used_Cars_XGB.sav)
model_loaded = pickle.load(open(r'4_Projects\1_Used_Cars_Saudi_Arabia\Model_Used_Cars_XGB.sav', 'rb'))

col1, col2 = st.columns(2)

# Left column (col1)
with col1:
    st.write(df_car.transpose())

with col2:
    st.header("Prediction")

    if st.button("Recommend"):
        recommendation = model_loaded.predict(df_car)
        if recommendation > 0:
            st.success("The recommended price is {:.2f} Riyal".format(recommendation[0]))
        else:
            st.write("Something went wrong, please try again.")