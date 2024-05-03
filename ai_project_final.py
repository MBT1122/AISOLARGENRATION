import pandas as pd
import numpy as np
import os
import zipfile
import streamlit as st
from functools import reduce
from tensorflow.keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

@st.cache
def load_data():
    # Plant Generation Data
    df_gen = pd.read_csv('Plant_2_Generation_Data.csv')
    # Plant Weather Data
    df_weather = pd.read_csv('Plant_2_Weather_Sensor_Data.csv')
    return df_gen, df_weather

def prepare_data(df_gen, df_weather):
    df_gen.drop('PLANT_ID', axis=1, inplace=True)
    df_weather.drop('PLANT_ID', axis = 1, inplace=True)

    Inverter_labels = {inverter_no:inverter_name for inverter_name,inverter_no in enumerate(df_gen['SOURCE_KEY'].unique(),1)}
    df_gen['Inverter_No'] = df_gen['SOURCE_KEY'].map(Inverter_labels)

    df_gen.drop('SOURCE_KEY',axis=1,inplace=True)
    df_weather.drop('SOURCE_KEY',axis=1,inplace=True)

    df_gen = df_gen[['DATE_TIME','Inverter_No' ,'DC_POWER', 'AC_POWER']]

    from functools import reduce
    def rename_columns(df, suffix):
        renamed_columns = {col: f"{col}_{suffix}" for col in df.columns if col != 'DATE_TIME'}
        return df.rename(columns=renamed_columns)

    grouped = df_gen.groupby('Inverter_No')
    dfs = []
    for i, group in enumerate(df_gen['Inverter_No'].unique(), start=1):
        df_group = grouped.get_group(group)
        dfs.append(rename_columns(df_group, i))

    df_new = reduce(lambda left, right: pd.merge(left, right, on=['DATE_TIME'], how='outer'), dfs)

    df_weather['DATE_TIME'] = df_weather['DATE_TIME'].astype(str)
    df_new['DATE_TIME'] = df_new['DATE_TIME'].astype(str)

    df = df_weather.merge(df_new,left_on='DATE_TIME',right_on='DATE_TIME',how='outer')

    tb = pd.date_range('15-05-2020','16-05-2020',freq='15min')
    tb=tb[:-1]
    ts = tb.strftime('%H:%M')
    block_dict = {}
    j=1
    for i in range(len(ts)):
        block_dict[ts[i]] =  j
        j+=1

    df['TIME'] = df['DATE_TIME'].apply(lambda x:str(x)[-8:-3])
    df['DATE'] = pd.to_datetime(df['DATE_TIME']).dt.date
    df['BLOCK'] = pd.to_datetime(df['TIME']).astype(str).apply(lambda x:block_dict[str(x)[-8:-3]])
    df.drop('DATE_TIME',axis=1,inplace=True)
    np.save('timestamp_block_dictionary.npy',block_dict)

    cols = df.columns.tolist()
    df = df[[cols[-1]]+[cols[-2]]+[cols[-3]]+cols[:-3]]
    return df

def plot_solar_generation(df, date_input, time_input, num_inverters):
    fixed_year = 2020
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

    hour = pd.to_datetime(time_input).hour

    if hour < 6 or hour >= 17:
        st.write("No solar generation during non-sun hours (17:00 - 06:00). No graph will be generated.")
        return

    full_date_input = f"{fixed_year}-{date_input}"
    day_start = pd.to_datetime(f"{full_date_input} {hour}:00:00")
    day_end = pd.to_datetime(f"{full_date_input} {hour+1}:00:00")

    mask = (df['DATE_TIME'].dt.hour == hour) & (df['DATE_TIME'].dt.date == pd.to_datetime(full_date_input).date())
    df_filtered = df.loc[mask]

    if df_filtered.empty:
        st.write("No data available for the selected hour.")
        return

    plt.figure(figsize=(10, 5))
    time_ticks = pd.date_range(start=day_start, end=day_end, freq='15T')
    for i in range(1, num_inverters + 1):
        dc_power_col = f'DC_POWER_{i}'
        if dc_power_col in df_filtered.columns:
            plt.plot(df_filtered['DATE_TIME'], df_filtered[dc_power_col], label=f'Inverter {i}')

    plt.title(f'Solar Power Generation for {date_input} during {hour}:00-{hour+1}:00')
    plt.xlabel('Time')
    plt.ylabel('DC Power')
    plt.xticks(time_ticks, [t.strftime('%H:%M') for t in time_ticks])
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def main():
    st.title("Solar Generation Analysis")

    df_gen, df_weather = load_data()
    df = prepare_data(df_gen, df_weather)

    date_input = st.text_input("Enter date (MM-DD format):")
    time_input = st.text_input("Enter time (HH:MM format):")
    num_inverters = st.number_input("Enter number of solar inverters:", min_value=1, step=1)

    if st.button("Plot Solar Generation"):
        plot_solar_generation(df, date_input, time_input, num_inverters)

if __name__ == "__main__":
    main()
