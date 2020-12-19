import streamlit as st
import time
import base64
import os
from urllib.parse import quote as urlquote
from urllib.request import urlopen
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import json
import random
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# importing the table and all other necessary files
table = pd.read_csv("https://raw.github.com/LucaUrban/prova_streamlit/main/table_final.csv")
with urlopen('https://raw.githubusercontent.com/eurostat/Nuts2json/master/2021/4326/60M/nutsrg_2.json') as response:
    eu_nut2 = json.load(response)

with urlopen('https://raw.githubusercontent.com/eurostat/Nuts2json/master/2021/4326/60M/nutsrg_3.json') as response:
    eu_nut3 = json.load(response)

# selection boxes columns
col_an = [col for col in list(table) if len(table[col].unique()) < 10 or is_numeric_dtype(table[col])]; var_in = col_an[0]
col_mul = [col for col in list(table) if is_numeric_dtype(table[col])]
lis_check = [{'label': col, 'value': col} for col in col_mul if col != col_mul[0]]

# showing the table with the data
st.write("Data contained into the dataset:", table)

st.write("Map")

px.set_mapbox_access_token(open("https://raw.github.com/LucaUrban/prova_streamlit/main/mapbox_token.txt").read())
map_box = px.choropleth_mapbox(table, geojson = eu_nut2, locations = table['Nuts'], featureidkey = 'properties.id',
                           color = 'FOUND_YEAR', color_continuous_scale = px.colors.cyclical.IceFire,
                           range_color = (table['FOUND_YEAR'].min(), table['FOUND_YEAR'].quantile(0.95)),
                           mapbox_style = "carto-positron",
                           zoom = 3, center = {"lat": 47.4270826, "lon": 15.5322329},
                           opacity = 0.5,
                           labels = {'FOUND_YEAR': 'FOUND_YEAR'})

st.plotly_chart(map_box, use_container_width=True)
