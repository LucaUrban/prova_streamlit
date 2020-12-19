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
col_an = [col for col in list(table) if len(table[col].unique()) < 10 or is_numeric_dtype(table[col])]
col_mul = [col for col in list(table) if is_numeric_dtype(table[col])]
lis_check = [{'label': col, 'value': col} for col in col_mul if col != col_mul[0]]

# showing the table with the data
st.write("Data contained into the dataset:", table)

# map-box part
st.header("Map")

st.sidebar.subheader("Map area")
nut_col = st.sidebar.selectbox("select the nut column", table.columns, 0)
map_feature = st.sidebar.selectbox("select the feature column", col_mul, 0)
map_q = st.sidebar.number_input("insert the quantile value", 0, 100, 50)

px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
map_box = px.choropleth_mapbox(table, geojson = eu_nut2, locations = table[nut_col], featureidkey = 'properties.id',
                           color = map_feature, color_continuous_scale = px.colors.cyclical.IceFire,
                           range_color = (table[map_feature].min(), table[map_feature].max()),
                           mapbox_style = "carto-positron",
                           zoom = 3, center = {"lat": 47.4270826, "lon": 15.5322329},
                           opacity = 0.5,
                           labels = {map_feature: map_feature})

st.plotly_chart(map_box, use_container_width=True)
