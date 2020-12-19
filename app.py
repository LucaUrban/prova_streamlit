import streamlit as st
import time
import base64
import os
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash_table
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from urllib.request import urlopen
import json
import random

# importing the table and all other necessary files
table = pd.read_csv("https://raw.github.com/LucaUrban/prova_streamlit/main/table_final.csv")

# showing the table with the data
st.write("Data contained into the dataset:", data)
