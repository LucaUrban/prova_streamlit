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

# importing the table and all other necessary files
table = pd.read_csv("https://raw.github.com/LucaUrban/prova_streamlit/main/table_final.csv")

# showing the table with the data
st.write("Data contained into the dataset:", data)
