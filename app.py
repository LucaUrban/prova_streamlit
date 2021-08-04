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
from plotly.subplots import make_subplots
from plotly.colors import n_colors
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import medcouple
import math
import scipy.stats as stats

st.title("Visual Information Quality Environment")
st.write("In this part you can upload your csv file either dropping your file or browsing it. Then the application will start showing all of the charts for the Dataset. " +
         "To change the file to be analyzed you have to regresh the page.")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    table = pd.read_csv(uploaded_file)

    # importing all other necessary files
    with urlopen('https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson') as response:
        eu_nut0 = json.load(response)
    
    with urlopen('https://raw.githubusercontent.com/eurostat/Nuts2json/master/2021/4326/60M/nutsrg_2.json') as response:
        eu_nut2 = json.load(response)

    with urlopen('https://raw.githubusercontent.com/eurostat/Nuts2json/master/2021/4326/60M/nutsrg_3.json') as response:
        eu_nut3 = json.load(response)

    # selection boxes columns
    col_an = [col for col in list(table) if len(table[col].unique()) < 10 or is_numeric_dtype(table[col])]
    col_mul = [col for col in list(table) if is_numeric_dtype(table[col])]
    lis_check = [{'label': col, 'value': col} for col in col_mul if col != col_mul[0]]

    widget = st.selectbox("what is the widget you want to display:",
                          ["Table", "Map Analysis", "Monodimensional Analysis", "Ratio Analysis", "Multidimensional Analysis", "Autocorrelation Analysis", 
                           "Feature Importance Analysis", "Heatmap", "Time series forecasting", "Anomalies check"], 0)
    
    if widget == "Table":
        # showing the table with the data
        st.header("Table")
        st.write("Data contained into the dataset:", table)
    
    if widget == "Map Analysis":
        # map-box part
        st.sidebar.subheader("Map area")
        nut_col = st.sidebar.selectbox("select the nut column", table.columns, 0)
        map_feature = st.sidebar.selectbox("select the feature column", col_mul, 0)
        map_q = st.sidebar.number_input("insert the quantile value", 0, 100, 50)

        st.header("Map")

        res = {nut_col: table[nut_col].unique(), map_feature: []}
        for nut_id in res[nut_col]:
            res[map_feature].append(table[table[nut_col] == nut_id][map_feature].quantile(map_q/100))
        res = pd.DataFrame(res)

        px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
        map_box = px.choropleth_mapbox(res, geojson = eu_nut2, locations = res[nut_col], featureidkey = 'properties.id',
                                   color = map_feature, color_continuous_scale = px.colors.cyclical.IceFire,
                                   range_color = (res[map_feature].min(), res[map_feature].max()),
                                   mapbox_style = "carto-positron",
                                   zoom = 3, center = {"lat": 47.4270826, "lon": 15.5322329},
                                   opacity = 0.5,
                                   labels = {map_feature: map_feature})

        st.plotly_chart(map_box, use_container_width=True)
    
    if widget == "Monodimensional Analysis":
        # mono variable analysis part
        st.header("Monodimension Analysis")

        st.sidebar.subheader("Monovariable Area")
        monoVar_col = st.sidebar.selectbox("select the monovariable feature", col_an, 6)
        monoVar_type = st.sidebar.selectbox("select the type of the chart", ["gauge plot", "pie chart"], 0)

        if monoVar_type == "gauge plot":
            monoVar_plot = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = table[monoVar_col].mean(),
                delta = {"reference": 2 * table[monoVar_col].mean() - table[monoVar_col].quantile(0.95)},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {'axis': {'range': [table[monoVar_col].min(), table[monoVar_col].max()]},
                         'steps' : [
                             {'range': [table[monoVar_col].min(), table[monoVar_col].quantile(0.05)], 'color': "lightgray"},
                             {'range': [table[monoVar_col].quantile(0.95), table[monoVar_col].max()], 'color': "gray"}]},
                title = {'text': "Gauge plot for the variable: " + monoVar_col}))
        else:
            monoVar_plot = px.pie(table, names = monoVar_col, title = "Pie chart for the variable: " + monoVar_col)

        st.plotly_chart(monoVar_plot, use_container_width=True)

    if widget == "Ratio Analysis":
        # ratio analysis part
        st.header("Ratio Analysis")

        st.sidebar.subheader("Ratio Area")
        ratio_num = st.sidebar.multiselect("select the ratio numerator", col_mul)
        ratio_den = st.sidebar.multiselect("select the ratio denominator", col_mul)
        
        res_ratio = pd.DataFrame(np.divide(np.nansum(table[ratio_num].values, axis = 1), np.nansum(table[ratio_den].values, axis = 1)), columns = ['R_1'])
        
        ratio_plot = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = res_ratio['R_1'].mean(),
            delta = {"reference": 2 * res_ratio['R_1'].mean() - res_ratio['R_1'].quantile(0.95)},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [res_ratio['R_1'].min(), res_ratio['R_1'].max()]},
                     'steps' : [
                         {'range': [res_ratio['R_1'].min(), res_ratio['R_1'].quantile(0.05)], 'color': "lightgray"},
                         {'range': [res_ratio['R_1'].quantile(0.95), res_ratio['R_1'].max()], 'color': "gray"}],},
            title = {'text': "Gauge plot for the variable: R_1"}))
        
        st.plotly_chart(ratio_plot, use_container_width=True)

        # map pplot + violin plot on the aggregated results
        left, right = st.beta_columns(2)
        with left: 
            ratio_vio_sel1 = st.selectbox("multivariable index col", table.columns, 0)
        with right:
            ratio_vio_sel2 = st.selectbox("multivariable index col", ['None'] + list(table.columns), 0)
        
        res_ratio['Sel'] = table[ratio_vio_sel1].str.slice(0, 2).values
        res = {ratio_vio_sel1: res_ratio['Sel'].unique(), 'R_1': []}
        for nut_id in res[ratio_vio_sel1]:
                  res['R_1'].append(res_ratio[res_ratio['Sel'] == nut_id]['R_1'].mean())
        res = pd.DataFrame(res)

        px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
        map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[ratio_vio_sel1], featureidkey = 'properties.ISO2',
                                       color = 'R_1', color_continuous_scale = px.colors.cyclical.IceFire,
                                       range_color = (res_ratio['R_1'].min(), res_ratio['R_1'].max()),
                                       mapbox_style = "carto-positron",
                                       zoom = 3, center = {"lat": 47.42, "lon": 15.53},
                                       opacity = 0.5,
                                       labels = {'R_1': 'R_1'})

        st.plotly_chart(map_box, use_container_width=True)
        
        uniques = res_ratio['Sel'].unique()
        cou_sel = st.selectbox("Choose the id of the country you want to explore", uniques, 0)
        if ratio_vio_sel2 == 'None':
            fig_vio = px.violin(res_ratio[res_ratio['Sel'] == cou_sel], y = "R_1", x = 'Sel', box = True, points = 'suspectedoutliers')
        else:
            res_ratio['Color'] = table[ratio_vio_sel2]
            fig_vio = px.violin(res_ratio[res_ratio['Sel'] == cou_sel], y = "R_1", x = 'Sel', color = 'Color', box = True, points = 'suspectedoutliers')
        st.plotly_chart(fig_vio, use_container_width=True)
    
    if widget == "Multidimensional Analysis":
        # multi variable analysis part
        st.header("Multidimension Analysis")

        st.sidebar.subheader("Multivariable Area")
        multi_index = st.sidebar.selectbox("multivariable index col", table.columns, 1)
        multi_time = st.sidebar.selectbox("multivariable time col", table.columns, 3)
        multiXax_col = st.sidebar.selectbox("multivariable X axis col", col_mul, 1)
        multiYax_col = st.sidebar.selectbox("multivariable Y axis col", col_mul, 2)
        multiSlider = st.sidebar.slider("multivarible time value", int(table[multi_time].min()), int(table[multi_time].max()), int(table[multi_time].min()))

        dff = table[table[multi_time] == multiSlider]
        multi_plot = px.scatter(x = dff[multiXax_col], y = dff[multiYax_col], hover_name = dff[multi_index])
        multi_plot.update_traces(customdata = dff[multi_index])
        multi_plot.update_xaxes(title = multiXax_col)
        multi_plot.update_yaxes(title = multiYax_col)

        st.plotly_chart(multi_plot, use_container_width=True)

        # time control charts
        el_id = st.selectbox("element ID for time control chart", table[multi_index].unique(), 1)

        dff_tcc = table[table[multi_index] == el_id][[multi_time, multiXax_col, multiYax_col]]
        if len(list(dff_tcc[multi_time].unique())) < dff_tcc.shape[0]:
            res = {multi_time: [], multiXax_col: [], multiYax_col: []}
            for el in list(dff_tcc[multi_time].unique()):
                res[multi_time].append(el)
                res[multiXax_col].append(dff_tcc[dff_tcc[multi_time] == el][multiXax_col].mean())
                res[multiYax_col].append(dff_tcc[dff_tcc[multi_time] == el][multiYax_col].mean())
            dff_tcc = pd.DataFrame(data = res)
        titles = ['<b>{}</b><br>{}'.format(el_id, multiXax_col), '<b>{}</b><br>{}'.format(el_id, multiYax_col)]

        for i in range(2):
            fig_tcc = go.Figure()
            if dff_tcc.shape[0] != 0:
                x_bar = []
                for inst in table[multi_index].unique():
                    inst_data = table[table[multi_index] == inst][list(dff_tcc)[i+1]]
                    if inst_data.count() != 0:
                        x_bar.append(inst_data.mean())

                x_barbar = round(sum(x_bar)/len(x_bar), 3)

                x_LCL = x_barbar - (1.88 * (dff_tcc[list(dff_tcc)[i+1]].quantile(0.95) - dff_tcc[list(dff_tcc)[i+1]].quantile(0.05)))
                x_UCL = x_barbar + (1.88 * (dff_tcc[list(dff_tcc)[i+1]].quantile(0.95) - dff_tcc[list(dff_tcc)[i+1]].quantile(0.05)))

                x_el = [i for i in range(int(dff_tcc[multi_time].min()), int(dff_tcc[multi_time].max()) + 1)]
                fig_tcc.add_trace(go.Scatter(x = dff_tcc[multi_time], y = dff_tcc[list(dff_tcc)[i+1]], mode = 'lines+markers', name = "Value"))
                fig_tcc.add_trace(go.Scatter(x = x_el, y = [x_UCL for _ in range(len(x_el))], mode = "lines", name = "Upper Bound"))
                fig_tcc.add_trace(go.Scatter(x = x_el, y = [x_LCL for _ in range(len(x_el))], mode = "lines", name = "Lower Bound"))

                fig_tcc.update_xaxes(showgrid = False)
                fig_tcc.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                                       xref = 'paper', yref = 'paper', showarrow=False, align = 'left',
                                       bgcolor = 'rgba(255, 255, 255, 0.5)', text = titles[i])
                fig_tcc.update_layout(xaxis_title = multi_time, yaxis_title = list(dff_tcc)[i+1])
                fig_tcc.update_layout(height = 250, margin = {'l': 20, 'b': 30, 'r': 10, 't': 10})

                st.plotly_chart(fig_tcc, use_container_width=True)
    
    if widget == "Autocorrelation Analysis":
        # crossfilter analysis part
        st.header("Autocorrelation Analysis")

        st.sidebar.subheader("Autocorrelation Area")
        cross_index = st.sidebar.selectbox("autocorrelation index col", table.columns, 1)
        cross_time = st.sidebar.selectbox("autocorrelation time col", table.columns, 3)
        cross_col = st.sidebar.selectbox("autocorrelation X axis col", col_mul, 1)
        crossSlider = st.sidebar.slider("autocorrelation time value", int(table[cross_time].min()), int(table[cross_time].max()-1), int(table[cross_time].min()))

        dff_cross_dw = table[table[cross_time] == crossSlider][[cross_index, cross_col]]
        dff_cross_up = table[table[cross_time] == crossSlider + 1][[cross_index, cross_col]]
        final_df_cross = pd.merge(dff_cross_dw, dff_cross_up, how = "inner", on = cross_index)

        cross_plot = px.scatter(x = final_df_cross[cross_col + "_x"], y = final_df_cross[cross_col + "_y"], hover_name = final_df_cross[cross_index])

        cross_plot.update_xaxes(title = cross_col)
        cross_plot.update_yaxes(title = cross_col + " Next Year")

        st.plotly_chart(cross_plot, use_container_width=True)

        st.subheader("Autocorrelation")
        st.write("Autocorrelation value: " + str(round(final_df_cross[cross_col + "_x"].corr(final_df_cross[cross_col + "_y"]), 5)))

        # difference timeseries plot
        el_id_diff = st.selectbox("element ID for differences timeseries chart", table[cross_index].unique())

        dff_diff = table[table[cross_index] == el_id_diff]
        if len(list(dff_diff[cross_time].unique())) < dff_diff.shape[0]:
            res = {cross_time: [], cross_col: []}
            for el in list(dff[cross_time].unique()):
                res[cross_time].append(el); res[cross_col].append(dff_diff[dff_diff[cross_time] == el][cross_col].mean())
            dff_diff = pd.DataFrame(data = res)
        title = '<b>{}</b><br>{}'.format(el_id_diff, cross_col)

        fig_diff = go.Figure(); flag = 0
        if dff_diff.shape[0] > 1:
            x = [[i, 0] for i in range(1, dff_diff.shape[0])]
            Y = [dff_diff[cross_col].iloc[dff_diff.shape[0] - i - 1] - dff_diff[cross_col].iloc[dff_diff.shape[0] - i] for i in range(1, dff_diff.shape[0])]
            reg = LinearRegression().fit(x, Y); coeff = reg.coef_; intercept = reg.intercept_

            fig_diff.add_trace(go.Scatter(x = [str(dff_diff[cross_time].iloc[dff_diff.shape[0] - i]) + "-" + str(dff_diff[cross_time].iloc[dff_diff.shape[0] - i - 1]) for i in range(1, dff_diff.shape[0])], 
                                     y = Y, mode = 'markers', name = "Value"))
            fig_diff.add_trace(go.Scatter(x = [str(dff_diff[cross_time].iloc[dff_diff.shape[0] - i]) + "-" + str(dff_diff[cross_time].iloc[dff_diff.shape[0] - i - 1]) for i in range(1, dff_diff.shape[0])], 
                                     y = [intercept + (i * coeff[0]) for i in range(dff_diff.shape[0])], 
                                     mode = 'lines', name = "Regression"))
            fig_diff.update_xaxes(showgrid=False)
            fig_diff.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                               xref='paper', yref='paper', showarrow=False, align='left',
                               bgcolor='rgba(255, 255, 255, 0.5)', text = title)
            fig_diff.update_layout(xaxis_title = cross_time, yaxis_title = list(dff_diff)[1])
            flag = 1

        fig_diff.update_layout(height = 400)

        st.plotly_chart(fig_diff, use_container_width = True)

        st.subheader("Regression Parameters")
        if flag == 1:
            st.write("Intercept: " + str(round(intercept, 4)))
            st.write("Slope: " + str(round(coeff[0], 4)))
        else: 
            st.write("None") 
            st.write("None")
    
    if widget == "Feature Importance Analysis":
        # pareto chart with feature importance on ridge regressor
        st.sidebar.subheader("Feature Importance Area")
        feaImp_target = st.sidebar.selectbox("multivariable index col", col_mul, 1)

        st.header("Feature Importance Analysis")

        fea_Imp_features = st.multiselect("Feature Importance multiselection box:", col_mul)
        scaler = StandardScaler(); train_nm = table[fea_Imp_features]

        for name_col in fea_Imp_features:
            train_nm[name_col].replace({np.nan : train_nm[name_col].mean()}, inplace = True)
        train_nm = scaler.fit_transform(train_nm)

        target = table[feaImp_target]
        target.replace({np.nan : 0}, inplace = True)

        Alpha = [.1, 1, 10, 100]; titles = tuple("Feature importance for alpha = " + str(alpha) for alpha in Alpha)
        Alpha = [[.1, 1], [10, 100]]

        # Create figure with secondary y-axis
        fig_tot = make_subplots(rows = 2, cols = 2, 
                                specs = [[{"secondary_y": True}, {"secondary_y": True}], 
                                         [{"secondary_y": True}, {"secondary_y": True}]], 
                                subplot_titles = titles)

        for num_row in range(2):
            for num_col in range(2):
                clf = Ridge(alpha = Alpha[num_row][num_col])
                clf.fit(train_nm, target)

                importance = clf.coef_
                for i in range(len(importance)):
                    if importance[i] < 0:
                        importance[i] *= -1
                dict_fin = {fea_Imp_features[i]: importance[i] for i in range(importance.shape[0])}
                dict_fin = {k: v for k, v in sorted(dict_fin.items(), key=lambda item: item[1], reverse = True)}
                dict_fin_per = {fea_Imp_features[i]: (importance[i] / np.sum(importance)) * 100 for i in range(importance.shape[0])}
                dict_fin_per = {k: v for k, v in sorted(dict_fin_per.items(), key=lambda item: item[1], reverse = True)}
                lis_final = []; res_par = 0
                for value in dict_fin_per.values():
                    res_par += value; lis_final.append(res_par)

                fig_tot.add_trace(
                    go.Bar(x = list(dict_fin_per.keys()), y = list(dict_fin_per.values()), 
                           marker_color = 'rgb(158,202,225)', marker_line_color = 'rgb(8,48,107)', 
                           marker_line_width = 1.5, opacity = 0.6, name = 'Value'),
                    row = num_row + 1, col = num_col + 1, secondary_y = False
                )

                fig_tot.add_trace(
                    go.Scatter(x = list(dict_fin_per.keys()), y = lis_final, line_color = 'rgb(255, 150, 0)'),
                    row = num_row + 1, col = num_col + 1, secondary_y = True
                )

                # Add figure title
                fig_tot.update_layout(
                    title_text = "Feature importances", showlegend = False
                )

                # Set x-axis title
                fig_tot.update_xaxes(title_text = "Variables")

                # Set y-axes titles
                fig_tot.update_yaxes(title_text="<b>Value</b> of importance", secondary_y=False)
                fig_tot.update_yaxes(title_text="<b>%</b> of importance", secondary_y=True)

        fig_tot.update_layout(height = 600)
        st.plotly_chart(fig_tot, use_container_width=True)
        
    if widget == "Heatmap":
        heat_cols = st.multiselect("Choose the columns for the correlation heatmap:", col_mul)
        
        if len(heat_cols) >= 2:
            fig_heat = px.imshow(table[heat_cols].corr(), x = heat_cols,  y = heat_cols, 
                                 labels = dict(color = "Corr Value"), color_continuous_scale = px.colors.sequential.Hot)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("Yuo have to choose at least two columns")
        
    if widget == "Time series forecasting":
        st.header("Time series forecasting")

        use_col = st.sidebar.selectbox("Chosen Variable", col_mul, 0)
        modality = st.sidebar.selectbox("Forecasting Method", ["Rolling Forecast", "Recurring Forecast"], 0)
        index = st.sidebar.selectbox("Index col", table.columns, 0)
        time = st.sidebar.selectbox("Time col", table.columns, 0)
 
        # pre-work
        data = table[[index, time, use_col]].sort_values(by=[time])
        res = np.array([]); ids = []
        for id in data[index].unique():
            el = data[data[index] == id][use_col]
            n = len(list(data[time].unique()))
            if el.shape[0] == n:
                res = np.concatenate([res, el.values])
                ids.append(id)
        res = res.reshape(res.shape[0]//n, n)
        col_mean = np.nanmean(res, axis = 1)

        #Find indices that you need to replace
        inds = np.where(np.isnan(res))

        #Place column means in the indices. Align the arrays using take
        res[inds] = np.take(col_mean, inds[1])
        
        # fit the init model and making the predictions
        pred_ar = np.array([]); pred_ma = np.array([]); pred_arma = np.array([]); pred_arima = np.array([])

        for i in range(res.shape[0]):
            pred_ar = np.append(pred_ar, AutoReg(res[i, 0:res.shape[1]-1], lags = 1).fit().predict(len(res), len(res)))
            pred_ma = np.append(pred_ma, ARIMA(res[i, 0:res.shape[1]-1], order=(0, 0, 1)).fit().predict(len(res), len(res)))
            pred_arma = np.append(pred_arma, ARIMA(res[i, 0:res.shape[1]-1], order=(2, 0, 1)).fit().predict(len(res), len(res)))
            pred_arima = np.append(pred_arima, ARIMA(res[i, 0:res.shape[1]-1], order=(1, 1, 1)).fit().predict(len(res), len(res), typ='levels'))
        
        
        # visual part
        mse_mins = np.array([mean_squared_error(pred_ar, res[:, res.shape[1]-1]), mean_squared_error(pred_ma, res[:, res.shape[1]-1]),
                             mean_squared_error(pred_arma, res[:, res.shape[1]-1]), mean_squared_error(pred_arima, res[:, res.shape[1]-1])])
        st.table(pd.DataFrame(mse_mins.reshape((1, 4)), columns = ['AR', 'MA', 'ARMA', 'ARIMA'], index = ['MSE error']))
         
        ch_model = st.selectbox("Choose the model you want to use to forecast the next periods", ['AR', 'MA', 'ARMA', 'ARIMA'])
        ch_id = st.selectbox("Choose element you want to forecast", ids)
        num_fut_pred = st.sidebar.number_input("Insert the number of periods you want to forecast ", 1, 10, 1)
        fig_forecasting = go.Figure()
        
        # forecasting
        par_for = []; rif = res[ids.index(ch_id)]
        for i in range(num_fut_pred + 1):
            # prediction based on the chosen model
            if ch_model == 'AR':
                pred = AutoReg(rif, lags = 1).fit().predict(len(rif), len(rif))[0]

            if ch_model == 'MA':
                pred = ARIMA(rif, order=(0, 0, 1)).fit().predict(len(rif), len(rif))[0]

            if ch_model == 'ARMA':
                pred = ARIMA(rif, order=(2, 0, 1)).fit().predict(len(rif), len(rif))[0]

            if ch_model == 'ARIMA':
                pred = ARIMA(rif, order=(1, 1, 1)).fit().predict(len(rif), len(rif))[0]
                
            par_for.append(pred); rif = np.append(rif, pred)
            # rolling forecasting
            if modality == "Rolling Forecast":
                rif = rif[1:]
        
        fig_forecasting.add_trace(go.Scatter(x = [max(list(data[time].unique())) + j for j in range(num_fut_pred + 1)], 
                                             y = [res[ids.index(ch_id), -1]] + par_for, mode = 'lines+markers', name = "Prediction", line = dict(color = 'firebrick')))
        fig_forecasting.add_trace(go.Scatter(x = list(data[time].unique()), y = data[data[index] == ch_id][use_col].values, mode = 'lines+markers', name = "Value", 
                                             line = dict(color = 'royalblue')))
        fig_forecasting.update_layout(xaxis_title = use_col, yaxis_title = time, title_text = "Values over time with future predictions")
        st.plotly_chart(fig_forecasting, use_container_width=True)
        
    if widget == "Anomalies check":
        use_col = st.sidebar.selectbox("Chosen Variable", col_mul, 0)
        
        # MLE normal
        mu_hat = table[use_col].mean()
        sigma_hat = math.sqrt(((table[use_col] - table[use_col].mean()) ** 2).sum() / table[use_col].count())

        # MLE exponential
        lambda_hat_exp = table[use_col].count() / table[use_col].sum()

        # MLE log-normal
        mu_hat_log = (np.ma.log(table[use_col].values)).sum() / table[use_col].count()
        sigma_hat_log = math.sqrt(((np.ma.log(table[use_col].values) - mu_hat_log) ** 2).sum() / table[use_col].count())
        
        # MLE weibull
        a, alpha_hat, b, beta_hat = stats.exponweib.fit(table[use_col], floc=0, fa=1)
        
        # computing the p-values for all the distributions
        result_norm = stats.kstest(table[[use_col]].values.flatten(), 'norm', (mu_hat, sigma_hat))
        result_exp = stats.kstest(table[[use_col]].values.flatten(), 'expon')
        result_lognorm = stats.kstest(table[[use_col]].values.flatten(), 'lognorm', (mu_hat_log, sigma_hat_log))
        result_weibull2 = stats.kstest(table[[use_col]].values.flatten(), 'dweibull', (beta_hat, alpha_hat))
        
        # visual part
        dis_fit = [[result_norm[1], result_exp[1], result_lognorm[1], result_weibull2[1]], 
                   [result_norm[1] > 0.05, result_exp[1] > 0.05, result_lognorm[1] > 0.05, result_weibull2[1] > 0.05]]
        st.table(pd.DataFrame(dis_fit, columns = ['Normal', 'Exponential', 'Log-Norm', 'Weibul'], index = ['P-value', 'P > t']))

        ch_distr = st.selectbox("Choose the distribution you want to use for the anomalies estimation", ['Normal', 'Exponential', 'Log-Norm', 'Weibull'])
        fig_distr = go.Figure(data = [go.Histogram(x = table[use_col], 
                                                   xbins = dict(start = table[use_col].min(), end = table[use_col].max(), size = (table[use_col].max() - table[use_col].min()) / 25),
                                                   autobinx = False, 
                                                   histnorm = 'probability density')])
        
        x_pos = np.linspace(table[use_col].min(), table[use_col].max(), 25)
        if ch_distr == 'Normal':
            fig_distr.add_trace(go.Scatter(x = x_pos, 
                                           y = stats.norm(mu_hat, sigma_hat).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        if ch_distr == 'Exponential':
            fig_distr.add_trace(go.Scatter(x = x_pos, 
                                           y = stats.expon(lambda_hat_exp).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        if ch_distr == 'Log-Norm':
            fig_distr.add_trace(go.Scatter(x = x_pos, 
                                           y = stats.lognorm(mu_hat_log, sigma_hat_log).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        if ch_distr == 'Weibull':
            fig_distr.add_trace(go.Scatter(x = x_pos, 
                                           y = stats.dweibull(alpha_hat, beta_hat).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        fig_distr.update_layout(title = 'Hist plot to comapre data with possible underlying distribution', xaxis_title = use_col + ' values', yaxis_title = use_col + ' PMF and ch. distr. PDF')
        st.plotly_chart(fig_distr, use_container_width=True)
         
        # outlier part
        tukey_const = st.number_input("Insert the constant for the fence interquantile value", 0.5, 7.5, 1.5)
        Q3 = table[use_col].quantile(0.75); Q1 = table[use_col].quantile(0.25); ITQ = Q3- Q1
        st.write(stats.skewtest(table[use_col].values)[1])
        if stats.skewtest(table[use_col].values)[1] >= 0.01:
            st.table(pd.DataFrame(np.array([table[table[use_col] <= Q1 - (2 * tukey_const * ITQ)].shape[0], 
                                            table[(table[use_col] >= Q1 - (2 * tukey_const * ITQ)) & (table[use_col] <= Q1 - (tukey_const * ITQ))].shape[0],
                                            table[(table[use_col] >= Q3 + (tukey_const * ITQ)) & (table[use_col] <= Q3 + (2 * tukey_const * ITQ))].shape[0],
                                            table[table[use_col] >= Q3 + (2 * tukey_const * ITQ)].shape[0]]).reshape(1, 4),
                                  index = ['Number'], columns = ['Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers']))
        else:
            # calculating the medcouple function for the tukey fence
            if table[use_col].dropna().values.shape[0] > 5000:
                MC = medcouple(table[use_col].values[np.random.choice(table[use_col].dropna().values.shape[0], 5000)])
            else:
                MC = medcouple(table[use_col].values)
                
            # calculating the tukey fence
            if MC > 0:
                st.table(pd.DataFrame(np.array([table[table[use_col] <= Q1 - (2 * tukey_const * math.exp(-4 * MC) * ITQ)].shape[0], 
                                                table[(table[use_col] >= Q1 - (2 * tukey_const * math.exp(-4 * MC) * ITQ)) & (table[use_col] <= Q1 - (tukey_const * math.exp(-4 * MC) * ITQ))].shape[0],
                                                table[(table[use_col] >= Q3 + (tukey_const * math.exp(3 * MC) * ITQ)) & (table[use_col] <= Q3 + (2 * tukey_const * math.exp(3 * MC) * ITQ))].shape[0],
                                                table[table[use_col] >= Q3 + (2 * tukey_const * math.exp(3 * MC) * ITQ)].shape[0]]).reshape(1, 4),
                                      index = ['Number'], columns = ['Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers']))
            else:
                st.table(pd.DataFrame(np.array([table[table[use_col] <= Q1 - (2 * tukey_const * math.exp(-3 * MC) * ITQ)].shape[0], 
                                                table[(table[use_col] >= Q1 - (2 * tukey_const * math.exp(-3 * MC) * ITQ)) & (table[use_col] <= Q1 - (tukey_const * math.exp(-3 * MC) * ITQ))].shape[0],
                                                table[(table[use_col] >= Q3 + (tukey_const * math.exp(4 * MC) * ITQ)) & (table[use_col] <= Q3 + (2 * tukey_const * math.exp(4 * MC) * ITQ))].shape[0],
                                                table[table[use_col] >= Q3 + (2 * tukey_const * math.exp(4 * MC) * ITQ)].shape[0]]).reshape(1, 4),
                                      index = ['Number'], columns = ['Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers']))
                
            
            
        
        
