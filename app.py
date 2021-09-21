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
                           "Feature Importance Analysis", "Heatmap", "Time series forecasting", "Anomalies check", "Consistency checks"], 0)
    
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
        
        uniques = list(res_ratio['Sel'].unique())
        cou_sel = st.selectbox("Choose the id of the country you want to explore", ['All ids'] + uniques, 0)
        res_ratio['Un Name'] = table['Institution Name']
        if cou_sel == 'All ids':
            if ratio_vio_sel2 == 'None':
                fig_vio = px.violin(res_ratio, y = "R_1", box = True, points = 'suspectedoutliers', title = 'Violin plot for the created ratio', hover_data = ['Un Name'])
            else:
                res_ratio['Color'] = table[ratio_vio_sel2]
                fig_vio = px.violin(res_ratio, y = "R_1", color = 'Color', box = True, points = 'suspectedoutliers', title = 'Violin plot for the created ratio', 
                                    hover_data = ['Un Name'])
        else:
            if ratio_vio_sel2 == 'None':
                fig_vio = px.violin(res_ratio[res_ratio['Sel'] == cou_sel], y = "R_1", x = 'Sel', box = True, points = 'suspectedoutliers', 
                                    title = 'Violin plot for the created ratio', hover_data = ['Un Name'])
            else:
                res_ratio['Color'] = table[ratio_vio_sel2]
                fig_vio = px.violin(res_ratio[res_ratio['Sel'] == cou_sel], y = "R_1", x = 'Sel', color = 'Color', box = True, points = 'suspectedoutliers', 
                                    title = 'Violin plot for the created ratio', hover_data = ['Un Name'])
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
        feaImp_target = st.sidebar.selectbox("Feature Importance target", col_mul, 1)
        id_sel_col = st.sidebar.selectbox("ID column", table.columns, 2)

        st.header("Feature Importance Analysis")
        
        left, right = st.beta_columns(2)
        with left: 
            fea_Imp_features = st.multiselect("Feature Importance multiselection box:", col_mul)
        with right:
            id_sel = st.selectbox("multivariable index col", ['All ids'] + list(table[id_sel_col].unique()), 0)
        
        scaler = StandardScaler()
        if id_sel == 'All ids':
            target = table[feaImp_target]
            train_nm = table[fea_Imp_features]
        else:
            target = table[table[id_sel_col] == id_sel][feaImp_target]
            train_nm = table[table[id_sel_col] == id_sel][fea_Imp_features]

        for name_col in fea_Imp_features:
            train_nm[name_col].replace({np.nan : train_nm[name_col].mean()}, inplace = True)
        train_nm = scaler.fit_transform(train_nm)

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
        var_clean = table[use_col].dropna().values
        
        # MLE normal
        mu_hat = var_clean.mean()
        sigma_hat = math.sqrt(((var_clean - var_clean.mean()) ** 2).sum() / var_clean.shape[0])

        # MLE exponential
        lambda_hat_exp = var_clean.shape[0] / var_clean.sum()

        # MLE log-normal
        mu_hat_log = (np.ma.log(var_clean)).sum() / var_clean.shape[0]
        sigma_hat_log = math.sqrt(((np.ma.log(var_clean) - mu_hat_log) ** 2).sum() / var_clean.shape[0])
        
        # MLE weibull
        a, alpha_hat, b, beta_hat = stats.exponweib.fit(var_clean, floc=0, fa=1)
        
        # computing the p-values for all the distributions
        result_norm = stats.kstest(var_clean, 'norm', (mu_hat, sigma_hat))
        result_exp = stats.kstest(var_clean, 'expon')
        result_lognorm = stats.kstest(var_clean, 'lognorm', (mu_hat_log, sigma_hat_log))
        result_weibull2 = stats.kstest(var_clean, 'dweibull', (beta_hat, alpha_hat))
        
        # visual part
        dis_fit = [[result_norm[1], result_exp[1], result_lognorm[1], result_weibull2[1]], 
                   [result_norm[1] > 0.05, result_exp[1] > 0.05, result_lognorm[1] > 0.05, result_weibull2[1] > 0.05]]
        st.table(pd.DataFrame(dis_fit, columns = ['Normal', 'Exponential', 'Log-Norm', 'Weibul'], index = ['P-value', 'P > t']))

        ch_distr = st.selectbox("Choose the distribution you want to use for the anomalies estimation", ['Normal', 'Exponential', 'Log-Norm', 'Weibull'])
        fig_distr = go.Figure(data = [go.Histogram(x = var_clean, 
                                                   xbins = dict(start = var_clean.min(), end = var_clean.max(), size = (var_clean.max() - var_clean.min()) / 25),
                                                   autobinx = False, 
                                                   histnorm = 'probability density')])
        
        x_pos = np.linspace(var_clean.min(), var_clean.max(), 25)
        if ch_distr == 'Normal':
            fig_distr.add_trace(go.Scatter(x = x_pos, y = stats.norm(mu_hat, sigma_hat).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        if ch_distr == 'Exponential':
            fig_distr.add_trace(go.Scatter(x = x_pos, y = stats.expon(lambda_hat_exp).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        if ch_distr == 'Log-Norm':
            fig_distr.add_trace(go.Scatter(x = x_pos, y = stats.lognorm(mu_hat_log, sigma_hat_log).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        if ch_distr == 'Weibull':
            fig_distr.add_trace(go.Scatter(x = x_pos, y = stats.dweibull(alpha_hat, beta_hat).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        
        fig_distr.update_layout(title = 'Hist plot to comapre data with possible underlying distribution', xaxis_title = use_col + ' values', yaxis_title = use_col + ' PMF and ch. distr. PDF')
        st.plotly_chart(fig_distr, use_container_width=True)
         
        # outlier part
        st.markdown('In the next part there is the effective detection of the outliers contained into the data. The detection is made by the **Tukey\'s fences**. ' + 
                    'These fences are calculated by refering to the next formulas and the applied formula depend on the type of distribution chosen ' + 
                    'and the **skewness** of the data. \n If the data is **skewed** the fences are calculated in this way: ')
        st.latex(r'''[Q_{1} - (k \cdot ITQ), Q_{3} + (k \cdot ITQ)]''')
        st.markdown('If the data distribution is **not skewed** the formula changes with a correction term and becomes:')
        st.latex(r'''[Q_{1} - (k \cdot e^{-4mc} \cdot ITQ), Q_{3} + (k \cdot e^{3mc} \cdot ITQ)]''')
        st.markdown('If mc > 0 and: ')
        st.latex(r'''[Q_{1} - (k \cdot e^{-3mc} \cdot ITQ), Q_{3} + (k \cdot e^{4mc} \cdot ITQ)]''')
        st.markdown('If mc < 0. \n In these equations $Q_{1}$ represent the first quantile, while $Q_{3}$ represents the third quantile, $ITQ = Q_{3} - Q_{1}$, k is the ' + 
                    '**Tukey\'s constant** and mc is the value of the **MedCouple** function. A value is treated as an outlier if it doesen\'t fit into these intervals. ' + 
                    'In this application we make a distinction between strong and weak outlier. A strong outlier $o_{s}$ is a value that given a $t_{f}$ value for the ' + 
                    'fence\'s correction term: ')
        st.markdown('$o_{s} < Q_{1} - 2 \cdot t_{f}$ if it\'s a left outlier and $o_{s} > Q_{3} + 2 \cdot t_{f}$ if it\'s a right one')
        st.markdown('While a weak outlier $o_{w}$ is defined as: ')
        st.markdown('$o_{w} \in [Q_{1} - 2 \cdot t_{f}, Q_{1} - t_{f}]$ if it\'s a left outlier and $o_{w} \in [Q_{3} + t_{f}, Q_{3} + 2 \cdot t_{f}]$ if it\'s a right one')
        st.markdown('In the next numeric input you can insert the value of the **Tukey\'s constant** (is usually setted to 1,5), from the previous formulas we can derive ' + 
                    'that a smaller **k** will reduce the fence\'s size (you will find more outliers but less significant), while a bigger **k** will have an opposite effect,' + 
                    ' so this value must be chosen wisely.')
        
        tukey_const = st.number_input("Insert the constant for the fence interquantile value", 0.5, 7.5, 1.5)
        Q3 = table[use_col].quantile(0.75); Q1 = table[use_col].quantile(0.25); ITQ = Q3- Q1
        
        if stats.skewtest(var_clean)[1] >= 0.025:
            df_AllOut = table[(table[use_col] <= Q1 - (tukey_const * ITQ)) | (table[use_col] >= Q3 + (tukey_const * ITQ))]
            df_StLeftOut = table[table[use_col] < Q1 - (2 * tukey_const * ITQ)]
            df_WeLeftOut = table[(table[use_col] >= Q1 - (2 * tukey_const * ITQ)) & (table[use_col] <= Q1 - (tukey_const * ITQ))]
            df_WeRightOut = table[(table[use_col] >= Q3 + (tukey_const * ITQ)) & (table[use_col] <= Q3 + (2 * tukey_const * ITQ))]
            df_StRightOut = table[table[use_col] > Q3 + (2 * tukey_const * ITQ)]
            st.table(pd.DataFrame(np.array([df_StLeftOut.shape[0], df_WeLeftOut.shape[0], df_WeRightOut.shape[0], df_StRightOut.shape[0]]).reshape(1, 4),
                                  index = ['Number'], columns = ['Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers']))
        else:
            # calculating the medcouple function for the tukey fence
            if var_clean.shape[0] > 5000:
                MC = np.array([medcouple(var_clean[np.random.choice(var_clean.shape[0], 5000)]) for _ in range(50)]).mean()
            else:
                MC = medcouple(var_clean)
            
            # calculating the tukey fence
            if MC > 0:
                df_AllOut = table[(table[use_col] <= Q1 - (tukey_const * math.exp(-4 * MC) * ITQ)) | (table[use_col] >= Q3 + (tukey_const * math.exp(3 * MC) * ITQ))]
                df_StLeftOut = table[table[use_col] < Q1 - (2 * tukey_const * math.exp(-4 * MC) * ITQ)]
                df_WeLeftOut = table[(table[use_col] >= Q1 - (2 * tukey_const * math.exp(-4 * MC) * ITQ)) & (table[use_col] <= Q1 - (tukey_const * math.exp(-4 * MC) * ITQ))]
                df_WeRightOut = table[(table[use_col] >= Q3 + (tukey_const * math.exp(3 * MC) * ITQ)) & (table[use_col] <= Q3 + (2 * tukey_const * math.exp(3 * MC) * ITQ))]
                df_StRightOut = table[table[use_col] > Q3 + (2 * tukey_const * math.exp(3 * MC) * ITQ)]
                st.table(pd.DataFrame(np.array([df_StLeftOut.shape[0], df_WeLeftOut.shape[0], df_WeRightOut.shape[0], df_StRightOut.shape[0]]).reshape(1, 4),
                                      index = ['Number'], columns = ['Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers']))
            else:
                df_AllOut = table[(table[use_col] <= Q1 - (tukey_const * math.exp(-3 * MC) * ITQ)) | (table[use_col] >= Q3 + (tukey_const * math.exp(4 * MC) * ITQ))]
                df_StLeftOut = table[table[use_col] < Q1 - (2 * tukey_const * math.exp(-3 * MC) * ITQ)]
                df_WeLeftOut = table[(table[use_col] >= Q1 - (2 * tukey_const * math.exp(-3 * MC) * ITQ)) & (table[use_col] <= Q1 - (tukey_const * math.exp(-3 * MC) * ITQ))]
                df_WeRightOut = table[(table[use_col] >= Q3 + (tukey_const * math.exp(4 * MC) * ITQ)) & (table[use_col] <= Q3 + (2 * tukey_const * math.exp(4 * MC) * ITQ))]
                df_StRightOut = table[table[use_col] > Q3 + (2 * tukey_const * math.exp(4 * MC) * ITQ)]
                st.table(pd.DataFrame(np.array([df_StLeftOut.shape[0], df_WeLeftOut.shape[0], df_WeRightOut.shape[0], df_StRightOut.shape[0]]).reshape(1, 4),
                                      index = ['Number'], columns = ['Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers']))
        
        # a more specific view of the ouliers by country or generic id and type
        left, right = st.beta_columns(2)
        with left: 
            out_id_col = st.selectbox("Outlier index col", table.columns, 0)
        with right:
            out_type = st.selectbox("Type of outliers you want to investigate", ['All', 'Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers'], 0)
        
        if out_type == 'All':
            df_AllOut['Sel'] = df_AllOut[out_id_col].str.slice(0, 2).values
            res = {out_id_col: df_AllOut['Sel'].unique(), 'Num. Out.': []}
            for nut_id in res[out_id_col]:
                  res['Num. Out.'].append(df_AllOut[df_AllOut['Sel'] == nut_id].shape[0])
            res = pd.DataFrame(res)

            px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
            map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[out_id_col], featureidkey = 'properties.ISO2',
                                           color = 'Num. Out.', color_continuous_scale = px.colors.cyclical.IceFire,
                                           range_color = (res['Num. Out.'].min(), res['Num. Out.'].max()),
                                           mapbox_style = "carto-positron",
                                           zoom = 3, center = {"lat": 47.42, "lon": 15.53},
                                           opacity = 0.5,
                                           labels = {'Num. Out.': 'Num. Out.'})
            st.plotly_chart(map_box, use_container_width=True)
         
        if out_type == 'Strong left outliers':
            df_StLeftOut['Sel'] = df_StLeftOut[out_id_col].str.slice(0, 2).values
            res = {out_id_col: df_StLeftOut['Sel'].unique(), 'Num. Out.': []}
            for nut_id in res[out_id_col]:
                  res['Num. Out.'].append(df_StLeftOut[df_StLeftOut['Sel'] == nut_id].shape[0])
            res = pd.DataFrame(res)

            px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
            map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[out_id_col], featureidkey = 'properties.ISO2',
                                           color = 'Num. Out.', color_continuous_scale = px.colors.cyclical.IceFire,
                                           range_color = (res['Num. Out.'].min(), res['Num. Out.'].max()),
                                           mapbox_style = "carto-positron",
                                           zoom = 3, center = {"lat": 47.42, "lon": 15.53},
                                           opacity = 0.5,
                                           labels = {'Num. Out.': 'Num. Out.'})
            st.plotly_chart(map_box, use_container_width=True)
            
        if out_type == 'Weak left outliers':
            df_WeLeftOut['Sel'] = df_WeLeftOut[out_id_col].str.slice(0, 2).values
            res = {out_id_col: df_WeLeftOut['Sel'].unique(), 'Num. Out.': []}
            for nut_id in res[out_id_col]:
                  res['Num. Out.'].append(df_WeLeftOut[df_WeLeftOut['Sel'] == nut_id].shape[0])
            res = pd.DataFrame(res)

            px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
            map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[out_id_col], featureidkey = 'properties.ISO2',
                                           color = 'Num. Out.', color_continuous_scale = px.colors.cyclical.IceFire,
                                           range_color = (res['Num. Out.'].min(), res['Num. Out.'].max()),
                                           mapbox_style = "carto-positron",
                                           zoom = 3, center = {"lat": 47.42, "lon": 15.53},
                                           opacity = 0.5,
                                           labels = {'Num. Out.': 'Num. Out.'})
            st.plotly_chart(map_box, use_container_width=True)
            
        if out_type == 'Weak right outliers':
            df_WeRightOut['Sel'] = df_WeRightOut[out_id_col].str.slice(0, 2).values
            res = {out_id_col: df_WeRightOut['Sel'].unique(), 'Num. Out.': []}
            for nut_id in res[out_id_col]:
                  res['Num. Out.'].append(df_WeRightOut[df_WeRightOut['Sel'] == nut_id].shape[0])
            res = pd.DataFrame(res)

            px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
            map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[out_id_col], featureidkey = 'properties.ISO2',
                                           color = 'Num. Out.', color_continuous_scale = px.colors.cyclical.IceFire,
                                           range_color = (res['Num. Out.'].min(), res['Num. Out.'].max()),
                                           mapbox_style = "carto-positron",
                                           zoom = 3, center = {"lat": 47.42, "lon": 15.53},
                                           opacity = 0.5,
                                           labels = {'Num. Out.': 'Num. Out.'})
            st.plotly_chart(map_box, use_container_width=True)
            
        if out_type == 'Strong right outliers':
            df_StRightOut['Sel'] = df_StRightOut[out_id_col].str.slice(0, 2).values
            res = {out_id_col: df_StRightOut['Sel'].unique(), 'Num. Out.': []}
            for nut_id in res[out_id_col]:
                  res['Num. Out.'].append(df_StRightOut[df_StRightOut['Sel'] == nut_id].shape[0])
            res = pd.DataFrame(res)

            px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
            map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[out_id_col], featureidkey = 'properties.ISO2',
                                           color = 'Num. Out.', color_continuous_scale = px.colors.cyclical.IceFire,
                                           range_color = (res['Num. Out.'].min(), res['Num. Out.'].max()),
                                           mapbox_style = "carto-positron",
                                           zoom = 3, center = {"lat": 47.42, "lon": 15.53},
                                           opacity = 0.5,
                                           labels = {'Num. Out.': 'Num. Out.'})
            st.plotly_chart(map_box, use_container_width=True)
        
        out_cou = st.selectbox("Choose the specific value for the id", ['All ids'] + list(res[out_id_col]), 0)
        
        if out_cou == 'All ids': 
            st.write(df_AllOut)
        else:
            st.write(df_AllOut[df_AllOut[out_id_col] == out_cou])
        
    if widget == "Consistency checks":
        con_checks_id_col = st.sidebar.selectbox("Index col", table.columns, 0)
        con_checks_time_col = st.sidebar.selectbox("Time column", table.columns, 0)
        
        con_checks_features = st.multiselect("Feature Importance multiselection box:", col_mul)
        
        res = dict()
        for id_inst in table[con_checks_id_col].unique():
            inst = table[table[con_checks_id_col] == id_inst][con_checks_features]; list_par = list()
            
            for var in con_checks_features:
                years = 0; res_par = 1
                for i in range(inst.shape[0]):
                    if not np.isnan(inst[var].iloc[i]):
                        res_par *= inst[var].iloc[i]; years += 1 
                if years != 0:
                    list_par.append(math.pow(math.fabs(res_par), 1/years))
                else:
                    list_par.append(np.nan)
            res[id_inst] = list_par
        
        indices = pd.DataFrame(res.values(), index = res.keys(), columns = con_checks_features)
        
        list_threshold = list()
        for col in con_checks_features:
            list_threshold.append(indices[col].quantile(0.075))
            
        el_row = list()
        for row in indices.index.values:
            for j in range(len(con_checks_features)):
                if not np.isnan(indices[con_checks_features[j]][row]) and indices[con_checks_features[j]][row] < list_threshold[j]:
                    if row not in el_row:
                        el_row.append(row)

        indices.drop(index = el_row, axis = 0, inplace = True)
         
        res = dict()
        # does the calculation with the delta+ and delta-minus for the multiannual checks and stores it into a dictionary 
        for id_inst in indices.index.values:
            inst = table[table[con_checks_id_col] == id_inst]; delta_pos = list(); delta_neg = list(); num_row = inst.shape[0]
            for var in con_checks_features:
                for i in range(1, num_row):
                    if not np.isnan(inst[var].iloc[num_row - i]) and not np.isnan(inst[var].iloc[num_row - i - 1]):
                        if inst[var].iloc[num_row - i - 1] - inst[var].iloc[num_row - i] < 0:
                            delta_neg.append(inst[var].iloc[num_row - i - 1] - inst[var].iloc[num_row - i])
                        else:
                            delta_pos.append(inst[var].iloc[num_row - i - 1] - inst[var].iloc[num_row - i])
                lis_app = list(); lis_app.append(delta_pos); lis_app.append(delta_neg)
                res[id_inst + "." + var] = lis_app
                delta_pos = list(); delta_neg = list()
         
        DV = dict() # the dictionary in wich we'll store all the DV and further the DM values for the variability from years
        for key, value in res.items():
            res_par = 0
            if len(value[0]) != 0 or len(value[1]) != 0:
                res_par = sum(value[0]) * sum(value[1])
            if not np.isnan(indices[key[key.find('.')+1:]][key[:key.find('.')]]) and indices[key[key.find('.')+1:]][key[:key.find('.')]] != 0:
                DV[key] = round(math.fabs(res_par)/indices[key[key.find('.')+1:]][key[:key.find('.')]], 3)
        
        dict_app = dict()
        for key, value in DV.items():
            if key[key.find('.')+1:] not in dict_app.keys():
                dict_app[key[key.find('.')+1:]] = [value]
            else:
                dict_app[key[key.find('.')+1:]].append(value)
        
        list_threshold = list()
        for key, value in dict_app.items():
            np_value = np.array(value); list_threshold.append(np.quantile(np_value, 0.95))
        
        cont = 0; dict_flag = dict()
        for key, value in dict_app.items():
            list_app = [[], []]
            for el in value:
                if el > list_threshold[cont]:
                    if el not in list_app[0]:
                        list_app[0].append(el); list_app[1].append(1)
                    else:
                        list_app[1][list_app[0].index(el)] += 1
            dict_flag[key] = list_app; cont += 1
        
        var_flag = list()
        for key, value in dict_flag.items():
            for i in range(len(value[0])):
                cont = 0
                while cont != value[1][i]:
                    for key_DV, value_DV in DV.items():
                        if key_DV[key_DV.find('.')+1:] == key and value_DV == value[0][i]:
                            if key_DV[:key_DV.find('.')] not in var_flag:
                                var_flag.append(key_DV)
                    cont += 1
        
        list_countries = []
        for inst in var_flag:
            if inst[:2] not in list_countries:
                list_countries.append(inst[:2])
        DV_fin_res = np.zeros((len(con_checks_features), len(list_countries)), dtype = int)
        
        for flag in var_flag:
            DV_fin_res[con_checks_features.find(flag[flag.find('.')+1:]), list_countries.find(flag[:2])] += 1
        st.write(DV_fin_res)
        DV_fin_tab = pd.DataFrame(DV_fin_res, index = con_checks_features, columns = list_countries)
