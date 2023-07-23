##!/usr/bin/python -u
#-*- coding:utf-8 -*-
#
# Streamlit dashboard for visualizing the impact of various geometric and operational parameters onto 
# the performance of a product in a webbrowser dashboard
#
# Astrid Walle - Astrid Walle CFDsolutions
# astridwalle@cfdsolutions.net
#---------------------------------------------------------------------------------------------------
#
# 2021-01-28	Script creation
VERSION="2021-01-28"
#---------------------------------------------------------------------------------------------------
#
### EXECUTION
# streamlit run StreamlitDashboard.py
#
#---------------------------------------------------------------------------------------------------
#
### SETTINGS
# Is the Dashboard supposed to run locally (local=1) or online (local=0)?
# TODO Make this setting as an commandline option to start the dashboard!
local=1
# streamlit run your_script.py [-- script args]
#---------------------------------------------------------------------------------------------------
#
### IMPORTS
import streamlit as st
import pandas as pd
import os
import hiplot as hip
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
#import cv2
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import s3fs
#from scipy.integrate import odeint, simps
#from sqlalchemy import create_engine
#import json
#


# S3 access
s3 = s3fs.S3FileSystem(anon=False)

#---------------------------------------------------------------------------------------------------
#
# FUNCTIONS / DATA PROCESSING

# read tables
#@st.cache
def read_table(file,header=0,index_col=False,sep=" "):
    df=pd.read_csv(file,header=header,index_col=index_col,sep=sep)
    return df

# feature_importance
@st.cache
def feature_importance(df,params,objective):
    X=np.array(df[params])
    y=np.array(df[objective])
    #model=LinearRegression()
    model=DecisionTreeRegressor()
    model.fit(X,y)
    #importances=model.coef_
    importances=model.feature_importances_
    return importances

# read png plots
@st.cache(hash_funcs={dict: lambda _: None})
def cache_png_plots(names,df,path):
    dict_name_={}
    for name in names:
        if not df[name].empty:
            dict_idx_={}  
            for i in range(len(df)):
                filename=df.loc[i,name]
                if filename.startswith("/"):
                    with open(filename, 'rb') as handle:
                        img = handle.read()
                        dict_idx_[i]=img
                else:
                    #with open(os.path.join(path,filename), 'rb') as handle:
                    url = os.path.join("s3://cfdsolutions-public/Data/",filename)
                    with s3.open(url, 'rb') as handle:    
                        img = handle.read()
                        dict_idx_[i]=img
            dict_name_[name]=dict_idx_
        else: 
            dict_name_[name]={}
    return dict_name_

# read all pickled 3D plots once into a dict
@st.cache(hash_funcs={dict: lambda _: None})
def cache_pkl_plots(names,df,path):
    dict_name_={}
    for name in names:
        if not df[name].empty:
            dict_idx_={}
            for i in range(len(df)):
                filename=df.loc[i,name]
                if filename.startswith("/"):
                    with open(filename, 'rb') as handle:
                        fig_pickled = pickle.load(handle)
                        dict_idx_[i]=fig_pickled    
                else:
                    #with open(os.path.join(path,filename), 'rb') as handle:
                    url = os.path.join("s3://cfdsolutions-public/Data/",filename)
                    with s3.open(url, 'rb') as handle: 
                        fig_pickled = pickle.load(handle)
                        dict_idx_[i]=fig_pickled                      
            dict_name_[name]=dict_idx_
        else: 
            dict_name_[name]={}
    return dict_name_

@st.cache(hash_funcs={dict: lambda _: None})
def cache_pkl_plot(path,file):
    #with open(os.path.join(path,file), 'rb') as handle:
    url = os.path.join("s3://cfdsolutions-public/Data/",file)
    with s3.open(url, 'rb') as handle:       
        fig_pickled = pickle.load(handle)

    return fig_pickled


#---------------------------------------------------------------------------------------------------
# Set-Up
#---------------------------------------------------------------------------------------------------

st.set_page_config(layout="wide")

st.markdown("""
<style>
.reportview-container .main footer, .reportview-container .main footer a {
    color: #0c0080;
}
</style>
    """, unsafe_allow_html=True)

st.title("CFD INSIGHTS")
st.write("This dashboard allows the visualization of multiple designs to gain a deeper understanding of the single parameter's impact.")
st. write("The designs and CFD results can be generated in a design space exploration or an optimization.")

st.markdown("---")

#---------------------------------------------------------------------------------------------------
# Side Bar Read-In Data
#---------------------------------------------------------------------------------------------------

path = "https://cfdsolutions-public.s3.eu-central-1.amazonaws.com/Data/"

df_params = read_table("https://cfdsolutions-public.s3.eu-central-1.amazonaws.com/Data/parameters.csv",sep=",")
df_res = read_table("https://cfdsolutions-public.s3.eu-central-1.amazonaws.com/Data/results.csv", sep=",")
df_opti = read_table("https://cfdsolutions-public.s3.eu-central-1.amazonaws.com/Data/opti.csv", sep=",")

parameter_names=list(df_params["Parameter"])
result_names=list(df_res["Result"])

contour2D_names=[i for i in df_opti.columns.values if "2D" in i]
contour3D_names=[i for i in df_opti.columns.values if "3D" in i]

df_plots=df_opti[contour2D_names + contour3D_names]

# Test for hardcoded Image Recognition
df_colorframe=list(df_opti["Separation Prob. color"])
df_prob=list(df_opti["Separation Prob."])

#df_opti=df_opti.drop(labels=(contour2D_names + contour3D_names),axis=1)
df_opti=df_opti[["Design"]+result_names+parameter_names]


#---------------------------------------------------------------------------------------------------
# Side Bar Parameter Exploration
#---------------------------------------------------------------------------------------------------

st.sidebar.write(" ")

# sweep through slider:
# https://discuss.streamlit.io/t/animate-st-slider/1441/7

# TODO: Checkbox: select specific value or a range? in sidebar!
# TODO: Make sidebar larger! --> Refactor code to use containers and columns!
# TODO: use st.cache to readin the files just once!
# TODO: Read the data and pictures directly from S3 bucket!
# TODO: Where to run the script pipeline to execute the transformation from plt to plotly? --> Lambda Function!
# TODO: Create git repo and create codeflow in AWS to commit to repo and rebuild the homepage!



parameter_names_sel=parameter_names.copy()
df_updated=df_opti.copy()
parameter_values={}

exp_explore=st.sidebar.expander(label="Parameter Space Exploration")
for name in parameter_names:
    slider_ph = exp_explore.empty()
    options=sorted(list(df_updated[name].unique()))
    parameter_values[name] = slider_ph.select_slider(label=name, options=options,value=[min(options),max(options)])#,key=name)

    df_updated=df_updated[((df_updated[name]>=parameter_values[name][0]) & (df_updated[name]<=parameter_values[name][1]))]
    
# generate the dataframe with the data subset for the selected parameter values
df_opti_sub=df_updated

#---------------------------------------------------------------------------------------------------
# Main Page
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# View Setup
#---------------------------------------------------------------------------------------------------

if st.checkbox("Show Geometry Parameter Space"):
    #st.write("**CFD Setup**")
    try:
        st.image(os.path.join(path,"geometry.gif"))
    except:
        pass
    
    st.markdown("---")

#---------------------------------------------------------------------------------------------------
# Raw data
#---------------------------------------------------------------------------------------------------
if st.checkbox("Show data table"):
    #st.write("**Raw Data Table**")
    st.write(df_opti_sub)

    st.markdown("---")
#
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# XY Plots / 3 columns
# add 2nd scatter with red color for parameter_values[name]
#---------------------------------------------------------------------------------------------------
if st.checkbox(label="Display XY plots",key="XY"):
    st.write("**XY Scatter Plots**")
    st.write("Select X and Y values from the parameters / result to display")

    # create 3 plots next to each other
    col11, col12 = st.columns(2)

    # PLot 1: select x and y values:
    paramx1=col11.selectbox("Select parameter for x-axis",options=df_opti.columns.values,key="paramx1")
    paramy1=col11.selectbox("Select parameter for y-axis",options=df_opti.columns.values,key="paramy1")
    #paramsize1=col1.selectbox("Select parameter for scatter size",options=df_opti.columns.values)
    #paramcolor1=col1.selectbox("Select parameter for scatter color",options=df_opti.columns.values)

    trace11=go.Scatter(x=df_opti[paramx1],y=df_opti[paramy1],name="full dataset",mode="markers",marker=dict(color="blue",size=10))
    fig1 = go.Figure(data=trace11)

    if not df_opti_sub.empty:
        trace21=go.Scatter(mode="markers", x=df_opti_sub[paramx1],y=df_opti_sub[paramy1],marker=dict(color="red",size=15),name="selection")
        fig1.add_trace(trace21)

    col11.plotly_chart(fig1,use_container_width=True)

    # PLot 2: select x and y values:
    paramx2=col12.selectbox("Select parameter for x-axis",options=df_opti.columns.values,key="paramx2")
    paramy2=col12.selectbox("Select parameter for y-axis",options=df_opti.columns.values,key="paramy2")
    #paramsize1=col1.selectbox("Select parameter for scatter size",options=df_opti.columns.values)
    #paramcolor1=col1.selectbox("Select parameter for scatter color",options=df_opti.columns.values)

    trace12=go.Scatter(x=df_opti[paramx2],y=df_opti[paramy2],name="full dataset",mode="markers",marker=dict(color="blue",size=10))
    fig2 = go.Figure(data=trace12)

    if not df_opti_sub.empty:
        trace22=go.Scatter(mode="markers", x=df_opti_sub[paramx2],y=df_opti_sub[paramy2],marker=dict(color="red",size=15),name="selection")
        fig2.add_trace(trace22)

    col12.plotly_chart(fig2,use_container_width=True)

    st.markdown("---")

#---------------------------------------------------------------------------------------------------
# 2D Plots
#---------------------------------------------------------------------------------------------------
if st.checkbox(label="Display 2D plots",key="2D"):
    # read all 2D plots
    if contour2D_names:
        dict_2D=cache_png_plots(contour2D_names,df_plots,path)

    st.write("**2D Contour Plots**")
    st.write("Select the type of 2D contour plot you want to get displayed for the selection")

    # Add additional functionality for interactive model training and inference
    # button_sep = st.radio("Looking for separation?",("No","Yes - Training Mode","Yes - Assistant Mode - Show Separation Prob."),key="radio_sep")

    # make the same indices for df_opti_sub and df_plots
    idx=df_opti_sub.index.values
    contour2D=st.selectbox("Select the 2D contour plot type",options=contour2D_names)

    cols=3
    col21,col22,col23=st.columns(cols)

    no_of_plots=len(idx)
    no_of_rows=int(no_of_plots/cols)+1
    for i in range(0,no_of_rows):
        #for j in range(0,cols):
        #index=idx[i*4+j]
        try:
            img=dict_2D[contour2D][idx[i*4+0]]
            # if button_sep == "Yes - Assistant Mode - Show Separation Prob.":
            #     color=df_colorframe[idx[i*4+0]]
            #     string=f'<p style="background-color:{color};color:#17202A;font-size:30px;border-radius:5%;">{"Design "+str(df_opti_sub["Design"][idx[i*4+0]])+" "+str(df_prob[idx[i*4+0]])+"%"}</p>'
            #     col21.markdown(string, unsafe_allow_html=True)
            # else:
            col21.write("Design "+str(df_opti_sub["Design"][idx[i*4+0]]))
            col21.image(img,use_column_width=True)
        except: pass
        try:
            img=dict_2D[contour2D][idx[i*4+1]]
            # if button_sep == "Yes - Assistant Mode - Show Separation Prob.":
            #     color=df_colorframe[idx[i*4+1]]
            #     string=f'<p style="background-color:{color};color:#17202A;font-size:30px;border-radius:5%;">{"Design "+str(df_opti_sub["Design"][idx[i*4+1]])+" "+str(df_prob[idx[i*4+0]])+"%"}</p>'
            #     col22.markdown(string, unsafe_allow_html=True)
            # else:
            col22.write("Design "+str(df_opti_sub["Design"][idx[i*4+1]]))
            col22.image(img,use_column_width=True)
        except: pass
        try:
            img=dict_2D[contour2D][idx[i*4+2]]
            # if button_sep == "Yes - Assistant Mode - Show Separation Prob.":
            #     color=df_colorframe[idx[i*4+2]]
            #     string=f'<p style="background-color:{color};color:#17202A;font-size:30px;border-radius:5%;">{"Design "+str(df_opti_sub["Design"][idx[i*4+2]])+" "+str(df_prob[idx[i*4+0]])+"%"}</p>'
            #     col23.markdown(string, unsafe_allow_html=True)
            # else:
            col23.write("Design "+str(df_opti_sub["Design"][idx[i*4+2]]))
            col23.image(img,use_column_width=True)
        except: pass
        # try:
        #     col24.image(df_plots.loc[idx[i*4+3],contour2D],use_column_width=True)   
        # except: pass

    st.markdown("---")

#---------------------------------------------------------------------------------------------------
# Hiplot
#---------------------------------------------------------------------------------------------------
if st.checkbox(label="Display HiPlot",key="HiPlot_check"):
    st.write("**HiPlot (Multiple parallel axes)**")

    # transform df into dict:
    dict_opti=df_opti.to_dict(orient="records")

    xp = hip.Experiment.from_iterable(dict_opti)
    xp.display_data(hip.Displays.PARALLEL_PLOT).update({
        "hide":["uid"],
    })
    xp.display_data(hip.Displays.XY).update({
            # Default X axis for the XY plot
            'axis_x': 'Design',
            # Default Y axis
            'axis_y': 'Power Number',
            # Configure lines
            'lines_thickness': 2.0,
            'lines_opacity': 0.9,
            # Configure dots
            'dots_thickness': 5.0,
            'dots_opacity': 1.0,
    })
    ret = xp.display_st(ret="filtered_uids",key="HiPlot")

    st.markdown("---")

#---------------------------------------------------------------------------------------------------
# Feature Importance
#---------------------------------------------------------------------------------------------------

if st.checkbox("Feature Importance"):
    objective=st.selectbox(label="Please select your objective function", options=result_names,index=0,key="opti")
    importances=feature_importance(df_opti,parameter_names,objective)

#for i,v in enumerate(importance):
#    st.write("Feature %0d, Score %.5f " %(i,v))

    df_bar=pd.DataFrame(data=np.column_stack([parameter_names,importances.T.tolist()]), columns=["feature","importance"])
    #st.write(df_bar)
    fig_bar = px.bar(df_bar,x="feature", y="importance")
    st.plotly_chart(fig_bar,use_column_width=True)

    st.markdown("---")

#---------------------------------------------------------------------------------------------------
# 3D Plots
#---------------------------------------------------------------------------------------------------
# read all 3D plots
if st.checkbox(label="Display 3D plot",key="3D"):
    try:
        fig=cache_pkl_plot(path,"Lambda2.pickle")
        st.plotly_chart(fig, use_container_width=True)
    except:
        pass

    #st.write("**3D plots currently disabled --> please contact astridwalle@cfdsolutions.net to get a demo of interactive 3D plots**")
    # if contour3D_names:
    #     dict_3D=cache_pkl_plots(contour3D_names,df_plots,path)

    # st.write("**3D Contour Plots**")
    # st.write("Select the type of 3D contour plot you want to get displayed for the selection")

    # idx=df_opti_sub.index.values

    # contour3D=st.selectbox("Select the 3D contour plot type",options=contour3D_names)

    # cols=3
    # col31,col32,col33=st.columns(cols)

    # no_of_plots=len(idx)
    # no_of_rows=int(no_of_plots/cols)+1
    # for i in range(0,no_of_rows):
    #     try:
    #         fig=dict_3D[contour3D][idx[i*4+0]]
    #         col31.plotly_chart(fig, use_container_width=True)
    #         col31.write("Design "+str(df_opti_sub["Design"][idx[i*4+0]]))
    #     except: pass
    #     try:
    #         fig=dict_3D[contour3D][idx[i*4+1]]
    #         col32.plotly_chart(fig, use_container_width=True)
    #         col32.write("Design "+str(df_opti_sub["Design"][idx[i*4+1]]))
    #     except: pass
    #     try:
    #         fig=dict_3D[contour3D][idx[i*4+2]]
    #         col33.plotly_chart(fig, use_container_width=True)
    #         col33.write("Design "+str(df_opti_sub["Design"][idx[i*4+2]]))
    #     except: pass

    # st.markdown("---")


#---------------------------------------------------------------------------------------------------
# Footer
#---------------------------------------------------------------------------------------------------


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.cfdsolutions.net/" target="_blank">Astrid Walle CFDsolutions</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

