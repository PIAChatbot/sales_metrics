import streamlit as st

import pandas as pd
import numpy as np
import os
    
import glob
from pyxlsb import open_workbook as open_xlsb
from io import BytesIO
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#______________________________________________________________MISE EN PAGE CSS_________________________________________________________________

st.set_page_config(page_title="Sales Metrics",page_icon = r'./images/Orange_favicon.png', layout="wide")

st.markdown(f'<p style="color:#FF7900;font-size:100px;">{"Base CA"}</p>', unsafe_allow_html=True)

df_baseCA = pd.read_csv(r'./csv_output/base_ca_anonymise.csv', sep=";",encoding='utf-8')

df_baseCA.Vendeur= df_baseCA.Vendeur.str.upper()
columns = ['Vendeur']+df_baseCA.columns[-6:].tolist()
df1 = df_baseCA[columns]
df1_1=df1.groupby('Vendeur').sum()
df1_1['Total'] = df1_1.sum(axis=1)
df1_1=df1_1.reset_index()

fig1 = px.pie(df1_1, values='Total', names='Vendeur')
fig1.update_layout(
    title={'text': "Répartition des ventes du dernier semestre",
           'font_color':'#FF7900',
           'font_size': 35},
    showlegend=True,  # Afficher une seule légende
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
    height=700)

st.plotly_chart(fig1)

st.markdown('_________')

fig = make_subplots(
    rows=1, cols=6, 
    subplot_titles=df_baseCA.columns[-6:].tolist(),
    specs=[[{"type": "domain"}] * 6],  # Chaque subplot est un camembert
)

# Ajouter les 6 pie charts
for i, col in enumerate(df_baseCA.columns[-6:], start=1):
    fig.add_trace(go.Pie(labels=df1["Vendeur"], values=df1[col], name=col), row=1, col=i)

# Mettre une légende unique
fig.update_layout(
    title_text="Détail mensuel",
    showlegend=True,  # Afficher une seule légende
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center")
)
st.plotly_chart(fig)






# tab1, tab2, tab3, tab4, tab18, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab19, tab20= st.tabs(["Year Over Year",
 # "Integrations/quarter",
 # "Validations/quarter",
 # "TOP Countries/Quarter",
 # "TOP Countries/Year",
 # "Translator tickets",
 # "Weekly view",
 # "TOP 3 Integration",
 # "Backlog evolution",
 # "Time spent",
 # "Top integrated devices",
 # "Top integrated modality",
 # "Top integration mode integrated",
 # "Verifications",
 # "New devices integrated",
 # "Validation Issues",
 # "New sites installs",
 # "DAK support",
 # "Number of IF tickets validated per year",
 # "New devices integrations over the years"
# ])
 

