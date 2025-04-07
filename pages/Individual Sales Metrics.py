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

st.set_page_config(page_title="vendeur",page_icon = r'./images/Orange_favicon.png', layout="wide")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f'<p style="color:#FF7900;font-size:100px;">{"Dashboard"}</p>', unsafe_allow_html=True)

with col2:
    
    st.markdown("""
        <style>
        div[data-baseweb="select"] {
            width: 300px !important; /* Largeur */
            height: 50px !important; /* Hauteur */
         --   background-color: #FF7900 !important; /* Background color */
            border-radius: 5px !important;
        }
        div[data-baseweb="select"] > div {
            font-size: 18px !important; /* Text size */
            color: #FF7900 !important; /* Text color */
        }
        </style>
    """, unsafe_allow_html=True)

    df_baseCA = pd.read_csv(r'./csv_output/base_ca_anonymise.csv', sep=";",encoding='utf-8')

    df_baseCA.Vendeur= df_baseCA.Vendeur.str.upper()
    selectbox_vendeur = st.selectbox(" ", df_baseCA.Vendeur.unique(), key="global_1")  # Selectbox sans label

tab1, tab2, tab3, tab4= st.tabs(["CA Général",
 "Top Client par CA",
 "Top Client par Domaine",
 "Balance CA",
])

df_baseCA_vendeur=df_baseCA[df_baseCA['Vendeur']==selectbox_vendeur]
df_baseCA_vendeur_grp=df_baseCA_vendeur.groupby('Domaine')[df_baseCA.columns[-6:]].sum()

df_baseCA_vendeur_grp['Total'] = df_baseCA_vendeur_grp.sum(axis=1)
df_baseCA_vendeur_grp["Percentage"] = (df_baseCA_vendeur_grp["Total"] / df_baseCA_vendeur_grp["Total"].sum()) * 100
df_baseCA_vendeur_grp=df_baseCA_vendeur_grp.sort_values('Total',ascending=False).reset_index()


fig1 = px.bar(df_baseCA_vendeur_grp, 
              x="Domaine", 
              y="Total")
              
fig1.update_traces(marker_color='white')
              
fig1.add_trace(go.Scatter(x=df_baseCA_vendeur_grp["Domaine"],
                            y=df_baseCA_vendeur_grp['Total'],
                            marker=dict(color='#FF7900', size=20),
                            textposition='top center',
                            textfont_size = 20,
                            mode='markers+text',
                            text=df_baseCA_vendeur_grp["Percentage"].map(lambda x: f'{x:.1f}%'),
                            name='Percentage CA'))

# fig1.update_layout(title={'text': "Répartition du CA par domaine pour le dernier semestre",
#            'font_color':"white",
#            'font_size': 35},
#     font=dict(color='#FF7900'),
#     xaxis=dict(title='Domaine',
#         titlefont=dict(color='#FF7900'),
#         showgrid=False),
#     yaxis= dict(title='Total CA',
#         titlefont=dict(color='#FF7900'),
#         showgrid=True),
#     legend=dict(\
#         orientation="h",
#         yanchor="bottom",
#         y=-0.12,
#         xanchor="left",
#         x=0),
#     legend_title_text='',
#     hovermode="x",
#     height=900,
#     margin=dict(l=70, r=70, t=100, b=70))
    
tab1.plotly_chart(fig1)

tab1.markdown('_________')

fig = make_subplots(
    rows=1, cols=6, 
    subplot_titles=df_baseCA.columns[-6:].tolist(),
    specs=[[{"type": "domain"}] * 6],  # Chaque subplot est un camembert
)

# Ajouter les 6 pie charts
for i, col in enumerate(df_baseCA.columns[-6:], start=1):
    fig.add_trace(go.Pie(labels=df_baseCA_vendeur["Domaine"], values=df_baseCA_vendeur[col], name=col), row=1, col=i)

# Mettre une légende unique
fig.update_layout(
    title_text="Détail mensuel",
    showlegend=True,  # Afficher une seule légende
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center")
)
tab1.plotly_chart(fig)


#____________________________________________________________________

selectbox2 = tab2.selectbox('Top client', ['10','20','30','All'],key = "2_1")


df_baseCA_vendeur_grp=df_baseCA_vendeur.groupby('Nom Groupe')[df_baseCA.columns[-6:]].sum()

df_baseCA_vendeur_grp['Total'] = df_baseCA_vendeur_grp.sum(axis=1)
df_baseCA_vendeur_grp["Percentage"] = (df_baseCA_vendeur_grp["Total"] / df_baseCA_vendeur_grp["Total"].sum()) * 100
if selectbox2 == "All":
    df_baseCA_vendeur_grp=df_baseCA_vendeur_grp.sort_values('Total',ascending=False).reset_index()
else :
    df_baseCA_vendeur_grp=df_baseCA_vendeur_grp.sort_values('Total',ascending=False).reset_index().head(int(selectbox2))


fig2 = px.bar(df_baseCA_vendeur_grp, 
              x="Nom Groupe", 
              y="Total")
              
fig2.update_traces(marker_color='white')
              
fig2.add_trace(go.Scatter(x=df_baseCA_vendeur_grp["Nom Groupe"],
                            y=df_baseCA_vendeur_grp['Total'],
                            marker=dict(color='#FF7900', size=20),
                            textposition='top center',
                            textfont_size = 20,
                            mode='markers+text',
                            text=df_baseCA_vendeur_grp["Percentage"].map(lambda x: f'{x:.1f}%'),
                            name='Percentage CA'))

# fig2.update_layout(title={'text': "Top "+selectbox2+" Clients par CA pour le dernier semestre",
#            'font_color':"white",
#            'font_size': 35},
#     font=dict(color='#FF7900'),
#     xaxis=dict(title='Nom Groupe',
#         titlefont=dict(color='#FF7900'),
#         showgrid=False),
#     yaxis= dict(title='Total CA',
#         titlefont=dict(color='#FF7900'),
#         showgrid=True),
#     legend=dict(\
#         orientation="h",
#         yanchor="bottom",
#         y=-0.12,
#         xanchor="left",
#         x=0),
#     legend_title_text='',
#     hovermode="x",
#     height=900,
#     margin=dict(l=70, r=70, t=100, b=70))
    
tab2.plotly_chart(fig2)

#_____________

fig2 = px.pie(df_baseCA_vendeur_grp, values='Total', names='Nom Groupe')
fig2.update_layout(showlegend=True,  # Afficher une seule légende
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
    height=700)

tab2.plotly_chart(fig2)

#____________________________________________________________________________________

import math

selectbox3 = tab3.selectbox('Top client', ['10', '20', '30', 'All'], key="3_1")

tab3.header("Top "+selectbox2+" Clients par Domaine en fonction du CA pour le dernier semestre")


domaines = df_baseCA_vendeur["Domaine"].unique()

for idx, domaine in enumerate(domaines, start=0):

    df_filtered = df_baseCA_vendeur[df_baseCA_vendeur["Domaine"] == domaine]
    df_grouped = df_filtered.groupby("Nom Groupe")[df_baseCA.columns[-6:]].sum()

    df_grouped["Total"] = df_grouped.sum(axis=1)
    df_grouped = df_grouped.sort_values("Total", ascending=False).reset_index()
    
    # Gérer le cas "All"
    top_n = int(selectbox3) if selectbox3 != "All" else len(df_grouped)
    df_grouped = df_grouped.head(top_n)

    df_grouped["Percentage"] = (df_grouped["Total"] / df_grouped["Total"].sum()) * 100

    fig = px.pie(df_grouped, values='Total', names='Nom Groupe')
    fig.update_layout(
        title={'text': domaine,
               'font_color':'#FF7900',
               'font_size': 35},
        showlegend=True,  # Afficher une seule légende
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
        height=700)

    tab3.plotly_chart(fig)


domaines=df_baseCA_vendeur.Domaine.unique()

df_global = []

for domaine in domaines:
    df_filtered = df_baseCA_vendeur[df_baseCA_vendeur["Domaine"] == domaine]
    df_grouped = df_filtered.groupby("Nom Groupe")[df_baseCA.columns[-6:]].sum()
    df_grouped["Total"] = df_grouped.sum(axis=1)
    df_grouped["Percentage"] = (df_grouped["Total"] / df_grouped["Total"].sum()) * 100
    df_grouped = df_grouped.sort_values("Total", ascending=False).reset_index().head(int(selectbox3))
    df_grouped["Domaine"] = domaine
    df_global.append(df_grouped)

df_global = pd.concat(df_global, ignore_index=True)

df_global['text'] = df_global['Total'].astype(str)+'<br>'+df_global['Nom Groupe']



fig3 = px.bar(df_global, 
              x="Domaine", 
              y="Total",
              color="Nom Groupe",
              text='text',
              barmode='stack',
              color_continuous_scale=px.colors.sequential.Darkmint)

fig3.update_layout(xaxis=dict(showgrid=False),
                    yaxis= dict(showgrid=True),
                    height=2000,
                    margin=dict(l=70, r=70, t=100, b=70),
                    showlegend=False,
                    legend_traceorder="reversed")
    
tab3.plotly_chart(fig3)

# _____________

import unicodedata
import pandas as pd
import re
from rapidfuzz import process, fuzz

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.upper()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')\
                                                                        .replace('-',' ')\
                                                                        # .replace(' ','')
    return text
    
def find_best_match(text, keywords):
    match, score, _ = process.extractOne(text, keywords, scorer=fuzz.partial_ratio)
    return match if score > 99.99    else None

def filter_rows_by_keywords(df_main, df_keywords, col1, col2, pop_col):
    keywords = df_keywords[col2].tolist()
    df_main['detected_keyword'] = df_main[col1].apply(lambda x: find_best_match(x, keywords))
    
    df_main = df_main[df_main['detected_keyword'].notna()].copy()
    df_main = pd.merge(df_main,df_keywords[['commune_norm','Commune','Population']],left_on='detected_keyword',right_on='commune_norm',how='left')
    return df_main[['Nom Groupe','groupe_norm','Commune','Population','CA']]
    
  
df_baseCA_vendeur_grp = df_baseCA_vendeur.groupby('Nom Groupe')[df_baseCA.columns[-6:]].sum()
df_baseCA_vendeur_grp['CA'] = df_baseCA_vendeur_grp.sum(axis=1)
df_baseCA_vendeur_grp = df_baseCA_vendeur_grp.reset_index()

# df_baseCA_vendeur_grp['groupe_norm'] = df_baseCA_vendeur_grp['Nom Groupe'].str.replace('VILLE DE ','')\
                                                                          # .str.replace(' DE ',' ')\
                                                                          # .str.replace(" D'",' ')\
                                                                          # .str.replace(" DU ",' ')\
                                                                          # .str.replace('EPCI CC','')\
                                                                          # .str.replace('PAYS ','')\
                                                                          # .str.replace("CC ",' ')\
                                                                          # .str.replace("CNES ",' ')\
                                                                          # .apply(normalize_text)\
                                                                          # .str.replace('COMMUNES','')\
                                                                          # .str.replace('COMMUNAUTES','')\
                                                                          # .str.replace('COMMUNE','')\
                                                                          # .str.replace('COMMUNAUTE','')\
                                                                          # .str.replace('AGGLOMERATION','')\
                                                                          # .str.replace('AGGLO','')\
                                                                          # .str.replace('ETENVIRO','')\
                                                                          # .str.replace('CODECOM','')\
                                                                          # .str.replace('CODE','')\
                                                                          # .str.replace("TERRITOIRES",'')\
                                                                          # .str.replace("TERRITOIRE",'')\
                                                                          # .str.replace("GRAND",'')

df_test_matching = pd.read_csv(r'./population/test_matching.csv',sep=';')
df_merged_df_baseCA_vendeur_grp = pd.merge(df_baseCA_vendeur_grp,df_test_matching, on = 'Nom Groupe', how = "left")
df_merged_df_baseCA_vendeur_grp['Ville'] = df_merged_df_baseCA_vendeur_grp['Ville'].apply(normalize_text).str.replace("'", '')

df_communes = pd.read_csv(r'./population/donnees_communes_2022.csv', sep=';', encoding='utf-8')
df_communes = df_communes[df_communes['DEP'].isin([8, 10, 51, 52, 54, 55, 57, 67, 68, 88])]
df_communes['commune_norm'] = df_communes['Commune'].apply(normalize_text).str.replace("'", '') 

df_merged_vendeur_communes = pd.merge(df_merged_df_baseCA_vendeur_grp,df_communes, left_on = 'Ville', right_on='commune_norm', how = "left")

tab4.header("Tableau non filtré")
tab4.write(df_baseCA_vendeur_grp)

tab4.header("Matching Nom Groupe / Commune INSEE")
tab4.write(df_merged_vendeur_communes)


#__________________

def categorize_population(population):
    if population < 1000:
        return "Moins de 1 000 hab"
    elif population < 20000:
        return "Moins de 20 000 hab"
    elif population < 50000:
        return "Moins de 50 000 hab"
    else:
        return "Plus de 50 000 hab"

df_merged_vendeur_communes["Categorie Population"] = df_merged_vendeur_communes["Population"].apply(categorize_population)

# Calcul de la moyenne par catégorie
avg_values = df_merged_vendeur_communes.groupby("Categorie Population").mean(numeric_only=True).round()

tab4.header("CA moyen par taille de Ville (pour les données disponibles selon matching)")
avg_values=avg_values.drop('REG', axis=1)
avg_values=avg_values.drop('DEP', axis=1)

avg_values.rename(columns={"Population": "Nombre hab. moyen", "CA ": "CA cumulé moyen"}, inplace=True)
tab4.write(avg_values)

stats = df_merged_vendeur_communes.groupby("Categorie Population")["CA"].agg(["mean", "std"]).reset_index()
stats.rename(columns={"mean": "moyenne", "std": "ecart_type"}, inplace=True)

df_merged_vendeur_communes = df_merged_vendeur_communes.merge(stats, on="Categorie Population", how="left")

df_merged_vendeur_communes["z_score"] = np.abs((df_merged_vendeur_communes["CA"] - df_merged_vendeur_communes["moyenne"]) / df_merged_vendeur_communes["ecart_type"])

# Détecter les valeurs aberrantes (z-score > 2 considéré comme éloigné)
outliers = df_merged_vendeur_communes[df_merged_vendeur_communes["z_score"] > 1.8]

# Affichage des valeurs les plus éloignées
tab4.header('Communes dont le CA dépasse les espérances')
tab4.write(outliers.sort_values(by="z_score", ascending=False))

# Détecter les valeurs aberrantes (z-score > 2 considéré comme éloigné)
outliers = df_merged_vendeur_communes[df_merged_vendeur_communes["z_score"] < 1.8]

# Affichage des valeurs les plus éloignées
tab4.header('Communes dont le CA est insuffisant')
tab4.write(outliers.sort_values(by="z_score", ascending=False))



tab4.header('Détail CA client')

selectbox4 = tab4.selectbox('Nom Groupe', df_baseCA_vendeur['Nom Groupe'].unique(),key = "4_1")

df_baseCA_vendeur_grp=df_baseCA_vendeur.groupby('Domaine')[df_baseCA.columns[-6:]].sum()
df_baseCA_vendeur_grp['CA'] = df_baseCA_vendeur_grp.sum(axis=1)
df_baseCA_vendeur_grp=df_baseCA_vendeur_grp.reset_index()

fig4 = px.pie(df_baseCA_vendeur_grp, values='CA', names='Domaine')
fig4.update_layout(
    title={'text': selectbox4,
           'font_color':'#FF7900',
           'font_size': 35},
    showlegend=True,  # Afficher une seule légende
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
    height=700)

tab4.plotly_chart(fig4)

