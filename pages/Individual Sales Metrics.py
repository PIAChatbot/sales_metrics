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
    
selectbox_REC = st.multiselect("Récurrence", df_baseCA['CA Récurrent'].unique(), default=df_baseCA['CA Récurrent'].unique().tolist(), key="global_2")  # Selectbox sans label


tab1, tab2, tab3, tab4= st.tabs(["CA Général",
 "Top Client par CA",
 "Top Client par Domaine",
 "Balance CA",
])




df_baseCA_vendeur=df_baseCA[df_baseCA['Vendeur']==selectbox_vendeur]

df_baseCA_vendeur = df_baseCA_vendeur[df_baseCA_vendeur['CA Récurrent'].isin(selectbox_REC)]


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
                            
tab1.header("Répartition du CA par domaine pour le dernier semestre")

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

# selectbox_REC = tab2.multiselect("Récurrence", df_baseCA['CA Récurrent'].unique(), default=df_baseCA['CA Récurrent'].unique().tolist(), key="global_2")  # Selectbox sans label

# df_baseCA_vendeur = df_baseCA_vendeur[df_baseCA_vendeur['CA Récurrent'].isin(selectbox_REC)]

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
tab2.header("Top "+selectbox2+" Clients par CA pour le dernier semestre")
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

avg_values = df_baseCA_vendeur.groupby("Categorie Population").mean(numeric_only=True).round()
std_values = df_baseCA_vendeur.groupby("Categorie Population").std(numeric_only=True).round()

# Fusionner les deux résultats dans un seul DataFrame pour un affichage plus clair
result = pd.concat([avg_values, std_values], keys=['Moyenne', 'Ecart Type'])

result=result.reset_index()

cols_numeriques_sans_population = df_baseCA_vendeur.select_dtypes(include='number').columns.difference(['Population'])

stat = df_baseCA_vendeur.groupby("Nom Groupe").agg(
    {**{col: 'sum' for col in cols_numeriques_sans_population},  # Somme pour les colonnes numériques
     **{col: 'first' for col in df_baseCA_vendeur.columns if col not in cols_numeriques_sans_population and col != 'Population'}}  # Conserver la première valeur pour les autres colonnes
)

for cat in result['Categorie Population'].unique():

    stat_cat = stat[stat['Categorie Population'] == cat]
    result_cat_moyenne = result[(result['Categorie Population'] == cat) & (result['level_0'] == "Moyenne")]
    result_cat_ecart_type = result[(result['Categorie Population'] == cat) & (result['level_0'] == "Ecart Type")]
    
    if not result_cat_moyenne.empty and not result_cat_ecart_type.empty:
        moyenne_ca = result_cat_moyenne['CA'].values[0]  # Valeur de CA pour la moyenne
        ecart_type_ca = result_cat_ecart_type['CA'].values[0]  # Valeur de CA pour l'écart type
        
        z_scores = np.abs(stat_cat['CA'] - moyenne_ca) / ecart_type_ca
        
        stat.loc[stat['Categorie Population'] == cat, 'z_score'] = z_scores



#____

# 1. Moyenne du CA par catégorie de population (et non plus par domaine)
moyenne_CA_par_catpop = stat.groupby('Categorie Population')['CA'].mean().reset_index()
moyenne_CA_par_catpop = moyenne_CA_par_catpop.rename(columns={'CA': 'CA_Moyenne'})

# 2. Fusionner avec le dataframe global
df = pd.merge(stat, moyenne_CA_par_catpop, on='Categorie Population', how='left')

# 3. Calculer la différence et identifier les clients à renforcer
df['Diff_CA'] = df['CA'] - df['CA_Moyenne']
df['Priorite'] = df['Diff_CA'].apply(lambda x: 'Travail à renforcer' if x < 0 else 'Travail OK')
clients_a_renforcer = df[df['Priorite'] == 'Travail à renforcer']

# 4. Identifier les meilleurs clients (top 10 CA) par catégorie de population
top_clients = df.groupby('Categorie Population').apply(lambda x: x.nlargest(10, 'CA')).reset_index(drop=True)

# 5. Top 3 domaines les plus fréquents parmi les meilleurs clients
def get_top_3_domains(group):
    top_domains = group['Domaine'].value_counts().head(3).index.tolist()
    return ' / '.join(top_domains)

top_3_domains = top_clients.groupby('Categorie Population').apply(get_top_3_domains).reset_index()
top_3_domains = top_3_domains.rename(columns={0: 'Domaine_Most_Frequent'})

# 6. Fusionner les domaines les plus fréquents avec les clients à renforcer
clients_a_renforcer = pd.merge(clients_a_renforcer, top_3_domains, on='Categorie Population', how='left')

# Créer un message plus détaillé pour la feuille de route du commercial
clients_a_renforcer['Message_Feuille_de_Route'] = clients_a_renforcer.apply(
    lambda row: f"Bonjour {row['Vendeur']},\n\n"
                f"Nous vous recommandons de renforcer votre activité pour le client '{row['Nom Groupe']}' situé à '{row['Commune']}' dans la catégorie de population '{row['Categorie Population']}'. "
                f"Actuellement, le chiffre d'affaires de ce client est inférieur à la moyenne pour sa catégorie de population, ce qui indique qu'il pourrait être nécessaire de mettre plus d'efforts sur ce compte.\n\n"
                f"En analysant les clients similaires dans la même catégorie de population, voici les 3 domaines les plus fréquents où les clients ayant un meilleur CA se sont montrés plus performants : {row['Domaine_Most_Frequent']}.\n"
                f"Nous vous conseillons donc de concentrer vos efforts sur ces domaines pour optimiser le potentiel de ce client.\n\n"
                f"Actions suggérées :\n"
                f"- Prendre contact avec le client pour discuter des opportunités de croissance dans ces domaines.\n"
                f"- Évaluer les besoins spécifiques de ce client pour adapter l'offre dans ces domaines.\n\n"
                f"Bonne chance dans votre démarche commerciale !\n\n"
                f"Bien cordialement,\nVotre équipe commerciale.",
    axis=1
)

# Affichage avec Streamlit

# Titre
tab4.title("Feuille de Route Commerciale")

# Introduction générale
tab4.markdown("""
    Cette feuille de route contient des recommandations pour les commerciaux en fonction des clients dont le chiffre d'affaires est inférieur à la moyenne de la catégorie de population.
    Vous trouverez les clients à renforcer ainsi que des domaines stratégiques à cibler pour améliorer leurs performances.
""")

# Afficher la table des clients à renforcer
tab4.subheader("Clients à Renforcer")
tab4.dataframe(clients_a_renforcer[['Nom Groupe', 'Vendeur', 'Commune', 'Categorie Population', 'Domaine_Most_Frequent', 'Message_Feuille_de_Route']])

# Ajouter un bouton pour télécharger le fichier CSV
csv = clients_a_renforcer[['Nom Groupe', 'Vendeur', 'Commune', 'Categorie Population', 'Domaine_Most_Frequent', 'Message_Feuille_de_Route']].to_csv(index=False)
tab4.download_button(label="Télécharger la feuille de route", data=csv, file_name="feuille_de_route_commerciale.csv", mime="text/csv")

# Afficher un message explicatif
tab4.markdown("""
    **Conseils pour les commerciaux :**  
    Pour chaque client à renforcer, nous vous conseillons de prendre contact avec eux et de discuter des opportunités dans les domaines les plus fréquents parmi les clients ayant un meilleur CA.  
    Ce tableau vous aidera à déterminer les priorités et à concentrer vos efforts sur les domaines les plus stratégiques.
""")


selectbox4 = tab4.selectbox('Nom Groupe', stat['Nom Groupe'].unique(),key = "4_1")

fig4 = px.pie(stat, values='CA', names='Domaine')
fig4.update_layout(
    title={'text': selectbox4,
           'font_color':'#FF7900',
           'font_size': 35},
    showlegend=True,  # Afficher une seule légende
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
    height=700)
tab4.plotly_chart(fig4)

tab4.header("Meilleur CA au regard du bassin de population")
tab4.write(stat.sort_values('z_score',ascending=False).head())
