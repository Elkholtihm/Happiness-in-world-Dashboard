# importation de donnes 
import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.subplots as ps 
from streamlit_extras.metric_cards import style_metric_cards
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

#___________________________________ set page___________________________________________________________________ 
st.set_page_config(
    page_title='Happiness',
    page_icon='😄',
    layout='wide',
    initial_sidebar_state='expanded'
)
st.subheader("World Happiness Dashboard")
#________________________________data cleaning and collection ______________________________________
# importation de donnees
df=["df"+str(i) for i in range(5,24)]
filepath = [r"C:\Users\user\Desktop\ci1\projects\dataviz\happiness\{}.csv".format(year) for year in range(2005, 2024)]
j=0
for file,d in zip(filepath,df) :
    d=pd.read_csv(file)
    j+=1
    if j==1:
        data=d
    else:
        data=pd.concat([data,d.iloc[0:,:]],axis=0)
#suppression de certain lignes
data.reset_index(drop=True, inplace=True)
data=data.loc[(data["Country"] != "Hong Kong S.A.R. of China")]
data=data.loc[(data["Country"] != "Taiwan Province of China")]
data=data.loc[(data["Country"] != "Kosovo")]
data=data.loc[(data["Country"] != "Somaliland region")]
data=data.loc[(data["Country"] != "Israel")]
data.loc[(data["Country"]=="State of Palestine"),["Life Expectancy"]]=69.5
Africa= [
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon", "Central African Republic",
    "Chad", "Comoros", "Democratic Republic of the Congo", "Djibouti", "Egypt", "Equatorial Guinea",
    "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho",
    "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger",
    "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan",
    "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"
]

# Asia
Asia = ["Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Cambodia", "China", "Cyprus", "Georgia",
    "India", "Indonesia", "Iran", "Iraq", "Japan",'Turkiye',"State of Palestine", "Jordan", "Kazakhstan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon",
    "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal", "North Korea", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar",
    "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria", "Taiwan", "Tajikistan", "Thailand", "Timor-Leste", "Turkey",
    "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen"]
# Europe
Europe = ["Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France",'Czechia', "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kosovo", "Latvia",
    "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway",
    "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine",
    "United Kingdom", "Vatican City"]
# North America
North_America = ["Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada", "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "El Salvador",
    "Grenada", "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Trinidad and Tobago", "United States"]
# South America
South_America = ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela"]
# Oceania
Oceania = ["Australia", "Fiji", "Kiribati", "Marshall Islands", "Micronesia", "Nauru", "New Zealand", "Palau", "Papua New Guinea", "Samoa",
    "Solomon  Islands", "Tonga", "Tuvalu", "Vanuatu"] 
continents= ['Africa', 'Asia', 'Europe', 'North_America',  'South_America','Oceania']
contries=[Africa,Asia,Europe,North_America,South_America,Oceania]
conditions = [data['Country'].isin(country_list) for country_list in contries]
data['Continent'] = np.select(conditions, continents,default=None)
df = data[data['Country'].isin(['Congo (Brazzaville)', 'Congo (Kinshasa)'])].reset_index(drop = True)
df = df.drop(['Continent', 'Country'], axis = 1)
congo_combined = df.groupby(['Year']).mean().reset_index()
congo_combined['Continent'] = 'Africa'
congo_combined['Country'] = "R.D.Congo"
congo_combined = congo_combined[['Country', 'Year', 'Happiness Score', 'GDP per Capita',
       'Social Support', 'Life Expectancy', 'Freedom', 'Generosity',
       'Corruption', 'Continent']]
data.drop(data[data['Country'].isin(['Congo (Brazzaville)', 'Congo (Kinshasa)'])].index, inplace=True)
data = pd.concat([data.reset_index(drop=True), congo_combined.reset_index(drop=True)], axis=0)
data = data.drop(data[(data["Country"] == "R.D.Congo") & (data["Year"].isin([2008, 2009]))].index)
from sklearn.impute import KNNImputer
selected_data = data.drop(['Country', 'Continent'], axis = 1)
imputer = KNNImputer(n_neighbors = 2)
imputed_data = imputer.fit_transform(selected_data)
to_dataframe = pd.DataFrame(imputed_data, columns = ['Year', 'Happiness Score', 'GDP per Capita',
       'Social Support', 'Life Expectancy', 'Freedom', 'Generosity',
       'Corruption'])
data = pd.concat([to_dataframe.reset_index(drop=True), data[['Country', 'Continent']].reset_index(drop=True)], axis=1)
col_oredre = ['Country', 'Year', 'Happiness Score', 'GDP per Capita',
       'Social Support', 'Life Expectancy', 'Freedom', 'Generosity',
       'Corruption', 'Continent']
data = data[col_oredre]
data["Year"] = data["Year"].astype(int)


#_________________________ordre de plot_chart________________________________________________________________________
cont1, cont2, cont3, cont4, cont5 = st.columns([0.22, 0.22, 0.22, 0.22, 0.22])
_,slider,_=st.columns([0.1,0.8,0.1])
st.divider()
_,col4,_=st.columns([0.01,0.88,0.01])
st.divider()
col5,col6=st.columns([0.5,0.5])
st.divider()
col9,col10=st.columns([0.5,0.5])
st.divider()
_,col8,_=st.columns([0.005,0.99,0.005])
st.divider()
_,col11,col18=st.columns([0.1,0.6,0.3])
st.divider()
col14, col15, col16 = st.columns([0.5, 0.5, 0.5])
st.divider()
col13 ,col12 = st.columns([0.5, 0.5])
_,col17,_=st.columns([0.1,0.8,0.1])
#__________________________sidebar_______________________________________________________________________________
with st.sidebar:
    st.header("Filter dataset")
    Year = st.sidebar.multiselect(
        label = "select year(s)",
        options = data["Year"].unique(),
        default=[2023],
    )
    continent=st.sidebar.multiselect(
        label="select continent(s)",
        options=continents,
        default=continents,
    )

    # Définit une liste de thèmes de couleurs pour la carte choroplèthe
    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    
    # Permet à l'utilisateur de sélectionner un thème de couleur dans la liste
    selected_color_theme = st.selectbox('select color : ', color_theme_list)

    selected_country = st.selectbox('select country : ', sorted(data['Country'].unique()), index=sorted(data['Country'].unique()).index('Morocco')) 

    selected_unique_year = st.selectbox('select year : ',sorted(data['Year'].unique(), reverse=True), index=sorted(data['Year'].unique(), reverse=True).index(2023))
    

#ordonner les charts
df_select = data.loc[(data["Continent"].isin(continent)) & (data["Year"].isin(Year))]
#st.dataframe(df_select.sort_values(by=["Happiness Score"],ascending=False))


 #___________________________metric_______________________________________________________________________________________--
def compact_metric(title, value, delta):
    st.markdown(
        f"""
        <div style="box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); padding: 10px; border-radius: 10px; text-align: center; width: 230px; height: 190px; background-color: #fffff;">
            <h3 style="font-size: 18px; margin-bottom: 10px;">{title}</h3>
            <h2 style="font-size: 24px; color: #55FD55; margin-bottom: 15px;">{value} 📈</h2>
            <p style="font-size: 14px; color: #555555;">{delta} 🌍</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def metric():
    with st.container():
        def draw_mwtric(col, param, compact_metric):
            with col:
                if param == 'Corruption':
                    m = df_select.groupby(by="Country")[param].mean()
                    compact_metric(f"Min {param} Score", round(m.min(), 2), f"Country: {m.idxmin()}")
                elif param == 'Happiness Score':
                    m = df_select.groupby(by="Country")[param].mean()
                    compact_metric(f"Max {param}", round(m.max(), 2), f"Country: {m.idxmax()}")
                else:
                    m = df_select.groupby(by="Country")[param].mean()
                    compact_metric(f"Max {param} Score", round(m.max(), 2), f"Country: {m.idxmax()}")
        paramaters = ["Happiness Score", "Freedom", 'Life Expectancy', "GDP per Capita", 'Corruption']
        colones = [cont1, cont2, cont3, cont4, cont5]
        for i, j in zip(paramaters, colones):
            draw_mwtric(j, i, compact_metric)
metric()
st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)



#______________________histogramme_________________________________________________________________________________________
#graphe N°:1 graphes à bar 

with slider:
    select_value = st.slider(label="select value : ", max_value=25, min_value=2, value=(10), step=1)
    st.write("Selected value:", select_value)

# Utilisation d'une colonne (col4) pour afficher un graphique basé sur la sélection
with col4:
    # Calcul des données en regroupant par pays et calculant la moyenne du score de bonheur
    data1 = df_select.groupby(by=["Country"])["Happiness Score"].mean().reset_index().sort_values(by=["Happiness Score"], ascending=False)
    
    # Sélection des premières lignes en fonction de la valeur choisie avec le curseur
    data1 = data1.head(select_value)
    
    # Création d'un graphique à barres avec Plotly express
    fig1 = px.bar(data1, x='Country', y='Happiness Score',
                  color='Happiness Score', color_continuous_scale='RdYlBu',
                  height=400, width=1000, template="plotly")
    
    # Mise à jour du texte à l'intérieur des barres avec les scores de bonheur arrondis
    fig1.update_traces(text=round(data1["Happiness Score"], 4), textposition='inside')
    
    # Mise à jour du layout du graphique avec les titres et les axes
    fig1.update_layout(xaxis_title='Country', yaxis_title='Happiness Score',
                       title=f'The Top {select_value} ranked happiest countries')
    
    # Affichage du graphique avec Streamlit
    st.plotly_chart(fig1, use_container_width=True)




#------------------------------------variation de happiness Score en fonction des Facteures------------------------------------------------------
    
#line1---------Happiness Score VS Social-Support-and-Freedom-----
def points(variable, data):
    # Fonction pour calculer les points moyens en fonction d'une variable
    data1 = data.sort_values(by=variable)
    x_mean = []
    y_mean = []

    # Boucle pour créer des points moyens par intervalle de 10
    for i in range(0, 160, 10):
        data2 = data1.iloc[i:i+10, :]
        x_mean.append(data2[variable].mean())
        y_mean.append(data2['Happiness Score'].mean())
    
    x_mean = np.array(x_mean)
    y_mean = np.array(y_mean)
    return x_mean, y_mean

# Création d'un nouveau DataFrame agrégé par pays et moyennant plusieurs variables
data2 = data.groupby(by=["Country"])[["Happiness Score", "Social Support", "Freedom", "Corruption"]].mean()

# Définition des dimensions de la figure
w, h = 1000, 320

# Utilisation de col5 pour afficher le graphique
with col5:
    # Utilisation de la colonne col5

    # Calcul des points moyens pour différentes variables
    x1, y1 = points("Social Support", data2)
    # Calcul des points moyens pour la variable "Social Support" dans le dataframe data2

    x2, y2 = points("Freedom", data2)
    # Calcul des points moyens pour la variable "Freedom" dans le dataframe data2

    # Création d'une figure avec Plotly
    fig2 = go.Figure()
    # Initialisation d'une nouvelle figure Plotly

    # Ajout des traces pour Social Support et Freedom
    fig2.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name="Social Support"))
    # Ajout d'une trace pour la variable "Social Support" à la figure

    fig2.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name="Freedom"))
    # Ajout d'une trace pour la variable "Freedom" à la figure

    # Mise à jour du layout de la figure
    fig2.update_layout(
        xaxis_title='Social Support / Freedom', # Mise à jour du titre de l'axe x
        yaxis_title="Happiness Score", # Mise à jour du titre de l'axe y
        width=w, height=h,# Mise à jour de la largeur et de la hauteur de la figure
        plot_bgcolor='black', # Mise à jour de la couleur de fond du tracé
        paper_bgcolor='black', # Mise à jour de la couleur de fond du papier (de la figure)
        title='Variation in Happiness Score based on Social Support and Freedom' # Mise à jour du titre du graphique
    )
    
    # Affichage du graphique dans l'application Streamlit
    st.plotly_chart(fig2, use_container_width=True)




#line2---------Happiness Score VS Life Expectancy and GDP-----
def mean_compute(param):
    # Initialize empty lists to store means
    x_mean = []
    y_mean = []

    # Iterate through data in intervals of 10
    for i in range(0, 160, 10):
        # Extract subset of data for each interval and compute means
        subset_data = sorted_data.iloc[i:i + 10] 
        x_mean.append(subset_data['Happiness Score'].mean())
        y_mean.append(subset_data[param].mean())

    # Convert lists to NumPy arrays
    x_mean = np.array(x_mean)
    y_mean = np.array(y_mean)
    return x_mean, y_mean

# Group data by 'Country' and compute mean for each column
grouped_data = data.drop(['Year', 'Continent'], axis=1).groupby('Country').mean()

# Plot the data using Plotly
with col6: 
    # Sort the data based on 'Happiness Score'
    sorted_data = grouped_data.sort_values(by='Happiness Score')

    # Compute means for 'Life Expectancy' and 'GDP per Capita'
    x_lif_exp, y_lif_exp = mean_compute('Life Expectancy')
    x_gdp, y_gdp = mean_compute('GDP per Capita')

    # Create a Plotly figure
    fig = go.Figure()
    
    # Add traces for 'Life Expectancy' and 'GDP per Capita'
    fig.add_trace(go.Scatter(x=x_lif_exp, y=y_lif_exp, mode='lines', name='Life Expectancy', yaxis='y'))
    fig.add_trace(go.Scatter(x=x_gdp, y=y_gdp, mode='lines', name='GDP per Capita', line=dict(color='red'), yaxis='y2'))

    # Update layout settings for the plot
    fig.update_layout(
        yaxis=dict(title='Life Expectancy'),
        yaxis2=dict(
            title='GDP per Capita',
            overlaying='y',
            side='right',
            showgrid=False,
        ),
        width=w, height=h, plot_bgcolor='black', paper_bgcolor='black',
        xaxis=dict(title='Happiness Score'), 
        title='Variation in Happiness Score based on Life Expectancy and GDP'
    )
    
    # Display the plot using Streamlit
    st.plotly_chart(fig, use_container_width=True)



#line2---------Happiness-Score-VS corruption-----
with col10:
    x3,y3=points("Corruption",data2)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=x3, y=y3, mode='lines', name="Corruption"))
    fig2.update_layout(title='Variation in Happiness Score based on Corruption Score',
                xaxis_title='Corruption',
                yaxis_title="Happiness Score",
                width=w, height=h,
                plot_bgcolor='black',  
                paper_bgcolor='black')
    st.plotly_chart(fig2, use_container_width=True)  



#------------------------Map🌍--------------------------------------------------------------------------
# Configuration de la barre latérale
    # Fonction pour créer une carte choroplèthe
    def make_choropleth(input_df, input_id, input_column, input_color_theme):
        # Crée une carte choroplèthe avec Plotly Express
        choropleth = px.choropleth(
            input_df,               # DataFrame contenant les données
            locations=input_id,     # Colonne spécifiant les emplacements
            color=input_column,     # Colonne spécifiant les valeurs de couleur
            locationmode="country names",  # Définit le mode d'emplacement pour les noms de pays
            color_continuous_scale=input_color_theme,  # Définit l'échelle de couleurs
            scope="world",          # Définit la portée de la carte au monde
            animation_frame="Year",  # Spécifie l'animation par année
            projection="natural earth",  # Choix de la projection de la carte
            labels={'Happiness Score':'Indice de bonheur'},  # Spécifie les étiquettes
            width=1000              # Définit la largeur de la carte
        )
        
        # Met à jour la mise en page pour une meilleure visualisation
        choropleth.update_layout(
            template='plotly_white',    # Définit le modèle de la carte en fond blanc
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Définit la couleur de fond du tracé en transparent
            paper_bgcolor='rgba(0, 0, 0, 0)',  # Définit la couleur de fond du papier en transparent
            margin=dict(l=0, r=0, t=0, b=0),  # Définit la marge à 0 pour une mise en page propre
            height=400,  # Définit la hauteur de la carte
        )
        return choropleth

# Affiche une carte choroplèthe dans une mise en page en colonne
with col8:
    # Définit l'en-tête de la section de la carte
    st.markdown('Worldwide Happiness Score distribution')
    
    # Crée la carte choroplèthe en utilisant la fonction définie
    choropleth = make_choropleth(data, 'Country', 'Happiness Score', selected_color_theme)
    
    # Affiche la carte choroplèthe en utilisant Plotly
    st.plotly_chart(choropleth, use_container_width=True)



#----------------------------------------Matrice de Correlation----------------------------------------------------------------
# Utilisation de col9 pour sélectionner des colonnes spécifiques dans le dataframe 'data'
with col9:
    # Suppression des colonnes "Country" et "Continent" du dataframe 'data'
    data3 = data.drop(["Country", "Continent"], axis=1)
    
    # Calcul de la matrice de corrélation
    corr_matrix = data3.corr()
    
    # Conversion de la heatmap Matplotlib en heatmap Plotly
    fig4 = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale='blues',  # Choix de la palette de couleurs
        annotation_text=corr_matrix.values.round(2),  # Arrondi des valeurs pour les annotations
        showscale=True  # Affichage de l'échelle des couleurs
    )
    
    # Mise à jour du layout de la figure
    fig.update_layout(
        title='Correlation Heatmap',  # Titre de la heatmap
        xaxis=dict(title='Features'),  # Titre de l'axe x
        yaxis=dict(title='Features')  # Titre de l'axe y
    )
    
    # Affichage de la heatmap avec Plotly dans Streamlit
    st.plotly_chart(fig4, use_container_width=True)




#----------------------------Variables per country and year--------------------------------------------------------------
# Filtrer les données en fonction du pays sélectionné et de l'année unique sélectionnée
second_filter_data = data[(data['Country'] == selected_country) & (data['Year'] == selected_unique_year)]

# Sélectionner les colonnes spécifiées pour l'analyse
columns = ['Happiness Score', 'GDP per Capita', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption']
data_selected = data[columns]

# Normaliser les données à l'aide de MinMaxScaler
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_selected)
data_normalized_df = pd.DataFrame(data_normalized, columns=columns)

# Concaténer les données normalisées avec les colonnes 'Country' et 'Year' de la table originale
data_normalized_df = pd.concat([data[['Country', 'Year']], data_normalized_df], axis=1)

# Refiltrer les données normalisées en fonction du pays et de l'année sélectionnés
second_filter_data = data_normalized_df[(data_normalized_df['Country'] == selected_country) & (data_normalized_df['Year'] == selected_unique_year)]

# Visualiser les facteurs impactant le bonheur pour le pays sélectionné
with col11:
    selected_melted_data = pd.melt(
        second_filter_data.drop(['Year', 'Happiness Score'], axis=1), 
        id_vars='Country', 
        var_name='Factors', 
        value_name='value'
    )
    fig = px.line_polar(selected_melted_data, r='value', theta='Factors', line_close=True, title=f'Factors Impacting Happiness - {selected_country}')
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True)

# Visualiser la tendance du bonheur au fil des ans pour le pays sélectionné
with col12:
    fig = px.line(data[(data['Country'] == selected_country)], x='Year', y='Happiness Score', color='Country', markers=True, title=f"Trend of Happiness Over the Years - {selected_country}")
    fig.update_traces(line_shape='linear', line=dict(dash='dash'), selector=dict(type='scatter'))
    st.plotly_chart(fig, use_container_width=True)




#_________________________________the evolution of rank over years_____________________________
# Boucle for parcourant les années de 2005 à 2023
for i in range(2005, 2024):
    # Filtrer les données pour l'année actuelle et les trier par le score de bonheur en ordre décroissant
    dr = data[data["Year"] == i].sort_values(by=["Happiness Score"], ascending=False).reset_index()
    
    # Créer une liste de classement pour chaque ligne dans le DataFrame trié
    rank = [j for j in range(1, dr.shape[0] + 1)]
    
    # Ajouter une colonne "Rank" au DataFrame contenant le classement pour chaque pays
    dr["Rank"] = rank
    
    # Concaténer le DataFrame actuel avec le DataFrame consolidé
    if i == 2005:
        df = dr
    else:
        df = pd.concat([df, dr], axis=0)

# Sélectionner les données spécifiques au pays choisi dans le DataFrame consolidé
with col13:
    datam = df[df["Country"] == selected_country]
    
    # Créer un graphique interactif avec Plotly pour visualiser l'évolution du classement du pays au fil des années
    figm = px.line(datam, x='Year', y='Rank', color='Country', symbol="Country", line_shape='spline')
    
    # Personnaliser la mise en page du graphique
    figm.update_layout(
    xaxis_title='Years',  # Titre de l'axe des x (années)
    yaxis_title=f"{selected_country} Rank",  # Titre de l'axe des y (classement du pays choisi)
    plot_bgcolor='black',  # Couleur de fond du graphique
    paper_bgcolor='black',  # Couleur de fond du papier (zone autour du graphique)
    title=f"Evolution of {selected_country}'s Happiness Rank Over Years"  # Titre du graphique
    )

    # Afficher le graphique interactif dans l'application Streamlit
    st.plotly_chart(figm, use_container_width=True)



# -------------------------------Mrank Metric-------------------------
# Définition d'une fonction pour afficher une métrique compacte sans indicateur de tendance
def compact_metric1(param):
    st.markdown(
        f"""
        <div style="box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); padding: 10px; border-radius: 10px; text-align: center; width: 230px; height: 50px; background-color: #00000;">
            <h3 style="font-size: 17px; margin-bottom: 10px;">{param} </h3>
        </div>
        """, unsafe_allow_html=True)

# Définition d'une fonction pour afficher une métrique compacte avec une tendance positive
def compact_metric2(param):
    st.markdown(
        f"""
        <div style="box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); padding: 10px; border-radius: 10px; text-align: center; width: 230px; height: 50px; background-color: #00000;">
            <p style="font-size: 17px; color: #42FF33;">{param[0:5]} <span style="color: #ffffff;">{param[5:]}</span> ⬆</p>
        </div>
        """, unsafe_allow_html=True)

# Définition d'une fonction pour afficher une métrique compacte avec une tendance négative
def compact_metric3(param):
    st.markdown(
        f"""
        <div style="box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); padding: 10px; border-radius: 10px; text-align: center; width: 230px; height: 50px; background-color: #00000;">
            <p style="font-size: 17px; color: #FC0303;">{param[0:5]} <span style="color: #ffffff;">{param[5:]}</span> ⬇</p>
        </div>
        """, unsafe_allow_html=True)

# Fonction principale pour afficher les métriques
def metric():
    # Récupération des données pour le pays sélectionné
    m = df[df['Country'] == selected_country][['Year', 'Rank']]
    
    # Affichage de la première métrique (informations de base sur le pays)
    with col14:
        compact_metric1(f"Country: {selected_country}")
    
    # Affichage de la deuxième métrique (meilleur classement avec indication de tendance positive)
    with col15:
        compact_metric2(f'best Rank: {m["Rank"].min()} (In {m.loc[m["Rank"].idxmin(), "Year"]})')
    
    # Affichage de la troisième métrique (pire classement avec indication de tendance négative)
    with col16:
        compact_metric3(f'worst Rank: {m["Rank"].max()} (In {m.loc[m["Rank"].idxmax(), "Year"]})')

# Appel de la fonction pour afficher les métriques
metric()

# Ajout d'une ligne de séparation entre les métriques
st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)


with col18:
    with st.container():
        life_expectancy = data.loc[(data["Country"]==selected_country) & (data["Year"]==selected_unique_year), "Life Expectancy"].values
        pib = data.loc[(data["Country"]==selected_country) & (data["Year"]==selected_unique_year),'GDP per Capita'].values
        s_s= data.loc[(data["Country"]==selected_country) & (data["Year"]==selected_unique_year),'Social Support'].values
        st.write(f"Pays: {selected_country}")
        st.write(f"Life expectancy: {round(life_expectancy[0],2)}")
        st.write(f"GDP Per Capita: {round(pib[0],6)}")
        st.write(f"Social support: {round(s_s[0],2)}")
with col17:
    expander=st.expander(label=f"{selected_country} data ")
    expander.write(data.loc[data["Country"]==selected_country])
