import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Dashboard de performance - Pneus Michelin")

# Sélection du dossier de données
categories = ['performance_on_search', 'tesla_KWD', 'electric_KWD', 'manufacturer_URL', 'electric_URL']
selected_category = st.sidebar.selectbox("Choisir une catégorie de pneu :", categories)

# Chemin vers les fichiers CSV
base_path = rf"D:\projects_python\jellyfish_michelin\Data Michelin\{selected_category}"

# Fonction de nettoyage des colonnes contenant des pourcentages concaténés
def clean_percentage_column(df, column_name):
    """ Nettoie une colonne contenant des pourcentages concaténés en extrayant la valeur numérique """
    df[column_name] = df[column_name].replace('%', '', regex=True).astype(float) / 100
    return df

# Liste des fichiers CSV pour chaque catégorie
files = {
    "appareils": "Appareils.csv",
    "Apparence dans les résultats de recherche" : "Apparence dans les résultats de recherche.csv",
    "dates": "Dates.csv",
    "filtres" : "Filtres.csv",
    "pages" : "Pages.csv",
    "pays": "Pays.csv",
    "requêtes": "Requêtes.csv"
}

# Fonction pour charger les données
def load_data(files, base_path):
    data = {}
    for key, filename in files.items():
        filepath = os.path.join(base_path, filename)
        df = pd.read_csv(filepath, encoding="utf-8")  # Ou "ISO-8859-1" si accents KO
        df["Catégorie"] = selected_category
        
        # Nettoyage des colonnes contenant des pourcentages concaténés
        if 'CTR' in df.columns:
            df = clean_percentage_column(df, 'CTR')
        if 'Position' in df.columns:
            df = clean_percentage_column(df, 'Position')

        data[key] = df
    
    # Ajouter le fichier Ranking_Sites.xlsx
    ranking_filepath = r"D:\projects_python\jellyfish_michelin\Data Michelin\EV USA Keywords - SEMrush.xlsx"
    ranking_df = pd.read_excel(ranking_filepath)  # Charge l'Excel avec les colonnes du ranking
    data["ranking"] = ranking_df
    
    return data

# Charger les données de la catégorie sélectionnée
data = load_data(files, base_path)

# Récupération des données spécifiques pour les filtres
dates_df = data["dates"]
pays_df = data["pays"]

# Filtrage par période (Date)
st.sidebar.subheader("Filtrer par période")
date_min = dates_df['Date'].min()
date_max = dates_df['Date'].max()

# Création d'un sélecteur de dates
start_date = st.sidebar.date_input("Date de début", date_min)
end_date = st.sidebar.date_input("Date de fin", date_max)

# Filtrage des données par période
dates_df['Date'] = pd.to_datetime(dates_df['Date'])
filtered_dates_df = dates_df[(dates_df['Date'] >= pd.to_datetime(start_date)) & (dates_df['Date'] <= pd.to_datetime(end_date))]

# Filtre par pays
st.sidebar.subheader("Filtrer par pays")
countries = pays_df['Pays'].unique()
selected_countries = st.sidebar.multiselect("Sélectionner des pays", countries, default=countries)

filtered_country_df = pays_df[pays_df['Pays'].isin(selected_countries)]

# Calcul des KPIs sur les données filtrées
total_clicks = filtered_dates_df["Clics"].sum()
total_impressions = filtered_dates_df["Impressions"].sum()
avg_ctr = filtered_dates_df["CTR"].mean()
avg_position = filtered_dates_df["Position"].mean()

# Affichage des KPIs
st.metric("Total Clics", total_clicks)
st.metric("Total Impressions", total_impressions)
st.metric("CTR Moyen", f"{avg_ctr:.2%}")
st.metric("Position Moyenne", f"{avg_position:.2f}")

# Alertes de performance
st.subheader("Alertes de Performance")
if avg_ctr < 0.02:
    st.warning("Attention : Le CTR est très faible ! Cela pourrait indiquer un problème d'engagement.")
if avg_position > 15:
    st.warning("Attention : La position moyenne est élevée. Cela pourrait affecter la visibilité de votre contenu.")

# 1. Répartition des Clics par Pays
country_clicks = filtered_country_df.groupby("Pays")["Clics"].sum()
st.subheader("Répartition des Clics par Pays")
st.bar_chart(country_clicks)

# 2. Pages les plus Populaires
pages_df = data["pages"]
page_clicks = pages_df.sort_values(by="Clics", ascending=False).head(10)
st.subheader("Pages les plus Populaires")
st.dataframe(page_clicks[['Pages les plus populaires', 'Clics']])

# 3. Tendances des Clics au Fil du Temps
fig, ax = plt.subplots()
ax.plot(filtered_dates_df['Date'], filtered_dates_df['Clics'], label='Clics', color='blue')
ax.set_xlabel('Date')
ax.set_ylabel('Clics')
ax.set_title('Évolution des Clics au fil du Temps')
st.pyplot(fig)

# 4. Performances par Type d’Appareil
appareils_df = data["appareils"]
device_clicks = appareils_df.groupby("Appareil")["Clics"].sum()
st.subheader("Performances par Type d'Appareil")
st.bar_chart(device_clicks)

# 5. Analyse des Requêtes Fréquentes
requêtes_df = data["requêtes"]
top_queries = requêtes_df.sort_values(by="Clics", ascending=False).head(10)
st.subheader("Requêtes les plus Fréquentes")
st.dataframe(top_queries[['Requêtes les plus fréquentes', 'Clics']])

# 6. Performances des Apparences dans les Résultats de Recherche
appearances_df = data["Apparence dans les résultats de recherche"]
appearance_clicks = appearances_df.groupby("Apparence dans les résultats de recherche")["Clics"].sum()
st.subheader("Performances des Apparences dans les Résultats de Recherche")
st.bar_chart(appearance_clicks)

# 7. Prédiction des clics futurs avec régression linéaire
# Préparation des données pour la régression linéaire
filtered_dates_df['Days'] = (filtered_dates_df['Date'] - filtered_dates_df['Date'].min()).dt.days
X = filtered_dates_df[['Days']]  # Variable indépendante (le nombre de jours)
y = filtered_dates_df['Clics']   # Variable dépendante (les clics)

# Création du modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Prévision des clics pour les 10 jours suivants
future_days = np.array([(filtered_dates_df['Date'].max() - filtered_dates_df['Date'].min()).days + i for i in range(1, 11)]).reshape(-1, 1)
predictions = model.predict(future_days)

# Affichage des prévisions
st.subheader("Prévisions des Clics pour les 10 Jours à Venir")
future_dates = pd.date_range(filtered_dates_df['Date'].max() + pd.Timedelta(days=1), periods=10)
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Prédiction des Clics': predictions
})
st.write(forecast_df)

# Optionnel: Visualisation des prévisions
fig, ax = plt.subplots()
ax.plot(filtered_dates_df['Date'], filtered_dates_df['Clics'], label='Clics réels', color='blue')
ax.plot(future_dates, predictions, label='Clics prévus', color='red', linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Clics')
ax.set_title('Prédiction des Clics Futurs')
ax.legend()
st.pyplot(fig)

# 8. Tendances du CTR au fil du temps
fig, ax = plt.subplots()
ax.plot(filtered_dates_df['Date'], filtered_dates_df['CTR'], label='CTR', color='green')
ax.set_xlabel('Date')
ax.set_ylabel('CTR')
ax.set_title('Évolution du CTR au fil du Temps')
st.pyplot(fig)

# 9. Tendances de la Position Moyenne au fil du temps
fig, ax = plt.subplots()
ax.plot(filtered_dates_df['Date'], filtered_dates_df['Position'], label='Position Moyenne', color='red')
ax.set_xlabel('Date')
ax.set_ylabel('Position Moyenne')
ax.set_title('Évolution de la Position Moyenne au fil du Temps')
st.pyplot(fig)



