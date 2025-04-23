import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Michelin",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Appliquer du CSS pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #005F9E;
        text-align: center;
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 2px solid #F0F2F6;
    }
    .subheader {
        font-size: 24px;
        font-weight: bold;
        color: #005F9E;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-left: 5px solid #005F9E;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 14px;
        color: #6C757D;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #212529;
    }
    .warning-card {
        background-color: #FDEBD0;
        border-left: 5px solid #F39C12;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .info-card {
        background-color: #E9F7EF;
        border-left: 5px solid #27AE60;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Dashboard de Performance - Pneus Michelin</h1>', unsafe_allow_html=True)

# Sélection du dossier de données
with st.sidebar:
    st.sidebar.title("🛞 Michelin Analytics")
    st.sidebar.markdown("---")
    
    categories = ['performance_on_search', 'tesla_KWD', 'electric_KWD', 'manufacturer_URL', 'electric_URL']
    selected_category = st.selectbox("Choisir une catégorie de pneu :", categories)

# Chemin vers les fichiers CSV
base_path = rf"C:\Users\MaximeKHAZNADJI\OneDrive - Sia Partners\data_ia_hetic\michelin\jellyfish_michelin\Data Michelin\Data Michelin\{selected_category}"

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
    ranking_filepath = r"C:\Users\MaximeKHAZNADJI\OneDrive - Sia Partners\data_ia_hetic\michelin\jellyfish_michelin\Data Michelin\Data Michelin\EV USA Keywords - SEMrush.xlsx"
    ranking_df = pd.read_excel(ranking_filepath)  # Charge l'Excel avec les colonnes du ranking
    data["ranking"] = ranking_df
    
    return data

# Charger les données de la catégorie sélectionnée
try:
    data = load_data(files, base_path)
    loading_success = True
except Exception as e:
    st.error(f"Erreur lors du chargement des données: {e}")
    loading_success = False

if loading_success:
    # Récupération des données spécifiques pour les filtres
    dates_df = data["dates"]
    pays_df = data["pays"]

    # Conversion de la colonne Date en datetime
    dates_df['Date'] = pd.to_datetime(dates_df['Date'])

    # Filtrage par période (Date)
    with st.sidebar:
        st.subheader("Filtrer par période")
        date_min = dates_df['Date'].min().date()  # Convertir en objet date
        date_max = dates_df['Date'].max().date()  # Convertir en objet date

        # Création d'un sélecteur de dates
        start_date = st.sidebar.date_input("Date de début", date_min)
        end_date = st.sidebar.date_input("Date de fin", date_max)

    # Filtrage des données par période
    filtered_dates_df = dates_df[(dates_df['Date'] >= pd.to_datetime(start_date)) & (dates_df['Date'] <= pd.to_datetime(end_date))]

    # Filtre par pays
    with st.sidebar:
        st.subheader("Filtrer par pays")
        countries = pays_df['Pays'].unique()
        selected_countries = st.multiselect("Sélectionner des pays", countries, default=countries)

    filtered_country_df = pays_df[pays_df['Pays'].isin(selected_countries)]

    # Calcul des KPIs sur les données filtrées
    total_clicks = filtered_dates_df["Clics"].sum()
    total_impressions = filtered_dates_df["Impressions"].sum()
    avg_ctr = filtered_dates_df["CTR"].mean()
    avg_position = filtered_dates_df["Position"].mean()
    
    # NOUVELLES MÉTRIQUES
    
    # 1. Ratio mobile/desktop
    appareils_df = data["appareils"]
    mobile_clicks = appareils_df[appareils_df["Appareil"] == "Mobile"]["Clics"].sum() if "Mobile" in appareils_df["Appareil"].values else 0
    mobile_ratio = mobile_clicks / total_clicks if total_clicks > 0 else 0
    
    # 2. Taux de croissance (comparaison avec période précédente)
    if filtered_dates_df.shape[0] > 0:
        current_period_clicks = filtered_dates_df["Clics"].sum()
        days_in_period = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        
        # Période précédente de même longueur
        previous_start = pd.to_datetime(start_date) - timedelta(days=days_in_period)
        previous_end = pd.to_datetime(start_date) - timedelta(days=1)
        
        previous_period_df = dates_df[(dates_df['Date'] >= previous_start) & (dates_df['Date'] <= previous_end)]
        previous_period_clicks = previous_period_df["Clics"].sum() if not previous_period_df.empty else 0
        
        growth_rate = ((current_period_clicks - previous_period_clicks) / previous_period_clicks * 100) if previous_period_clicks > 0 else 100
    else:
        growth_rate = 0
    
    # 3. Score d'engagement
    # Défini comme la somme de CTR pondérée par les impressions
    engagement_score = (filtered_dates_df["CTR"] * filtered_dates_df["Impressions"]).sum() / filtered_dates_df["Impressions"].sum() if filtered_dates_df["Impressions"].sum() > 0 else 0
    
    # 4. Score SEO (formule simplifiée basée sur la position moyenne)
    # Plus la position est basse (proche de 1), meilleur est le score (max 100)
    seo_score = max(0, min(100, 100 - (avg_position * 10)))
    
    # 5. Valeur estimée du trafic
    # Valeur approximative en euros, basée sur un coût moyen par clic pour les mots-clés similaires
    avg_cpc = 1.2  # Coût par clic estimé en euros
    traffic_value = total_clicks * avg_cpc
    
    # Affichage des KPIs en grille
    st.markdown('<h2 class="subheader">Indicateurs Clés de Performance (KPIs)</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total des Clics</div>
            <div class="metric-value">{:,}</div>
        </div>
        """.format(int(total_clicks)), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Position Moyenne</div>
            <div class="metric-value">{:.2f}</div>
        </div>
        """.format(avg_position), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Score SEO</div>
            <div class="metric-value">{:.1f}/100</div>
        </div>
        """.format(seo_score), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Impressions</div>
            <div class="metric-value">{:,}</div>
        </div>
        """.format(int(total_impressions)), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">CTR Moyen</div>
            <div class="metric-value">{:.2%}</div>
        </div>
        """.format(avg_ctr), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Score d'Engagement</div>
            <div class="metric-value">{:.2%}</div>
        </div>
        """.format(engagement_score), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Taux de Croissance</div>
            <div class="metric-value">{:.1f}%</div>
        </div>
        """.format(growth_rate), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Ratio Mobile</div>
            <div class="metric-value">{:.1%}</div>
        </div>
        """.format(mobile_ratio), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Valeur Estimée du Trafic</div>
            <div class="metric-value">{:.0f} €</div>
        </div>
        """.format(traffic_value), unsafe_allow_html=True)

    # Alertes de performance
    st.markdown('<h2 class="subheader">Alertes et Recommandations</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        alerts = []
        
        if avg_ctr < 0.02:
            alerts.append("Le CTR est très faible (< 2%). Envisagez d'améliorer les meta descriptions et les titres.")
        
        if avg_position > 15:
            alerts.append("La position moyenne est élevée (> 15). Cela pourrait affecter significativement la visibilité.")
        
        if growth_rate < -10:
            alerts.append(f"Baisse importante du trafic ({growth_rate:.1f}%) par rapport à la période précédente.")
        
        if mobile_ratio < 0.3:
            alerts.append("Faible taux de trafic mobile. Vérifiez l'optimisation mobile de votre site.")
            
        if seo_score < 40:
            alerts.append(f"Score SEO faible ({seo_score:.1f}/100). Revoyez votre stratégie de référencement.")
        
        if not alerts:
            st.markdown('<div class="info-card">Aucune alerte critique détectée pour la période sélectionnée.</div>', unsafe_allow_html=True)
        else:
            for alert in alerts:
                st.markdown(f'<div class="warning-card">{alert}</div>', unsafe_allow_html=True)
    
    with col2:
        # Recommandations
        st.markdown('<div class="info-card">Recommandation: Cibler davantage les appareils mobiles car ils représentent une part croissante du trafic.</div>', unsafe_allow_html=True)
        
        if seo_score < 50:
            st.markdown('<div class="info-card">Recommandation: Améliorer le contenu SEO pour les mots-clés principaux afin de remonter dans les résultats de recherche.</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">Recommandation: Optimiser les pages à fort trafic pour augmenter le taux de conversion.</div>', unsafe_allow_html=True)

    # Affichage des graphiques principaux
    st.markdown('<h2 class="subheader">Analyse Temporelle</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Clics & Impressions", "CTR & Position", "Performance par Dimension"])
    
    with tab1:
        # Tendances des Clics au Fil du Temps
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(filtered_dates_df['Date'], filtered_dates_df['Clics'], label='Clics', color='#1F77B4', marker='o', linewidth=2)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Clics', fontsize=12, color='#1F77B4')
        ax.tick_params(axis='y', labelcolor='#1F77B4')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Évolution des Clics au fil du Temps', fontsize=16, pad=20)
        
        # Créer un deuxième axe Y pour les impressions
        ax2 = ax.twinx()
        ax2.plot(filtered_dates_df['Date'], filtered_dates_df['Impressions'], label='Impressions', color='#FF7F0E', marker='s', linewidth=2)
        ax2.set_ylabel('Impressions', fontsize=12, color='#FF7F0E')
        ax2.tick_params(axis='y', labelcolor='#FF7F0E')
        
        # Légende
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        # Tendances du CTR et de la Position au fil du temps
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # CTR Plot
        ax1.plot(filtered_dates_df['Date'], filtered_dates_df['CTR'], color='green', marker='o', linewidth=2)
        ax1.set_ylabel('CTR', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title('Évolution du CTR au fil du Temps', fontsize=16, pad=20)
        ax1.set_ylim(bottom=0)
        # Formater y-axis en pourcentage
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Position Plot (inversé pour que les meilleures positions soient en haut)
        ax2.plot(filtered_dates_df['Date'], filtered_dates_df['Position'], color='red', marker='o', linewidth=2)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Position', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_title('Évolution de la Position Moyenne au fil du Temps', fontsize=16, pad=20)
        ax2.invert_yaxis()  # Inverser l'axe Y pour que les meilleures positions soient en haut
        
        plt.tight_layout()
        st.pyplot(fig)
        
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Répartition des Clics par Pays
            country_clicks = filtered_country_df.groupby("Pays")["Clics"].sum().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(country_clicks.index, country_clicks.values, color='skyblue')
            
            # Ajouter les valeurs au-dessus des barres
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Pays', fontsize=12)
            ax.set_ylabel('Clics', fontsize=12)
            ax.set_title('Répartition des Clics par Pays', fontsize=16, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Performances par Type d'Appareil
            device_clicks = appareils_df.groupby("Appareil")["Clics"].sum().sort_values(ascending=False)
            
            # Création d'un graphique en camembert
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(
                device_clicks.values, 
                labels=device_clicks.index,
                autopct='%1.1f%%', 
                startangle=90,
                colors=['#ff9999','#66b3ff','#99ff99','#ffcc99']
            )
            
            # Égaliser les axes pour obtenir un cercle
            ax.axis('equal')  
            ax.set_title('Répartition des Clics par Type d\'Appareil', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
    
    # Pages les plus Populaires
    st.markdown('<h2 class="subheader">Analyse de Contenu</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Pages Populaires", "Requêtes Fréquentes"])
    
    with tab1:
        pages_df = data["pages"]
        page_clicks = pages_df.sort_values(by="Clics", ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(page_clicks['Pages les plus populaires'], page_clicks['Clics'], color='#5DA5DA')
        
        # Ajouter les valeurs à côté des barres
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 5, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', 
                    va='center', fontsize=10)
        
        ax.set_xlabel('Clics', fontsize=12)
        ax.set_title('Top 10 des Pages les plus Populaires', fontsize=16, pad=20)
        ax.invert_yaxis()  # Pour afficher la page la plus populaire en haut
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        # Analyse des Requêtes Fréquentes
        requêtes_df = data["requêtes"]
        top_queries = requêtes_df.sort_values(by="Clics", ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(top_queries['Requêtes les plus fréquentes'], top_queries['Clics'], color='#60BD68')
        
        # Ajouter les valeurs à côté des barres
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 5, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', 
                    va='center', fontsize=10)
        
        ax.set_xlabel('Clics', fontsize=12)
        ax.set_title('Top 10 des Requêtes les plus Fréquentes', fontsize=16, pad=20)
        ax.invert_yaxis()  # Pour afficher la requête la plus fréquente en haut
        plt.tight_layout()
        st.pyplot(fig)
    
    # Prédiction des clics futurs avec régression linéaire
    st.markdown('<h2 class="subheader">Prévisions et Analytics Avancés</h2>', unsafe_allow_html=True)
    
    if filtered_dates_df.shape[0] > 3:  # Au moins 3 points de données pour la régression
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
        
        # S'assurer que les prédictions ne sont pas négatives
        predictions = np.maximum(predictions, 0)

        # Création des dates futures
        future_dates = pd.date_range(filtered_dates_df['Date'].max() + pd.Timedelta(days=1), periods=10)
        
        # Création d'un DataFrame pour les prédictions
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Prédiction des Clics': predictions.astype(int)
        })

        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Visualisation des prévisions
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Données historiques
            ax.plot(filtered_dates_df['Date'], filtered_dates_df['Clics'], 
                   marker='o', linestyle='-', color='blue', label='Clics réels')
            
            # Prédictions
            ax.plot(future_dates, predictions, 
                   marker='s', linestyle='--', color='red', label='Clics prévus')
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Clics', fontsize=12)
            ax.set_title('Prédiction des Clics pour les 10 Jours à Venir', fontsize=16, pad=20)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
        with col2:
            st.markdown('<p class="metric-title" style="font-size: 16px; margin-top: 0px;">Prévisions des 10 prochains jours</p>', unsafe_allow_html=True)
            st.dataframe(forecast_df.set_index('Date'))
            
            # Afficher les statistiques de performance du modèle
            coef = model.coef_[0]
            trend_direction = "à la hausse" if coef > 0 else "à la baisse"
            avg_predicted = predictions.mean()
            
            st.markdown(f"""
            <div class="info-card">
                <b>Analyse de la tendance:</b><br>
                La tendance générale est {trend_direction}.<br>
                Clics moyens prévus: {int(avg_predicted)} par jour.<br>
                Variation quotidienne moyenne: {abs(coef):.1f} clics.
            </div>
            """, unsafe_allow_html=True)
            
    # Analyse de corrélation
    st.markdown('<h2 class="subheader">Analyse de Corrélation</h2>', unsafe_allow_html=True)
    
    # Relation entre Position et CTR
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(filtered_dates_df['Position'], filtered_dates_df['CTR'], 
              alpha=0.7, color='purple', s=50)
    
    # Ajouter une ligne de tendance (régression linéaire)
    if len(filtered_dates_df) > 1:
        z = np.polyfit(filtered_dates_df['Position'], filtered_dates_df['CTR'], 1)
        p = np.poly1d(z)
        ax.plot(filtered_dates_df['Position'], p(filtered_dates_df['Position']), 
               linestyle='--', color='red', linewidth=2)
    
    ax.set_xlabel('Position Moyenne', fontsize=12)
    ax.set_ylabel('CTR', fontsize=12)
    ax.set_title('Corrélation entre Position et CTR', fontsize=16, pad=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #F8F9FA; border-radius: 10px;">
        <p style="color: #6C757D; font-size: 14px;">Dashboard de Performance Michelin | Développé par Sia Partners | Dernière mise à jour: Avril 2025</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("Veuillez vérifier les chemins des fichiers et réessayer.")