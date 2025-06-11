import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.datasets import load_iris
import io
import zipfile
from io import BytesIO
from scipy.stats import shapiro, kstest, norm, t, chi2, probplot
from statsmodels.stats.proportion import proportion_confint

# --- Page Configuration ---
st.set_page_config(
    page_title="Visualisation de Données",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark and Technological Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;500&display=swap');

    /* Main app styling */
    body {
        font-family: 'Roboto', sans-serif;
        color: #e0e6ed !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%) !important;
        border-right: 2px solid #00ffff;
        box-shadow: 5px 0 15px rgba(0, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stButton label {
        color: #b0c4de !important;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Main title */
    h1 {
        color: #00ffff;
        text-align: center;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        padding-top: 1.5rem;
    }

    /* Sub-headers */
    h2, h3, h4 {
        color: #00ffff;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.4);
    }

    /* Selectbox and radio styling */
    .stSelectbox, .stRadio {
        background-color: rgba(15, 15, 35, 0.8) !important;
        border: 1px solid #00ffff;
        border-radius: 5px;
        padding: 10px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #0f3460 0%, #16537e 100%);
        border-bottom: 2px solid #00ffff;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #b0c4de !important;
        font-family: 'Roboto', sans-serif;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: #00ffff !important;
        color: #0f0f23 !important;
        text-shadow: none;
    }

    /* Expander styling */
    .st-expander {
        border: 1px solid #00ffff;
        border-radius: 10px;
        background: rgba(15, 15, 35, 0.8);
    }
    .st-expander header {
        font-size: 1.2rem;
        color: #00ffff;
        font-family: 'Orbitron', monospace;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }

    /* DataFrame styling */
    .stDataFrame {
        background: rgba(15, 15, 35, 0.8) !important;
        border: 1px solid rgba(0, 255, 255, 0.3);
    }

    /* Uploader styling in sidebar */
    [data-testid="stFileUploader"] {
        border: 2px dashed #00ffff;
        background-color: #1a1a2e;
        padding: 20px;
        border-radius: 10px;
    }

    /* Alert styling */
    .stAlert {
        border-radius: 0.5rem;
        background: rgba(15, 15, 35, 0.8);
        border: 1px solid #00ffff;
        color: #e0e6ed !important;
    }

    /* Uploaded file styling */
    .uploaded-file {
        color: #00ff88 !important;
        font-weight: bold;
    }

    /* Footer styling */
    .footer {
        font-size: 0.8rem;
        color: #00ffff;
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 1px solid #00ffff;
        border-radius: 5px;
    }

    /* Author info box */
    .author-info {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 2px solid #00ffff;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 10px 30px rgba(0, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- Plotting Theme Configuration ---
plt.style.use('dark_background')

# Configuration matplotlib (not used in this code but included for consistency)
plt.rcParams.update({
    'figure.facecolor': '#0c0c0c',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#00ffff',
    'axes.labelcolor': '#e0e6ed',
    'xtick.color': '#e0e6ed',
    'ytick.color': '#e0e6ed',
    'grid.color': '#4a5568',
    'text.color': '#e0e6ed',
    'legend.facecolor': '#0f0f23',
    'legend.edgecolor': '#00ffff'
})

# Configuration du template Plotly
plotly_dark_template = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor='#1a1a2e', 
        paper_bgcolor='#0c0c0c', 
        font_color='#e0e6ed',
        xaxis=dict(gridcolor='#4a5568', linecolor='#e0e6ed'),
        yaxis=dict(gridcolor='#4a5568', linecolor='#e0e6ed'),
        title_font_color='#00ffff', 
        xaxis_title_font_color='#00ffff',
        yaxis_title_font_color='#00ffff',
        legend=dict(bgcolor='rgba(15,15,35,0.8)', bordercolor='#00ffff')
    )
)

# --- Application Title ---
st.title("📊 Exploration et Visualisation de Données")
st.markdown("<p style='text-align: center; color: #b0c4de; font-family: Roboto, sans-serif;'>Une plateforme interactive pour explorer et visualiser vos données quantitatives et qualitatives.</p>", unsafe_allow_html=True)

# Fonction pour charger les données demo
@st.cache_data
def load_demo_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Ajout de variables qualitatives supplémentaires pour la démo
    np.random.seed(42)  # Pour la reproductibilité
    df['region'] = np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], size=len(df))
    df['qualite'] = np.random.choice(['Faible', 'Moyenne', 'Haute'], size=len(df), p=[0.2, 0.5, 0.3])

    return df

# Zone d'upload de fichiers
st.sidebar.header("Importation des Données")
data_source = st.sidebar.radio("Source des données", ["Données de démo", "Importer mes données"], key="data_source_radio")

df = None

if data_source == "Données de démo":
    df = load_demo_data()
    st.sidebar.success("Utilisation des données de démo Iris")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Importez votre fichier (CSV ou Excel)",
        type=["csv", "xlsx", "xls"],
        help="Format attendu: CSV ou Excel avec en-têtes de colonnes"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.sidebar.success(f"Fichier chargé: **{uploaded_file.name}**")
            st.sidebar.write(f"🔍 **{len(df)}** observations, **{len(df.columns)}** variables")
            st.sidebar.markdown(f"<p class='uploaded-file'>Fichier chargé: {uploaded_file.name}</p>", unsafe_allow_html=True)

            # Aperçu des données
            with st.sidebar.expander("Aperçu des données"):
                st.write(f"Lignes: {df.shape[0]}, Colonnes: {df.shape[1]}")
                st.dataframe(df.head(3))

        except Exception as e:
            st.sidebar.error(f"Erreur lors du chargement du fichier: {str(e)}")
            st.error(f"Impossible de charger ou de traiter le fichier. Erreur : {str(e)}")
            st.stop()  # Arrête l'exécution si le chargement échoue

# Vérification si des données sont disponibles
if df is None:
    if data_source == "Importer mes données":
        st.warning("Veuillez importer un fichier de données ou sélectionner 'Données de démo'.")
    else:
        st.warning("Chargement des données de démo en cours ou échec.")
    st.stop()  # Arrête l'exécution si aucune donnée n'est chargée

# --- Analyse Exploratoire Générale ---
st.header("Aperçu Général des Données")
col_info1, col_info2 = st.columns(2)
with col_info1:
    st.subheader("Informations sur le DataFrame")
    # Créer un DataFrame pour les informations des colonnes
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.notnull().sum().astype(int),
        'Dtype': df.dtypes.astype(str)
    }).reset_index(drop=True)
    st.dataframe(info_df, use_container_width=True)

with col_info2:
    st.subheader("Statistiques Descriptives (Variables Quantitatives)")
    st.dataframe(df.describe(include=np.number).style.format("{:.2f}"))

st.subheader("Statistiques Descriptives (Variables Qualitatives)")
st.dataframe(df.describe(include='object'))

st.markdown("---")  # Séparateur visuel

# --- Section de Visualisation par Variable ---
st.header("Analyse par Variable")

# Sidebar pour la sélection des variables
st.sidebar.header("Paramètres de Visualisation")
all_vars = df.columns.tolist()
quantitative_vars = df.select_dtypes(include=np.number).columns.tolist()
qualitative_vars = df.select_dtypes(exclude=np.number).columns.tolist()

# Ajout des informations sur l'auteur dans la sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="author-info">
    <h4>🧾 À propos de l'auteur</h4>
    <p><b>Nom:</b> N'dri</p>
    <p><b>Prénom:</b> Abo Onesime</p>
    <p><b>Rôle:</b> Data Analyst / Scientist</p>
    <p><b>Téléphone:</b> 07-68-05-98-87 / 01-01-75-11-81</p>
    <p><b>Email:</b> <a href="mailto:ndriablatie123@gmail.com" style="color:#00ff88;">ndriablatie123@gmail.com</a></p>
    <p><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/abo-onesime-n-dri-54a537200/" target="_blank" style="color:#00ff88;">Profil LinkedIn</a></p>
    <p><b>GitHub:</b> <a href="https://github.com/Aboonesime" target="_blank" style="color:#00ff88;">Mon GitHub</a></p>
</div>
""", unsafe_allow_html=True)

# Permettre à l'utilisateur de choisir le type *ou* de sélectionner directement
analysis_mode = st.sidebar.radio("Choisir par:", ["Type de variable", "Nom de variable"])

selected_var = None

if analysis_mode == "Type de variable":
    var_type = st.sidebar.radio("Type de variable", ["Quantitative", "Qualitative"], key="var_type_radio")
    if var_type == "Quantitative":
        if not quantitative_vars:
            st.error("Aucune variable quantitative trouvée dans les données.")
            st.stop()
        selected_var = st.sidebar.selectbox("Sélectionnez une variable quantitative", quantitative_vars, key="quant_select")
    else:  # Qualitative
        if not qualitative_vars:
            st.error("Aucune variable qualitative trouvée dans les données.")
            st.stop()
        selected_var = st.sidebar.selectbox("Sélectionnez une variable qualitative", qualitative_vars, key="qual_select")
else:  # Nom de variable
    if not all_vars:
        st.error("Aucune colonne trouvée dans les données.")
        st.stop()
    selected_var = st.sidebar.selectbox("Sélectionnez une variable", all_vars, key="all_var_select")
    # Déterminer le type de la variable sélectionnée
    if selected_var in quantitative_vars:
        var_type = "Quantitative"
    elif selected_var in qualitative_vars:
        var_type = "Qualitative"
    else:
        st.error(f"Type de variable inconnu pour {selected_var}")
        st.stop()

# Fonctions de visualisation pour variables quantitatives
def plot_quantitative(var):
    if var not in df.columns:
        st.error(f"La variable '{var}' n'existe pas dans le DataFrame.")
        return
    if not pd.api.types.is_numeric_dtype(df[var]):
        st.error(f"La variable '{var}' n'est pas quantitative.")
        return

    st.subheader(f"Analyse de la variable quantitative : `{var}`")

    # Statistiques descriptives d'abord
    st.markdown("**Statistiques Descriptives Clés**")
    stats = df[var].describe().reset_index()
    stats.columns = ['Métrique', 'Valeur']
    st.dataframe(stats.style.format({'Valeur': '{:.2f}'}), use_container_width=True)

    st.markdown("---")
    st.markdown("**Visualisations**")

    # Configuration des graphiques
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Histogramme et Densité**")
        fig_hist = px.histogram(df, x=var, marginal="rug",
                                title=f'Distribution de {var}',
                                color_discrete_sequence=['#00ffff'],
                                opacity=0.7,
                                template=plotly_dark_template)
        fig_hist.update_layout(bargap=0.1, height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.markdown("**Boîte à Moustaches (Boxplot)**")
        fig_box = px.box(df, y=var, points="all",
                         title=f'Boîte à Moustaches de {var}',
                         color_discrete_sequence=['#00ff88'],
                         template=plotly_dark_template)
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)

    # Graphique de Violon sur toute la largeur
    st.markdown("**Graphique de Violon**")
    fig_violin = px.violin(df, y=var, box=True, points="all",
                           title=f'Distribution (Violon) de {var}',
                           color_discrete_sequence=['#4a5568'],
                           template=plotly_dark_template)
    fig_violin.update_layout(height=400)
    st.plotly_chart(fig_violin, use_container_width=True)

# Fonctions de visualisation pour variables qualitatives
def plot_qualitative(var):
    if var not in df.columns:
        st.error(f"La variable '{var}' n'existe pas dans le DataFrame.")
        return
    if pd.api.types.is_numeric_dtype(df[var]):
        st.warning(f"La variable '{var}' semble numérique mais est traitée comme qualitative.")
        df[var] = df[var].astype(str)

    st.subheader(f"Analyse de la variable qualitative : `{var}`")

    # Tableau de fréquences d'abord
    st.markdown("**Tableau de Fréquences**")
    counts = df[var].value_counts()
    percent = df[var].value_counts(normalize=True).map('{:.1%}'.format)
    freq_table = pd.DataFrame({'Effectif': counts, 'Pourcentage': percent})
    st.dataframe(freq_table, use_container_width=True)

    # Limiter le nombre de catégories pour les graphiques
    unique_count = df[var].nunique()
    if unique_count > 50:
        st.warning(f"La variable '{var}' a {unique_count} catégories uniques. Les graphiques pourraient être illisibles.")
        df_subset = df
    else:
        df_subset = df

    st.markdown("---")
    st.markdown("**Visualisations**")

    # Configuration des graphiques
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Diagramme à Barres**")
        counts_subset = df_subset[var].value_counts().reset_index()
        counts_subset.columns = [var, 'Count']
        fig_bar = px.bar(counts_subset, x=var, y='Count', color=var,
                         title=f'Distribution de {var}',
                         text='Count',
                         color_discrete_sequence=['#00ffff', '#00ff88', '#4a5568', '#ffaa00', '#ff4444'],
                         template=plotly_dark_template)
        fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
        fig_bar.update_layout(showlegend=False, height=400, xaxis_title=var, yaxis_title='Nombre d\'occurrences')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("**Diagramme Circulaire (Pie Chart)**")
        fig_pie = px.pie(df_subset, names=var, title=f'Proportions de {var}',
                         hole=0.3,
                         color_discrete_sequence=['#00ffff', '#00ff88', '#4a5568', '#ffaa00', '#ff4444'],
                         template=plotly_dark_template)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05]*len(df_subset[var].unique()))
        fig_pie.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Treemap sur toute la largeur si beaucoup de catégories
    if unique_count > 10:
        st.markdown("**Treemap**")
        fig_tree = px.treemap(df_subset, path=[px.Constant("Total"), var],
                              title=f'Répartition de {var} (Treemap)',
                              color=var,
                              color_discrete_sequence=['#00ffff', '#00ff88', '#4a5568', '#ffaa00', '#ff4444'],
                              template=plotly_dark_template)
        fig_tree.update_layout(height=500)
        st.plotly_chart(fig_tree, use_container_width=True)

# Affichage des graphiques selon le type de variable sélectionné
if selected_var:
    if var_type == "Quantitative":
        plot_quantitative(selected_var)
    else:
        plot_qualitative(selected_var)
else:
    st.info("Veuillez sélectionner une variable dans la barre latérale pour commencer l'analyse.")

# --- Export Section ---
st.sidebar.markdown("---")
st.sidebar.header("Export des Graphiques")
export_button = st.sidebar.button("Générer et Télécharger les Graphiques")

if export_button and selected_var:
    zip_buffer = BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            if var_type == "Quantitative":
                # 1. Histogramme
                fig_hist_export = px.histogram(df, x=selected_var, marginal="rug", title=f'Distribution de {selected_var}',
                                               color_discrete_sequence=['#00ffff'], template=plotly_dark_template)
                img_buf_hist = fig_hist_export.to_image(format="png", scale=2)
                zip_file.writestr(f"histogramme_{selected_var}.png", img_buf_hist)

                # 2. Boxplot
                fig_box_export = px.box(df, y=selected_var, points="all", title=f'Boîte à Moustaches de {selected_var}',
                                        color_discrete_sequence=['#00ff88'], template=plotly_dark_template)
                img_buf_box = fig_box_export.to_image(format="png", scale=2)
                zip_file.writestr(f"boxplot_{selected_var}.png", img_buf_box)

                # 3. Violon
                fig_violin_export = px.violin(df, y=selected_var, box=True, points="all", title=f'Distribution (Violon) de {selected_var}',
                                              color_discrete_sequence=['#4a5568'], template=plotly_dark_template)
                img_buf_violin = fig_violin_export.to_image(format="png", scale=2)
                zip_file.writestr(f"violon_{selected_var}.png", img_buf_violin)

            else:  # Qualitative
                counts_export = df[selected_var].value_counts().reset_index()
                counts_export.columns = [selected_var, 'Count']
                df_export = df

                # 1. Diagramme à barres
                fig_bar_export = px.bar(counts_export, x=selected_var, y='Count', color=selected_var, title=f'Distribution de {selected_var}',
                                        text='Count', color_discrete_sequence=['#00ffff', '#00ff88', '#4a5568', '#ffaa00', '#ff4444'],
                                        template=plotly_dark_template)
                fig_bar_export.update_traces(texttemplate='%{text}', textposition='outside')
                fig_bar_export.update_layout(showlegend=False)
                img_buf_bar = fig_bar_export.to_image(format="png", scale=2)
                zip_file.writestr(f"barres_{selected_var}.png", img_buf_bar)

                # 2. Diagramme circulaire
                fig_pie_export = px.pie(df_export, names=selected_var, title=f'Proportions de {selected_var}', hole=0.3,
                                        color_discrete_sequence=['#00ffff', '#00ff88', '#4a5568', '#ffaa00', '#ff4444'],
                                        template=plotly_dark_template)
                fig_pie_export.update_traces(textposition='inside', textinfo='percent+label')
                img_buf_pie = fig_pie_export.to_image(format="png", scale=2)
                zip_file.writestr(f"pie_{selected_var}.png", img_buf_pie)

                # 3. Treemap (si applicable)
                if df[selected_var].nunique() <= 50:
                    fig_tree_export = px.treemap(df_export, path=[px.Constant("Total"), selected_var], title=f'Répartition de {selected_var} (Treemap)',
                                                 color=selected_var, color_discrete_sequence=['#00ffff', '#00ff88', '#4a5568', '#ffaa00', '#ff4444'],
                                                 template=plotly_dark_template)
                    img_buf_tree = fig_tree_export.to_image(format="png", scale=2)
                    zip_file.writestr(f"treemap_{selected_var}.png", img_buf_tree)

        st.sidebar.success("Graphiques prêts pour le téléchargement !")
        st.sidebar.download_button(
            label="Télécharger le fichier ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"graphiques_{selected_var}.zip",
            mime="application/zip"
        )
    except ValueError as ve:
        if "kaleido" in str(ve):
            st.sidebar.error("Erreur d'exportation: Le package 'kaleido' est requis. Installez-le avec 'pip install kaleido'")
        else:
            st.sidebar.error(f"Erreur d'exportation d'image: {ve}")
    except Exception as e:
        st.sidebar.error(f"Une erreur est survenue lors de la création du ZIP: {e}")

elif export_button and not selected_var:
    st.sidebar.warning("Veuillez d'abord sélectionner une variable à visualiser.")

# Informations supplémentaires
st.sidebar.markdown("---")
st.sidebar.info("""
**Instructions:**
1. Importez vos données (CSV/Excel) ou utilisez les données de démo Iris.
2. Sélectionnez une variable à analyser dans la barre latérale.
3. Explorez les statistiques et visualisations générées.
4. Utilisez le bouton d'export pour télécharger les graphiques de la variable sélectionnée (nécessite `kaleido`).
""")

# Pied de page
st.markdown("---")
st.markdown("""
<div class="footer">
    © 2025 Statistics visualisation | Created with ❤️ using Streamlit & Plotly by Onesime
</div>
""", unsafe_allow_html=True)