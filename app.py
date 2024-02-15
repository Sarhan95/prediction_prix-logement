import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import joblib
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
####### importation des donnes
data=pd.read_csv("Housing.csv")
data_presproc=pd.read_csv("df_preprocess.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

####### creation une application streamlit
st.subheader("prediction prix d'un logement par des modéles machine learnig")
st.title("auteur: zakaria sarhan")
st.markdown(" projet Machine learning")


st.sidebar.title("sommaire") 
pages=["contexte de project","exploration des données","visualisation des données","modélisation","prédiction des nouvelles données"] 
page=st.sidebar.radio("aller vers les pages",pages)
if page==pages[0]:
    st.image("immobilier.jpg")
    st.write("Prédire le prix des logements avec l'apprentissage automatique (machine learning) est une tâche courante et peut être abordée de différentes manières en fonction des données disponibles")
    st.write("Collecte de données :Rassemblez un ensemble de données comprenant des caractéristiques pertinentes pour prédire le prix des logements, telles que la superficie, le nombre de chambres, l'emplacement, les équipements, etc. Ces données doivent inclure à la fois les caractéristiques variables indépendantes et les prix des logements variable dépendante")
    st.write("Exploration et prétraitement des données :nalysez les données pour comprendre les tendances, les corrélations et les éventuels problèmes.raitez les valeurs manquantes, éliminez les doublons, et normalisez ou standardisez les données si nécessaire")
    st.write("choix de modél Sélectionnez un modèle d'apprentissage automatique adapté à la tâche de prédiction de prix. Les modèles couramment utilisés pour la régression (prédiction de valeurs continues) incluent les forêts d'arbres décisionnels, les réseaux de neurones, les machines à vecteurs de support (SVM), etc")
    st.write("Division des données :Divisez vos données en ensembles d'entraînement et de test. L'ensemble d'entraînement est utilisé pour former le modèle, tandis que l'ensemble de test est utilisé pour évaluer ses performances.Entraînement du modèl")
    st.write("Utilisez l'ensemble de test pour évaluer les performances du modèle en termes d'erreurs de prédiction. Des métriques telles que l'erreur quadratique moyenne (RMSE) ou l'erreur absolue moyenne (MAE) peuvent être utilisées")
    st.write("Optimisation du modèle :Si nécessaire, ajustez les hyperparamètres du modèle pour améliorer les performances. Cela peut impliquer l'utilisation de techniques telles que la validation croisée.")

if page==pages[1]:
    st.write("exploration data")
    @st.cache_data(persist=True)
    def load_data():
        data=pd.read_csv("Housing.csv")
        return data 
    df=load_data()
    if st.sidebar.checkbox("afficher data",False):
        st.subheader("base de données des logement immobilier")
    st.write(df)
    if st.checkbox("afficher les valeurs manquantes"):
        st.write(df.isna().sum())
    st.write("dimension de notre jeu de donnée:")
    st.write(df.shape)
    st.write("statistique descriptive")
    st.write(df.describe())

elif page==pages[2]:
    st.write("data visualisation")
    if st.checkbox("afficher les graphique"):
        st.title("analyse univarié quantitative")
        for col in data.select_dtypes(exclude="object"):
            plt.subplots()
            sns.histplot(data[col])
            st.pyplot()
            plt.show()
    st.title("analyse univariée variable qualitative")  
    if st.checkbox("afficher les graphique pour variable qualitative"):
        for col in data.select_dtypes("object"):
            val_cont=data[col].value_counts()
            plt.figure(figsize=(10,6))
            sns.barplot(x=val_cont.index,y=val_cont,palette='viridis')
            plt.title(f'Barplot for {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            st.pyplot()
            plt.show()
            st.write(val_cont)
            
    if st.checkbox("relation la variable avec les variables explicatives"):
        st.title('relation prix avec les variables explicative')      
        for col in list(data.select_dtypes("object")):
            st.write(data.groupby(col)["price"].mean().sort_values(ascending=False))
            sns.boxplot(data=data,y=col,x="price",palette='viridis')
            st.pyplot()
            plt.show()
    if st.checkbox("etude correlation entre les variables"):
        st.title('analyse bivariée')
        df_corr=data.select_dtypes(include=("float","int"))
        data_cor=df_corr.corr(method="pearson")
        fig,ax=plt.subplots()
        sns.heatmap(data_cor,annot=True,fmt=".2",cmap='viridis')
        st.pyplot(fig)


elif page==pages[3]:
    st.subheader(" modelisation")
    st.title("entrainement model")
    @st.cache_data(persist=True)
    @st.cache(allow_output_mutation=True) 
    def split_df(data):
        data_presproc=pd.read_csv("df_preprocess.csv")
        x=data.drop("price",axis=1)
        y=data["price"]
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
                                                       
        
        return x_train.copy(),x_test.copy(),y_train.copy(),y_test.copy()
    x_train,x_test,y_train,y_test=split_df(data_presproc)
    st.write("Données divisées avec succès !")
    x=data_presproc.drop("price",axis=1)

    num=list(x.select_dtypes(exclude="object"))
    stand=StandardScaler()
    x_train[num]=stand.fit_transform(x_train[num])
    x_test[num]=stand.transform(x_test[num])
    
    classifier=st.sidebar.selectbox("regression",("regression linear","random forest","knieghbhors regressor")
    )

    if classifier=="random forest":
        st.sidebar.subheader("hyper parametre de model")
        max_depthe=st.sidebar.number_input("choisir le nobmre de langueur de arbre",5,30,step=3)
        min_samples_splite=st.sidebar.number_input("choisir le nobmre d'echantillion",2,10,step=1)
        min_samples_leafe=st.sidebar.number_input("choisir le nobmre d'echantillion",1,10)
        ######## partie entrainement model
        random_forest=joblib.load("model_final.joblib")
        if st.sidebar.button("excution",key="regression"):
            st.subheader("random forest resultat")
        rd_forest=RandomForestRegressor(random_state=123,
                                        max_depth=max_depthe,
                                        min_samples_split=min_samples_splite,
                                        min_samples_leaf=min_samples_leafe


        )
        rd_forest.fit(x_train,y_train)
        y_rd=rd_forest.predict(x_test)
        score=np.round(mean_squared_error(y_test,y_rd,squared=False),2)
        coif_det=np.round(r2_score(y_test,y_rd),2)
        st.subheader("performance de model")
        st.write("score des erreur:",score)
        st.write("coeficient de detremination:",coif_det)
       

       ########### regression linear pour la suite a bientot zikooo
    if classifier=="regression linear":
        st.subheader("regression linear multipe")
        if st.sidebar.button("excution",key="regression"):
            st.subheader("regression linear resultat")
            rg_ln=LinearRegression()
            rg_ln.fit(x_train,y_train)
            y_reg=rg_ln.predict(x_test)
            score=np.round(mean_squared_error(y_test,y_reg,squared=False),2)
            coif_det=np.round(r2_score(y_test,y_reg),2)
            st.subheader("performance de model")
            st.write("score des erreur:",score)
            st.write("coeficient de detremination:",coif_det)

    if classifier=="knieghbhors regressor":
        st.sidebar.subheader("les hyper parametre")
        n_neighbor=st.sidebar.number_input("le nombre de voisin",1,10,step=1)
        poid=st.sidebar.radio("distrubition des point",("uniform","distance"))

        if st.sidebar.button("excution",key="regression"):
            st.subheader("Kneigbhors regression resultat")

            KNN=KNeighborsRegressor(
                n_neighbors=n_neighbor,
                weights=poid
            )
            KNN.fit(x_train,y_train)
            y_k=KNN.predict(x_test)
            score=np.round(mean_squared_error(y_test,y_k,squared=False),2)
            coif_det=np.round(r2_score(y_test,y_k),2)
            st.subheader("performance de model")
            st.write("score des erreur:",score)
            st.write("coeficient de detremination:",coif_det)
elif page==pages[4]:
    st.subheader("prediction sur des nouvelles données")

    
    
    



        
       
        

        




       
                

            
    
    

    
    
    
        
    
    
   

    
    
    






 