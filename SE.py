# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 21:18:40 2021

@author: utilisateur
"""

# Bibliotheques utilisés:
import pandas as pd
import numpy as np
import streamlit as st
import os #Miscellaneous operating system interfaces
#import cv2 #import OpenCV
from sklearn import metrics
import matplotlib.pyplot as plt # Pour l'affichage d'images
from matplotlib import cm # Pour importer de nouvelles cartes de couleur
import itertools # Pour créer des iterateurs
import streamlit as st
import pydot
import graphviz

from PIL import Image

from sklearn.metrics import classification_report

st.set_option('deprecation.showPyplotGlobalUse', False)

###################
#PAGE CONFIGURATION
###################

st.set_page_config(page_title="Projet Rakuten", 
                   page_icon=":robot_face:",
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )

#########
#SIDEBAR
########
new_title = '<p style="font-family:sans-serif; color:RED; font-size: 42px;">RAKUTEN</p>'
st.sidebar.title(new_title,"Projet Rakuten")
st.sidebar.write('')

st.sidebar.markdown("* Promotion Décembre 2020")
st.sidebar.markdown("## Participants ")
st.sidebar.markdown('Fatoumata Barry')
st.sidebar.markdown("Emmanuel Bonnet")
st.sidebar.markdown("Edgar Hidalgo")
st.sidebar.markdown("Eric Marchand")
st.sidebar.markdown('')
st.sidebar.markdown('')

st.sidebar.markdown("### ** Sommaire **")

navigation = st.sidebar.radio('',["Introduction", "DataViz", "Méthode",  "Bilan"])

#CONTACT
########
expander = st.sidebar.expander('SUPPORT')
expander.write("Lien GitHub : https://github.com/Ragdehl/Rakuten_py/tree/main/Livrables/It%C3%A9ration_4 ")

classes = {
                #Livres
                '10':' Livres type romain, Couvertures de livres ',
               '2280':' Livres, journaux et revues anciennes',
               '2403':' Livres, BD et revues de collection',
               '2705':' Livres en général',
               '2522':' Cahiers, carnets, marque pages',
    
                #Jeux
               '40':' Jeux videos, CDs + mais aussi equipements, cables, etc. ',
               '50':' Equipements/complements consoles, gamers ',
               '2905':' Jeux vidéos pour PC',
               '2462':' Equipement jeux, jeux video, play stations',
               '60':' Consoles ',
    
                #Jouets & Figurines
               '1280':' Jouets pour enfants, poupées nounours, equipements enfants',
               '1281':' Jeux socitété pour enfants, Boites et autres, couleurs flashy',
               '1300':' Jeux techniques, Voitures/drones télécomandés, Equipement, petites machines ',
               '1180':' Figurines et boites ',   
               '1140':' Figurines, Personnages et objets, parfois dans des boites ',
                '1160':' Cartes collectionables, Rectangles, beaucoup de couleurs ',
               
                #Meubles
               '1320':' Matériel et meubles bébé poussettes, habits',
               '1560':' Meubles, matelas canapés lampes, chaises',
        
                #Equipements
                '2582':' Matériel, meubles et outils pour le jardin',
               '2583':' Equipements technique pour la maison et exterieur (piscines), produits',
               '2585':' Idem 2583:  Equipements technique pour la maison et exterieur (piscines), produits',
                '1302':' Equipements, Habits, outils, jouets, objets sur fond blanc',
                '2220':' Equipements divers pour animaux',
        
                #Déco
               '1920':' Oreillers, coussins, draps',
               '2060':' Décorations',
        
                #Autre
                '1301':' Chaussetes bébés, petites photos ',
               '1940':' Alimentations, conserves boites d gateaux',
    
              }

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nltk
from nltk import word_tokenize
import re
import string

nltk.download('punkt')
from nltk.corpus import stopwords
import unicodedata
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt 
from nltk.stem.snowball import FrenchStemmer
nltk.download('stopwords')
import string

if navigation == "Introduction":
    st.title("Présentation du projet")
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .big-font2 {
        font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)
 
    st.markdown('<p class="big-font2">Les techniques de classification « image plus texte » sont fortement sollicitées par les entreprises de e-commerce notamment pour: </p>', unsafe_allow_html=True)	
         
    st.markdown('<p class="big-font2"> La classification de produit à grande échelle pour mieux gérer l’offre d’un site </p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">  - La recommandation de produit génère en moyenne 35% de revenu supplémentaire </p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">  - L’amélioration de l’expérience client </p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">  Ce projet s’inscrit ainsi dans le challenge Rakuten France Multimodal Product Data Classification. Ce dernier requiert de prédire le code type des produits à partir d’une description texte et d’une image.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font2">  Un modèle de référence est indiqué par le site du challenge: l’objectif du projet est de faire mieux.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font2">  Cette référence est en réalité composée de deux modèles distincts : un pour les images, un pour le texte :</p>', unsafe_allow_html=True)
                   
    st.markdown('<p class="big-font">  1.	Pour les données images, une version du modèle Residual Networks (ResNet), le ResNet50 pré-entraîné avec un jeu de données Imagenet;</p>', unsafe_allow_html=True) 
    st.markdown('<p class="big-font">  27 couches différentes du haut sont dégelées, dont 8 couches de convolution pour l''entraînement. </p>', unsafe_allow_html=True) 
    st.markdown('<p class="big-font">  2.	Pour les données textes, un classificateur RNN simplifié est utilisé. Seuls les champs de désignation sont utilisés dans ce modèle de référence. </p>', unsafe_allow_html=True) 
                    
    st.markdown('<p class="big-font"> Les données appartiennent à 27 classes distinctes. Pour évaluer la qualité de la classification, il est demandé d’utiliser la métrique weighted-F1-score.</p>', unsafe_allow_html=True) 
    st.markdown('<p class="big-font"> Il s’agit de la moyenne des F1-scores de toutes les classes pondérées par le nombre de représentants dans ces classes. Le F1-score de chaque classe est la .</p>', unsafe_allow_html=True) 
    st.markdown('<p class="big-font"> moyenne harmonique de la précision et du rappel pour cette classe.</p>', unsafe_allow_html=True) 
    st.markdown('<p class="big-font"> Le modèle de référence obtient les résultats suivants : </p>', unsafe_allow_html=True) 
    st.markdown('<p class="big-font2">  -0.5534 pour le modèle image (ResNet) </p>', unsafe_allow_html=True) 
    st.markdown('<p class="big-font2">  -0.8113 pour le modèle texte (RNN)</p>', unsafe_allow_html=True)
                          
                                    
    
if navigation == "DataViz" :
     # os.chdir('C:\\Users\\utilisateur\\Documents\\PROJET RAKUTEN')

     #X = pd.read_csv('X_train_update.csv',index_col=0)

     #y = pd.read_csv('Y_train_CVw08PX.csv',index_col=0)

     df = pd.read_csv('X_train_update.csv',index_col=0)

     designation = df['designation']
     description = df['description']
     # ETUDE DES NAN'S
     st.header('Observation of NAN values')     
     figW=plt.figure() 
     sns.heatmap(df.isna(),cmap="coolwarm", center = 10.0, cbar=False) #(bins = 100, color='gray')    
     st.pyplot (figW)
     
     if st.button('Show DATA Designation'):
         st.write(designation)
         
     if st.button('Show DATA Description'):    
         st.write(description)
     
     st.header('Length of Words for RAKUTEN DATASET FEATURES')
     st.write()
    
     # ETUDE DE LA COLONNE DESIGNATION
     
     df['designation'] = df['designation'].apply(lambda _: str(_))
     words_per_review = df.designation.apply(lambda x: len(x.split(" ")))
         
     if st.button('Show Plot Designation'):
         st.header('Length of words for Designation')     
         figx=plt.figure() 
         sns.histplot (words_per_review, kde=True,binwidth = 10, color='black') #(bins = 100, color='gray')
         plt.xlim(0,200)    
         st.pyplot (figx)
          
     df['description'] = df['description'].apply(lambda _: str(_))
     words_per_review2 = df.description.apply(lambda x: len(x.split(" ")))
     
 
     if st.button('Show Plot Description'):
         st.header('Length of words for Description') 
         figy=plt.figure() 
         sns.histplot (words_per_review2, kde=True, binwidth = 10, color='black') #(bins = 100, color='gray')
         plt.xlim(0,200)    
         st.pyplot (figy)
          
        
     target = pd.read_csv('Y_train_CVw08PX.csv',index_col=0)
     st.header('Study of target RAKUTEN ')
     st.write() 
     df['target'] = target 
     
     if st.button('Show Plot Target_Bar'):
         figz2=plt.figure() 
         percent_target = 100 * df['target'].value_counts()/len(df)
         percent_target.plot.bar(color='gray')
         sns.set_palette('Accent')
         font = {'family' : 'normal',
         'weight' : 'bold',
         'size'   : 14}            
         plt.rc('font', **font)           
         plt.title("Barplot of different product codes")
         plt.ylabel('% of all targets')
         st.pyplot (figz2)
                   
     if st.button('Show Plot Target_Pie'):
         figa=plt.figure(figsize=(18,12)) 
         df['target'].value_counts().plot.pie()  
         plt.xticks(rotation=45)
         st.pyplot (figa)
       
     




if navigation == "Méthode" :
    #os.chdir('C:\\Users\\utilisateur\\Documents\\PROJET RAKUTEN\\Demo')
    #from rakuten_demo import *
    #cat = CatModel()
    st.title('Méthode')
    st.title('')
    st.title('')

    st.write('''
Afin de résoudre le problème de classification plusieurs étapes ont été réalisées.
             
Tout d'abord les différents types de données ont été traitées à partir de modèles de type :
- ConvNet, pour les données images
 - RNN et Machine Learning, pour les données textes.
            
Ensuite, après plusieurs tentatives de paramétrisation et d'hyperparamétrisation, les meilleures modèles et versions de modèle ont été intégrés dans un modèle de concaténation (cf. Graph 1 et Graph 2).
             
Remarque : Le modèle de type concaténé présente l'avantage d'intégrer à la fois les données images et données textes pour prédire une même cible finale. Ce qui permet d'enrichir le modèle et d'améliorer les résultats.
             ''')

    st.title('')
    #tf.keras.utils.plot_model(cat.model, show_shapes = True, show_layer_names = True)
    st.subheader("Graph 1 : Architecture du modèle - version manuelle")
    image = Image.open('C:\\Users\\utilisateur\\Documents\\PROJET RAKUTEN\\mod_bleu_conc.png')
    st.image(image, 300);
    
    st.title('')
    #tf.keras.utils.plot_model(cat.model, show_shapes = True, show_layer_names = True)
    st.subheader("Graph 2 : Architecture du modèle - version logiciel")
    image = Image.open('C:\\Users\\utilisateur\\Documents\\PROJET RAKUTEN\\architecture.png')
    st.image(image, 300);
    
if navigation == "Bilan" :
    st.markdown("""
    <style>
    .big-font3 {
        font-size:25px !important;
    }
    </style>
    """, unsafe_allow_html=True)
         
    st.markdown('<p class="big-font3"> Les techniques de classification « image plus texte » sont fortement sollicitées par les entreprises de e-commerce notamment pour: </p>', unsafe_allow_html=True)	
    st.markdown('<p class="big-font3"> Les résultats de scoring f1 weighted  obtenus en interne, à la date de diffusion du rapport sont de *0.8783*.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font3"> Les prédictions soumises le 9 août 2021 sur le site du challenge ont permis à notre FEEEScientest d''atteindre un score de *0.8628*.</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font3"> L''équipe est par ailleurs classée 9ème.</p>', unsafe_allow_html=True)
    
    st.title('')
    st.subheader("Classement - site du challenge")
    image = Image.open('C:\\Users\\utilisateur\\Documents\\PROJET RAKUTEN\\classement.png')
    st.image(image, 300);