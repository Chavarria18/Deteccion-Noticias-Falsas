import streamlit as st 
import nltk
import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from streamlit_metrics import metric, metric_row
import altair as alt
import seaborn as sn
from sklearn.metrics import plot_confusion_matrix
def app():
    ###Pagina web 
    st.title('Modelos de clasificación')
    st.write('Para el entrenamiento del clasificador se utilizo el siguiente dataset, obtuvido del [Github](https://github.com/jpposadas/FakeNewsCorpusSpanish) ')
   
    

    ####Modelo 
    df= pd.read_csv('./data-final.csv',sep=';',encoding= 'unicode_escape')
    
    print(df.shape)
    st.write(df) 
    st.write('Shape of dataset:', df.shape)



    df.loc[(df['Category'] == 0) , ['Category']] = 'Fake'
    df.loc[(df['Category'] == 1) , ['Category']] = 'True'

   ### print(df.head())

    labels = df.Category

    ###dividir el conjunto de datos en dos sets uno para entrenamiento y otro para probar/test
    x_train,x_test,y_train,y_test=train_test_split(df['Text'].values.astype('str'), labels, test_size=0.27884615384,train_size=0.72115384615, random_state=0)

    ###Quitar las stopwords 
    ###nltk.download('stopwords')
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words("spanish"), max_df=0.7)

    ###se transforma en un conjutno de entrenamiento y pruebas
    tfidf_train=tfidf_vectorizer.fit_transform(x_train)
    tfidf_test=tfidf_vectorizer.transform(x_test)

    #se inicializa PassiveAggressiveClassifier 
    pa_classifier=PassiveAggressiveClassifier(max_iter=50)
    pa_classifier.fit(tfidf_train,y_train)

    #Se inicializa el RandomForestClassifier
    randomforest = RandomForestClassifier(max_depth=2, random_state=0)
    randomforest.fit(tfidf_train,y_train)
    #Se inicaliza el SVM classifier
    svmclasf = svm.SVC(probability=True)
    svmclasf.fit(tfidf_train,y_train)


    #####----------------
    ###Acurracy Score
    #####--------------- 
    ###Se realizan las pruebas de los clasificadores
    ###--->Passive agressive 
    y_pred=pa_classifier.predict(tfidf_test)
    score=accuracy_score(y_test,y_pred)
    
    ###--->Random Forest
    y_pred2=randomforest.predict(tfidf_test)
    score2=accuracy_score(y_test,y_pred2)
    ###--->SVM
    y_pred3=svmclasf.predict(tfidf_test)
    score3=accuracy_score(y_test,y_pred3) 
    

  


    ###Mostrar los resultados de cada uno de los entrenamientos y pruebas
    classifier_name = st.selectbox(
        'Escoge el algoritmo de clasificación',
        ('Passive Aggresive', 'SVM', 'Random Forest')
    )
    st.write("Reporte de metricas de desempeño")
    if classifier_name == 'Passive Aggresive':
       
        st.metric("Accuracy", str( round(score,2)*100) +"%")       
       
        clf_rep = classification_report(y_test, y_pred,output_dict = True)
        clf_rep_df = pd.DataFrame(clf_rep)
        st.write(clf_rep_df.T)    
        st.write("Matriz de confusion; 0 es Falsa y 1 es Verdadera")
        clf_matrix = confusion_matrix(y_test, y_pred)
        clf_matrix_df = pd.DataFrame(clf_matrix)
        st.write(clf_matrix_df.T)    

   
      

    if classifier_name == 'Random Forest':
        st.metric("Accuracy", str( round(score2,2)*100) +"%") 

   
        clf_rep = classification_report(y_test, y_pred2,output_dict = True)
        clf_rep_df = pd.DataFrame(clf_rep)
        st.write(clf_rep_df.T)
        st.write("Matriz de confusion; 0 es Falsa y 1 es Verdadera")
        clf_matrix = confusion_matrix(y_test, y_pred2)
        clf_matrix_df = pd.DataFrame(clf_matrix)
        st.write(clf_matrix_df.T)  
    if classifier_name == 'SVM':
        st.metric("Accuracy", str( round(score3,2)*100) +"%")                  

        clf_rep = classification_report(y_test, y_pred3,output_dict = True)
        clf_rep_df = pd.DataFrame(clf_rep)
        st.write(clf_rep_df.T)
        st.write("Matriz de confusion; 0 es Falsa y 1 es Verdadera")
        clf_matrix = confusion_matrix(y_test, y_pred3)
        clf_matrix_df = pd.DataFrame(clf_matrix)
        st.write(clf_matrix_df.T)  
  
    st.write("Comparación resultados desempeño algoritmos")
    
    source = pd.DataFrame({
        'Exactitud (%)': [score, score2, score3],
        'Clasificador': ['Passive Agressive', 'Random Forest', 'Support Vector Machine']
     })
 
    bar_chart = alt.Chart(source).mark_bar().encode(
        y='Exactitud (%)',
        x='Clasificador',
    )
 
    st.altair_chart(bar_chart, use_container_width=True)
   

   




