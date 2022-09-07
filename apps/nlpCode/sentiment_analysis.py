"""
Codigo con la funcion para analizar el sentimiento de la noticia introducida
"""
from random import seed
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentiment_analysis_spanish import sentiment_analysis as se
from textblob import TextBlob
from autocorrect import Speller 
from annotated_text import annotated_text
from pysentimiento import analyzer



###nltk.download('punkt')
###nltk.download('vader_lexicon')
def sentiment(cuerpo,titulo):
    
    analisador = analyzer.SentimentAnalyzer(lang='es')
    analisador2 = se.SentimentAnalysisSpanish()
    sia = SentimentIntensityAnalyzer()
    blobcuerpo = TextBlob(cuerpo)
    blobtitulo = TextBlob(titulo)
    st.title("Resultado analisis del sentimiento")
    
    
    ###Analisis sentimiento para el titulo 
    if(titulo == ""):
        st.write("Porfavor introducir el titulo de la noticia")
    if(titulo != ""):
        try:
            titulotraducido = str(blobtitulo.translate(from_lang='es', to='en'))  
            analysistitulo = TextBlob(titulotraducido)
            st.write("El analisis del sentimiento para el titulo:"+ str(analysistitulo.sentiment))
            
        except:
            st.write("No se puedo analizar el sentimiento del titulo:" + str(blobtitulo))


    ###Analisis sentimiento para el cuerpo
    if(cuerpo == ""):
        st.write("Porfavor introducir contenido de la noticia")
    if(cuerpo != ""):
        cuerpotraducido = str(blobcuerpo.translate(from_lang='es', to='en'))     
    
        analysis = TextBlob(cuerpotraducido)       
        st.write("El analisis del sentimiento para el contenido  fue:")
        col1, col2 = st.columns(2)
       
        col1.metric("Polaridad", str( round(analysis.sentiment[0], 2)*100) +"%")
        col2.metric("Subjetividad",  str( round(analysis.sentiment[1] ,2)*100) +"%")  
        col1, col2,col3 = st.columns(3)
        st.write("El analisis del sentimiento para el contenido  fue:")
        col1.metric("% Negativo", str( round( (sia.polarity_scores(cuerpotraducido)['neg']*100), 2)) +"%")
        col2.metric("% Positivo", str( round( (sia.polarity_scores(cuerpotraducido)['pos']*100), 2)) +"%")
        col3.metric("% Compuesta", str( round( (sia.polarity_scores(cuerpotraducido)['compound']*100), 2) ) +"%")
  


     
       





        
        words: list[str] = nltk.word_tokenize(cuerpotraducido)
        words2: list[str] = nltk.word_tokenize(cuerpo)
        text = nltk.Text(words)
        text2 = nltk.Text(words2)
        fd = text.vocab()
        fd2 = text2.vocab()
        fd.tabulate(3)
        fd2.tabulate(3)
       
        """ 
      
        text_notas = []
        
        for x in words:
            x2 = TextBlob(x)
            try:
                x2 = str(x2.translate(from_lang='en', to='es')) 
            except:
                continue
          
            if(sia.polarity_scores(x)['neg'] ==1 ):
                text_notas.append((x2+" ","Negativo "))
            elif(sia.polarity_scores(x)['pos'] ==1 ):
                text_notas.append((x2+" ","Positivo "))
            else:
                 text_notas.append(x2+" ")

        st.write("Palabras positivas y negativas:")
        annotated_text(*text_notas) 
        """
        text_notas2 = []
        ##probas={POS: 0.994, NEG: 0.003, NEU: 0.003})
        """ 
        for x in cuerpo.split(" "):           
            if(analisador.predict(x).output =='NEG'):
                text_notas2.append((x+" ","Negativo "))
            elif(analisador.predict(x).output =='POS'):
                text_notas2.append((x+" ","Positivo "))
            else:
                text_notas2.append((x+" "))
        """
        for x in cuerpo.split(" "):
            #st.write(x)
            #st.write(analisador2.sentiment(x))  
            if(analisador2.sentiment(x) <0.1):
                text_notas2.append((x+" ","Negativo "))
            if(analisador2.sentiment(x) > 0.1 and analisador2.sentiment(x) <0.70):
                text_notas2.append((x+" "))
            if(analisador2.sentiment(x) >0.8):
                text_notas2.append((x+" ","Positivo "))

            
             
            
        
        st.write("Palabras positivas y negativas:")
        annotated_text(*text_notas2) 

       
     
            

       