"""
Codigo con la funcion para analizar el sentimiento de la noticia introducida
"""
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
#from sentiment_analysis_spanish import sentiment_analysis as se
from textblob import TextBlob
from autocorrect import Speller 
from annotated_text import annotated_text




###nltk.download('punkt')
###nltk.download('vader_lexicon')
def sentiment(cuerpo,titulo):
    sia = SentimentIntensityAnalyzer()
    blobcuerpo = TextBlob(cuerpo)
    blobtitulo = TextBlob(titulo)

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
        st.write("El analisis del sentimiento para el contenido  fue:"+str(analysis.sentiment) + str(sia.polarity_scores(cuerpotraducido)))
        
        words: list[str] = nltk.word_tokenize(cuerpotraducido)
        words2: list[str] = nltk.word_tokenize(cuerpo)
        text = nltk.Text(words)
        text2 = nltk.Text(words2)
        fd = text.vocab()
        fd2 = text2.vocab()
        fd.tabulate(3)
        fd2.tabulate(3)
       
        for x in fd:
            x2 = TextBlob(x)
            try:
                x2 = str(x2.translate(from_lang='en', to='es')) 
            except:
                continue
            """ 
            st.write(x2)
            x3 = TextBlob(x)
            st.write(sia.polarity_scores(x))  
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

        annotated_text(*text_notas) 
            
     
            

       