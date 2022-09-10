
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from autocorrect import Speller

from apps.nlpCode.conteos import conteo


from .nlpCode.sentiment_analysis import sentiment
from .nlpCode.machine_model import machine_analysis
from .nlpCode.conteos import conteo
from .nlpCode.topic_modeling import tema

def app():
    st.title('Inicio')
    titulo = st.text_area('Titulo de la noticia', '')
    cuerpo = st.text_area('Cuerpo de la noticia', '')
    buton = st.button('ANALIZAR')

    if buton:
        ##machine_analysis(cuerpo)
        ##sentiment(cuerpo,titulo)
        ##conteo(titulo,cuerpo)
        tema(titulo,cuerpo)
  

    
   
    """ 
    mask = np.array(Image.open("../"))
    mask[mask == 1] = 255

    wordcloud = WordCloud(background_color = "white", stopwords = exclure_mots, max_words = 50, mask = mask).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show();*/
    """ 

  


  
 