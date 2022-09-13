from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import random
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
import random 
import stanza
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import streamlit as st
import pyLDAvis 
import pyLDAvis.sklearn 
import seaborn as sns
import nltk
def tema(titulo,cuerpo):
    st.title("Modelado del Tema")
    st.subheader("Nube de palabras principales relacionadas con el tema")
    #tokens_without_sw = [word for word in word_tokenize(",".join(topic_words)) if not word in stopwords.words("spanish")]
    tokens_without_sw = [word for word in word_tokenize(cuerpo) if not word in stopwords.words("spanish")]
    wordcloud = WordCloud().generate(",".join(tokens_without_sw))
    tokens_without_sw2 = [word for word in word_tokenize(titulo.lower()) if not word in stopwords.words("spanish")]  
    wordcloud2 = WordCloud().generate(",".join(tokens_without_sw2))
    col1, col2 = st.columns(2)
    with col1:
        st.image(wordcloud.to_array())
    with col2:
        st.image(wordcloud2.to_array())  

    ###----> Palabras del tema 
  
    count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words=stopwords.words("spanish"))  
    doc_term_matrix = count_vect.fit_transform(word_tokenize(cuerpo))  
    LDA = LatentDirichletAllocation(n_components=5, random_state=42)  
    LDA.fit(doc_term_matrix)
    palabras_cuerpo = [] 
    for i,topic in enumerate(LDA.components_):  
        print(f'Top 10 palabras relacionadas al tema#{i}:')
        print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
        print('\n') 
        listado = [count_vect.get_feature_names_out()[i] for i in topic.argsort()[-10:]] 
        for x in listado:
            if(x not in palabras_cuerpo):
                palabras_cuerpo.append(x)  
   
    sns.set_style('darkgrid')  
    sns.set_style('ticks')                           
    palabras_cuerpo2 = [x.lower() for x in palabras_cuerpo]
    print(palabras_cuerpo2)
    print(word_tokenize(cuerpo))
    filter_words =  [x.lower() for x in word_tokenize(cuerpo) if x.lower()  in palabras_cuerpo2 ]
  
    nlp_words=nltk.FreqDist(filter_words)  
    fig =nlp_words.plot(20,title='Frecuencia palabras relacionadas al tema de la noticia')
    fig.set_xlabel("Palabras")
    fig.set_ylabel("Conteo")
    st.pyplot(fig.figure)
   
   



 
        


   

   

    """ 
    palabras_titulo = []
    count_vect2 = CountVectorizer(max_df=0.5, stop_words=stopwords.words("spanish"))  
    doc_term_matrix2 = count_vect2.fit_transform(word_tokenize(titulo))  
    LDA2 = LatentDirichletAllocation(n_components=5, random_state=42)  
    LDA2.fit(doc_term_matrix2)
    for i,topic in enumerate(LDA.components_):  
        print(f'Top 10 palabras relacionadas al tema del titulo#{i}:')
        print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
        print('\n')
    print(sent_tokenize(cuerpo))    
    ##max_df=0.9 , min_df=10, stop_words=stopwords.words("spanish")
    count_vect = CountVectorizer(stop_words=stopwords.words("spanish")) 
    doc_term_matrix = count_vect.fit_transform(sent_tokenize(cuerpo))     
    
    LDA = LatentDirichletAllocation(n_components=5, random_state=42)  
    LDA.fit(doc_term_matrix)  
    first_topic = LDA.components_[0]  
    print(first_topic)
  

    
    for i in range(10):  
        random_id = random.randint(0,len(count_vect.get_feature_names()))
        print(count_vect.get_feature_names()[random_id])

        top_topic_words = first_topic.argsort()[-10:] 
        for i,topic in enumerate(LDA.components_):  
            print(f'Top 10 words for topic #{i}:')
            print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
            print('\n')
    """
