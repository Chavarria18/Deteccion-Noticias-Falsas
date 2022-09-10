noticia = """

Después de uno de sus días más complicados, Isabel II ya se encuentra en su despacho del Palacio de Buckingham trabajando a pleno rendimiento tras recuperarse de su muerte. “La de ayer fue la primera vez en 70 años de reinado que la monarca incumple su horario laboral, pero la muerte la dejó totalmente indispuesta durante la tarde y hasta altas horas de la noche”, han explicado los portavoces de los Windsor.

Según ha podido saber la prensa, la Reina de Inglaterra se ha levantado bastante cansada, pero tras tomarse un té y unas galletas ha ido cogiendo buen color y ahora está mejor que nunca. “La muerte de ayer fue un susto importante, pero por suerte la Reina ya se encuentra perfectamente”, tranquilizan desde Buckingham Palace. Isabel II se pasará todo el día tratando de recuperar el trabajo perdido en el día de ayer.

Solo interrumpirá sus labores cuando, esta misma tarde, ofrezca una rueda de prensa para agradecer las muestras de apoyo. “Pedirá a la gente que se ha congregado a las puertas del palacio que haga menos ruido porque así no hay manera de concentrarse para trabajar”, explican fuentes oficiales. Tras sufrir una muerte en el día de ayer, Isabel II ha prometido que se cuidará más para poder seguir con su reinado sin más sustos.

La prensa inglesa también se ha hecho eco de un momento incómodo que se ha vivido en el trono de Inglaterra. Los hechos se produjeron cuando Carlos de Gales bajó a la cámara a sentarse en el trono y se encontró en él a su madre, algo que generó una conversación bastante embarazosa entre ellos.



"""

titulo = """

Isabel II ya está en su despacho trabajando a pleno rendimiento tras recuperarse de su muerte
"""

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
import random 
import stanza
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

stanza.download('es')
nlp = stanza.Pipeline('es')
stemmer = SnowballStemmer('spanish')
stemmed_text = [stemmer.stem(i) for i in word_tokenize(noticia)]
##print(stemmed_text)


count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words=stopwords.words("spanish"))  
print(count_vect)
doc_term_matrix = count_vect.fit_transform(word_tokenize(noticia))  
from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=5, random_state=42)  
LDA.fit(doc_term_matrix) 
first_topic = LDA.components_[0]  

print(LDA.components_[0])

topic_words = [] 


topics= LDA.components_[0]
     



for i,topic in enumerate(LDA.components_[0]):    
    for x in count_vect.get_feature_names_out():
        print("algo")
        ###print(nlp(x))
       

        
        


#tokens_without_sw = [word for word in word_tokenize(",".join(topic_words)) if not word in stopwords.words("spanish")]
tokens_without_sw = [word for word in word_tokenize(noticia) if not word in stopwords.words("spanish")]


print(tokens_without_sw)



wordcloud = WordCloud().generate(",".join(tokens_without_sw))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



tokens_without_sw = [word for word in word_tokenize(titulo.lower()) if not word in stopwords.words("spanish")]
print(tokens_without_sw)
wordcloud = WordCloud().generate(",".join(tokens_without_sw))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()




