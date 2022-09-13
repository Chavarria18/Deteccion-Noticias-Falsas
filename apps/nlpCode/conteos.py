import time
from autocorrect import Speller 
from spellchecker import SpellChecker
import streamlit as st
from pyrae import dle
from textblob import TextBlob
import os
import re
spell2 = SpellChecker(language="es")
#aumentar el numero de palabras soeces


def verificarpalabra(palabra):
    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))    

   
    with open(os.path.join(__location__,'palabras_soeces.txt'), 'r') as file:
        raw = file.readlines()    
    flag = True
    for line in raw :       
        if palabra.lower() == line.lower().rstrip('\n'):            
            flag = True
            break
        else:
            flag = False
    return flag


def conteo(titulo, cuerpo):

    contadorf = 0
    contadorp = 0
 
    spell = Speller(lang='es')
    for x in cuerpo.split(" "):  
        x = x.replace(".","")  
        if(x != spell(x)):
            resultado = dle.search_by_word(word=x).to_dict()
            if resultado['title'] == "Diccionario de la lengua española | Edición del Tricentenario | RAE - ASALE":
                ###st.write(x+ "Puede contener una falta ortografica")
                time.sleep(2)
                contadorf +=1
        print(verificarpalabra(x))
        if(verificarpalabra(x)):
            st.write(x+ "Puede considerarse como una mala palabra")
            contadorp +=1       
                               

    
    st.write("Conteos de palabras:")
    col1, col2 = st.columns(2)    
    col1.metric("Soeces", contadorp)
    col2.metric("Faltas ortógraficos", contadorf)
            