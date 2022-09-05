     contador = 0
 
    spell = Speller(lang='es')
    for x in cuerpo.split(" "):
        if(x == spell(x)):
           
            continue
        else:
            st.write(x)
            contador +=1
    st.write(contador) 
