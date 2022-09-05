import streamlit as st
from multiapp import MultiApp
from apps import home, mlmain # import your app modules here


app = MultiApp()


col1, mid, col2 = st.columns([1,1,20])
with col1:
    st.image('./newspaper.png', width=60)
with col2:
    st.markdown("""
# Detección de noticias falsas

""")



# Add all your application here
app.add_app("Inicio", home.app)
app.add_app("Modelo Machine Learning", mlmain.app)




# The main app
app.run()

###No solo detectar noticias falsas, si no noticias que busquen atacar a ciertos grupos de la sociedad