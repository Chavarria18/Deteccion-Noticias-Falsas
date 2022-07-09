mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"chavarria181386@unis.edu.gt\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml