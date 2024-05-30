import streamlit as st
from interface.streamlit_app import main

main()

import streamlit as st

footer="""<style>
.footer {
position: fixed;
font-size: 1px;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Brought to you by <br>Alex Chiu, Emilia Sato, Estelle Gauquelin, Yuri Serizawa</br></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
