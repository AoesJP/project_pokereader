from interface.streamlit_app import main
import streamlit as st
main()

footer="""<style>
.footer {
font-size: 1px;
left: 0;
bottom: 0;
width: 100%;
text-align: center;
margin-top: 140px;
}
</style>
<div class="footer">
<p>Brought to you by <br>Alex C, Emilia S, Estelle G, Yuri S</br></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
