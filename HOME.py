import streamlit as st

st.set_page_config(page_title="Sales Metrics",page_icon = r'./images/Orange_favicon.png', layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width: 500;}
    button[title="View fullscreen"]{visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True,)

col1, col2, col3 = st.columns([1, 5, 1])

with col1:
    st.markdown(' ')

with col2:
    orange_logo=r'./images/OBS_logo.png'
    st.markdown(' ')
    st.image(orange_logo, use_column_width='auto')
        

with col3:
    st.markdown(' ')