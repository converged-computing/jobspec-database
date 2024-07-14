import streamlit as st
from predict_page import show_predict_page
from predict_test_page import show_predict_test_page
from train_new_adapter_page import show_train_adapter_page

st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 275px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)
page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Predict testset", "Train new adapter"))

if page == "Predict":
    show_predict_page()
elif page == "Predict testset":
    show_predict_test_page()
else:
    show_train_adapter_page()
