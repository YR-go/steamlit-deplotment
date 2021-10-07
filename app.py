#app.py
'''
import app1
import app2
import streamlit as st

PAGES = {
    "App1":app1,
    "App2":app2
}

st.sidebar.title('Navigation임다')
selection = st.sidebar.radio("GO to 프로젝트를 골라", list(PAGES.keys()))
page = PAGES[selection]
page.app()

'''

import streamlit as st
from multiapp import MultiApp
from apps import home, data_stats, parkingLot, CafeAPP # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("HOME", home.app)
app.add_app("Data Analysis", data_stats.app)
app.add_app('AI Parking Lot System', parkingLot.app)
app.add_app("Cafe App with NFC", CafeAPP.app)
# The main app
app.run()