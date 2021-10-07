import streamlit as st
from multiapp import MultiApp
from apps import home, data_stats, parkingLot, cafeAPP

app = MultiApp()

# Add all your application here
app.add_app("HOME", home.app)
app.add_app("Data Analysis", data_stats.app)
app.add_app('AI Parking Lot System', parkingLot.app)
app.add_app("Cafe App with NFC", cafeAPP.app)
# The main app
app.run()