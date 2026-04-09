import streamlit as st
import sys
import os

# Set current directory to path
sys.path.append(os.getcwd())

# Import the main dashboard from Market_Regime
from Market_Regime import dashboard

if __name__ == "__main__":
    dashboard()
