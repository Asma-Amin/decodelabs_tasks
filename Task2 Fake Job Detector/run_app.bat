@echo off
REM One-click launcher for Windows (run from the project folder)
if not exist venv (python -m venv venv)
call venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
