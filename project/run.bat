REM Check if Flask is installed
python -c "import flask" 2> nul

REM If Flask is not installed, install it using pip
if %errorlevel% neq 0 (
    pip install flask
)
pip install flask scikit-learn
REM Run the app
jupyter nbconvert --to script the_data.ipynb
the_data.py
cd ..
jupyter nbconvert --to script the_model.ipynb
the_model.py