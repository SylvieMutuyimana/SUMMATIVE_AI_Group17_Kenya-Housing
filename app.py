from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = pd.read_csv('merged_dataset.csv')

# Load the trained model
clf = joblib.load('the_model.joblib')

# Create an instance of the OneHotEncoder class and fit it to the Town column of the dataset
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(data[['Town']])

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    towns = data['Town'].unique()
    sorted_towns = sorted(towns)
    # If the form has been submitted, make a prediction and show the results
    if request.method == 'POST':
        # Get the input values from the form
        sq_mtrs = request.form.get('sq_mtrs', type=int)
        bedrooms = request.form.get('bedrooms', type=int)
        bathrooms = request.form.get('bathrooms', type=int)
        town = request.form.get('town')
        duration = request.form.get('duration')
        duration_value = request.form.get('duration_value', type=int)

        # Collect all the inputs except the town so that we can predict for each other town
        X_num = pd.DataFrame({'Bedrooms': [bedrooms], 'Bathrooms': [bathrooms],'sq_mtrs': [sq_mtrs]})
        X_num = pd.concat([X_num]*len(data['Town'].unique()), ignore_index=True)
        X_cat = pd.DataFrame({'Town': data['Town'].unique()})
        X_cat_encoded = pd.DataFrame(ohe.transform(X_cat).toarray(), columns=ohe.get_feature_names_out(['Town']))
        X = pd.concat([X_num, X_cat_encoded], axis=1)

        # Make the predictions for all towns and adjust based on the duration and num values
        price_preds = clf.predict(X)
        if duration == 'months':
            price_preds *= duration_value / 12
        else:
            price_preds *= duration_value / 365

        # Add the predicted prices to the data_copy DataFrame for the corresponding Town
        data_copy = data.copy()
        data_copy['Price'] = np.nan
        data_copy['Price'] = data_copy['Price'].astype(float) # Convert to float dtype
        for i, town_name in enumerate(ohe.categories_[0]):
            mask = data_copy['Town'] == town_name
            data_copy.loc[mask, 'Price'] = price_preds[i]

        # Get the top 5 highest priced towns and the 5 lowest priced towns
        lowest_price_towns = data_copy[['Town', 'Price']].sort_values('Price').drop_duplicates().head(5).to_dict('records')
        len_lowest_price_towns = len(lowest_price_towns)
        highest_price_towns = data_copy[['Town', 'Price']].sort_values('Price', ascending=False).drop_duplicates().head(5).to_dict('records')
        len_highest_price_towns = len(highest_price_towns)

        # Round the final prediction to 2 decimal places
        price_pred = round(data_copy.loc[data_copy['Town'] == town, 'Price'].iloc[0], 2)
        
        # Render the results table with the additional rows for lowest and highest price towns
        return render_template('index.html', towns=sorted(ohe.categories_[0]), 
                        bedrooms=bedrooms, bathrooms=bathrooms,
                        sq_mtrs = sq_mtrs ,
                        town=town, duration=duration, 
                        duration_value=duration_value, price_pred=price_pred,
                        lowest_price_towns=lowest_price_towns, highest_price_towns=highest_price_towns,
                        len_lowest_price_towns=len_lowest_price_towns, len_highest_price_towns=len_highest_price_towns
                        )
    # If the form has not been submitted yet, show the form without any results
    return render_template('index.html', towns=sorted_towns)


if __name__ == '__main__':
    app.run(debug=True)
