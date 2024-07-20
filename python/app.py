from flask import Flask, request, jsonify, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
import jwt
from flask_cors import CORS
from flask_mysqldb import MySQL
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
import requests


app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# MySQL Configuration
app.config['SECRET_KEY'] = "SomeJwtTokenkvghtcugihji"
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'crop_prediction'
mysql = MySQL(app)

# Function to fetch weather data from OpenWeather API
def get_weather_data(city_name, api_key):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

# Constants
OPENWEATHER_API_KEY = 'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude={part}&appid={API key}'

# Endpoint to get weather data
@app.route('/weather', methods=['GET'])
def get_weather():
    city_name = request.args.get('city')
    if not city_name:
        return jsonify({'error': 'City parameter is missing'}), 400

    weather_data = get_weather_data(city_name, OPENWEATHER_API_KEY)
    if weather_data:
        return jsonify(weather_data)
    else:
        return jsonify({'error': 'Failed to fetch weather data'}), 500

# Secret key for JWT

# Read CSV data into a pandas DataFrame
crop_data = pd.read_csv(r'D:\crop_predictions\dataset\crop_data.csv')
rainfall_data = pd.read_csv(r'D:\crop_predictions\dataset\District_wise_rainfall_data.csv')
agriculture_data = pd.read_csv(r'D:\crop_predictions\dataset\agriculture_data.csv')

# Function to calculate crop age
def calculate_crop_age(planting_date):
    planting_date = datetime.strptime(planting_date, '%Y-%m-%d')
    current_date = datetime.now()
    crop_age = (current_date - planting_date).days
    return crop_age

# Function to predict fertilizer recommendation based on crop name and age
def predict_fertilizer(crop_name, crop_age):
    # Filter data for the specified crop
    crop_info = crop_data[crop_data['Plant Name'] == crop_name]
    
    # Find the growth stage based on crop age
    growth_stage = ''
    for index, row in crop_info.iterrows():
        start_day = row['Start Day']
        end_day = row['End Day']
        if start_day <= crop_age <= end_day:
            growth_stage = row['Stage']
            break
    
    # Find the recommended fertilizer for the growth stage
    if growth_stage:
        fertilizer_recommendation = crop_info[crop_info['Stage'] == growth_stage]['Recommended Fertilizer'].values[0]
        return fertilizer_recommendation
    else:
        return "No recommendation found for the given crop age."


# Rename the 'DISTRICT' column in rainfall_data to match with 'District_Name' in agriculture_data
rainfall_data.rename(columns={'DISTRICT': 'District_Name'}, inplace=True)

# Merge the datasets on 'District_Name'
merged_data = pd.merge(agriculture_data, rainfall_data, on='District_Name', how='inner')

# Function to preprocess data and predict yield
def predict_yield(crop_name, district_name, area,season):
    # Collecting only the data from the karnataka
    karnataka_data = agriculture_data[agriculture_data['State_Name'] =='Karnataka']
    karnataka_data.reset_index(drop=True, inplace=True)
    karnataka_data.index += 1
    karnataka_data
    
    karnataka_data.info()

    ## Removing or droping the missing values
    karnataka_data = karnataka_data.dropna()
    karnataka_data.drop('State_Name',axis=1,inplace= True)
    karnataka_data.drop("Crop_Year",axis=1,inplace= True)
    karnataka_data

    # Data understaning and separting the data based on the dtype
    dict_types = dict(karnataka_data.dtypes)
    dict_types
    continous = []
    categorical = []
    for name,type in dict_types.items():
        if type == str('float64'):
            continous.append(name)
        else:
            categorical.append(name)
    print('The Continous variables are:  ',continous)
    print('\nThe Categorical variables are:  ',categorical)
    print(len(continous),len(categorical))

    # Applying Co-realation between the continous variables
    karnataka_data[continous].corr()

    karnataka_data.describe()

    karnataka_data['Crop'].unique()

    ## cleaning and replacing the data
    karnataka_data["Crop"].replace("Arhar/Tur","Pigeonpea",inplace = True)
    karnataka_data["Crop"].replace("Bajra","Pearl millet",inplace = True)
    karnataka_data["Crop"].replace("Cotton(lint)","Cotton",inplace = True)
    karnataka_data["Crop"].replace("Jowar","Sorghum",inplace = True)
    karnataka_data["Crop"].replace("Moong(Green Gram)","Green Gram",inplace = True)
    karnataka_data["Crop"].replace("Rapeseed &Mustard","Mustard",inplace = True)
    karnataka_data["Crop"].replace("Gram","Chickpea",inplace = True)
    karnataka_data["Crop"].replace("Dry chillies","Chillies",inplace = True)
    karnataka_data["Crop"].replace("Other Kharif pulses","Black Gram",inplace = True)
    karnataka_data["Crop"].replace("Small millets","Little millet",inplace = True)
    karnataka_data["Crop"].replace("Other  Rabi pulses","Red Beans",inplace = True)
    karnataka_data["Crop"].replace("Arcanut (Processed)","Arcanut",inplace = True)
    karnataka_data["Crop"].replace("Atcanut (Raw)","Arcanut",inplace = True)
    karnataka_data["Crop"].replace("Citrus Fruit","Lemon",inplace = True)
    karnataka_data["Crop"].replace("Other Fresh Fruits","Water Mellon",inplace = True)
    karnataka_data["Crop"].replace("Pome Fruit","Pomegranate",inplace = True)
    karnataka_data["Crop"].replace("Urad","Yellow Gram",inplace = True)
    karnataka_data["Crop"].replace("Cowpea(Lobia)","Cowpea",inplace = True)
    karnataka_data["Crop"].replace("Dry ginger","Ginger",inplace = True)
    karnataka_data["Crop"].replace("Peas & beans (Pulses)","Bengal Gram",inplace = True)
    karnataka_data["Crop"].replace("Beans & Mutter(Vegetable)","Green beans",inplace = True)
    karnataka_data["Crop"].replace("Cashewnut Processed","Cashewnut",inplace = True)
    karnataka_data["Crop"].replace("Cashewnut Raw","Cashewnut",inplace = True)

    karnataka_data['Season'].unique()

    # Applying the onehot encoder converting categorical data into numerical data
    

    enc = OneHotEncoder(drop='first')
    cat_data = enc.fit_transform(karnataka_data[categorical])
    karnataka_data.drop(categorical,axis=1,inplace = True)
    cat_data = pd.DataFrame(cat_data.toarray(), columns=enc.get_feature_names_out(categorical))
    cat_data = cat_data.astype(int)
    column_names = cat_data.columns
    new_column_names = [name.split('_')[-1] for name in column_names]
    column_mapping = {old_name: new_name for old_name, new_name in zip(cat_data.columns, new_column_names)}
    cat_data.rename(columns=column_mapping, inplace=True)
    karnataka_data = pd.concat([karnataka_data, cat_data], axis='columns')

    karnataka_data.dropna(inplace = True)
    karnataka_data

    #  sepearting the input and the output variable
    X = karnataka_data.drop('Production',axis=1)
    y = karnataka_data['Production']

    ## Applying the Train Test split
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size =0.8,random_state =3)

    # ElastiNet Regression
 
    elastic_model = ElasticNet(alpha = 1,l1_ratio = 0.9)
    elastic_model.fit(X_train,y_train)

    print('Coefficiet:',elastic_model.coef_,'\n','Intercept',elastic_model.intercept_)

    ypred_train = elastic_model.predict(X_train)
    ypred_test = elastic_model.predict(X_test)

    print("Train Score:", r2_score(y_train, ypred_train))
    print('Test Score:', r2_score(y_test, ypred_test))

    # Prepare test data
    test_data = pd.DataFrame({
        'District_Name': [district_name],
        'Season': [season],
        'Crop': [crop_name],
        'Area': [area]})

    # OneHotEncoding test data using the same encoder
    test_cat_data = enc.transform(test_data[categorical])
    print(test_cat_data)
    test_data.drop(categorical, axis=1, inplace=True)
    test_cat_data = pd.DataFrame(test_cat_data.toarray(), columns=enc.get_feature_names_out(categorical))
    test_cat_data = test_cat_data.astype(int)

    # Renaming columns
    test_cat_data.rename(columns=column_mapping, inplace=True)

    # Combining encoded test data with the original test data
    test_data = pd.concat([test_data, test_cat_data], axis='columns')

    pred=elastic_model.predict(test_data)

    return f"The Crop yield will be {pred}"




# Predict fertilizer endpoint
@app.route('/predict-fertilizer', methods=['POST'])
def predict_fertilizer_endpoint():
    data = request.json
    crop_name = data['crop_name']
    planting_date = data['planting_date']
    crop_age = calculate_crop_age(planting_date)
    recommendation = predict_fertilizer(crop_name, crop_age)
    return jsonify({'recommendation': recommendation})

# Predict yield endpoint
@app.route('/predict-yield', methods=['POST'])
def predict_yield_endpoint():
    data = request.json
    crop_name = data['crop_name']
    district_name = data['district_name']  # Specify the district name for which you want to predict yield
    season = data['season']
    acres = data['acres']  # Number of acres
    predicted_yield = predict_yield(crop_name, district_name,season, acres)
    return jsonify({'predicted_yield': predicted_yield})


if __name__ == '__main__':
    app.run(debug=True)
