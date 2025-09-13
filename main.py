from flask import Flask, request, render_template
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor


app = Flask(__name__)

data = pd.read_csv('cleaned_data.csv')

locations = sorted(data['location'].unique())
data1 = pd.get_dummies(data, drop_first=True)
X = data1.drop('price', axis=1)
y = data1['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgboost_model = xgb.XGBRegressor()
xgboost_model.fit(X_train, y_train)


def predict_price(user_input, model, encoder_columns):
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=encoder_columns, fill_value=0)
    predicted_price = model.predict(input_encoded)
    return f"{predicted_price[0]:.2f}"


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/card')
def card():
    predicted_price = None
    if request.method == 'POST':
        location = request.form['location']
        bhk = int(request.form.get('bhk'))
        bath = int(request.form.get('bath'))
        sqft = int(request.form.get('total_sqft'))
        
        user_input = {
            'location': location,
            'total_sqft': sqft,
            'bath': bath,
            'bhk': bhk
        }

        if user_input['location'] not in data.location:
            user_input['location'] = 'other'

        predicted_price = predict_price(user_input, xgboost_model, X.columns)
    return render_template('card.html', locations=locations, predicted_price=predicted_price)

@app.route('/card/predict', methods=['POST'])
def predict():
    location = request.form['location']
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    sqft = int(request.form.get('total_sqft'))
    
    user_input = {
        'location': location,
        'total_sqft': sqft,
        'bath': bath,
        'bhk': bhk
    }

    predicted_price = predict_price(user_input, xgboost_model, X.columns)
    
    return render_template('card.html', locations=locations, predicted_price=predicted_price, place_name=location)

@app.route('/property')
def property():
    return render_template('property.html')

@app.route('/upload')
def index():
    return render_template('upload_form.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        data = pd.read_csv(filename)
        data.dropna(inplace=True)

        data1 = pd.get_dummies(data, drop_first=True)
        
        price_columns = [col for col in data1.columns if col.lower() == 'price']
        if price_columns:
            price_column = price_columns[0]
            X = data1.drop(price_column, axis=1)
            y = data1[price_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        
        model_choice = request.form['model']

        if model_choice == 'LinearRegression':
            model = LinearRegression()
        elif model_choice == 'RandomForest':
            model = RandomForestRegressor()
        elif model_choice == 'XGBoost':
            model = xgb.XGBRegressor()
        elif model_choice == 'SVM':
            model = SVR()
        elif model_choice == 'Dtree':
            model = DecisionTreeRegressor()
        elif model_choice == 'lasso':
            model = Lasso()
        elif model_choice == 'MLPRegressor':
            model = MLPRegressor()
        else:
            return 'Invalid model choice', 400

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**0.5

        return render_template('results.html', r2=r2, rmse=rmse)
    else:
        return 'Invalid file type. Only CSV files are allowed.', 400

if __name__ == '__main__':
    app.run(debug=True)
