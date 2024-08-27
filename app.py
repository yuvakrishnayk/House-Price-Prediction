from flask import Flask, render_template, request
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load your data and train the model
df = pd.read_csv("test.csv")
features = ["beds", "baths", "size", "lot_size"]
X = df[features]
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=features)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.1, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    beds = int(request.form['beds'])
    baths = int(request.form['baths'])
    size = int(request.form['size'])
    lot_size = int(request.form['lot_size'])
    
    input_data = [[beds, baths, size, lot_size]]
    predicted_price = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Predicted House Price: {predicted_price:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
