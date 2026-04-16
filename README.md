# 🚗 Car Price Prediction Web App

A Machine Learning-powered web application that predicts the resale price of used cars based on various features such as company, manufacturing year, fuel type, and kilometers driven.

Built using **Flask**, **Scikit-learn**, and an interactive frontend, this project provides real-time predictions along with visualization and comparison features.

---

## 📌 Features

* 🔮 Predict car resale price instantly
* 📊 Shows price range (±20% confidence)
* ⚖️ Compare two cars side-by-side
* 📉 Depreciation analysis over time
* 📈 Feature importance visualization
* 🔁 Retrain model directly from UI
* 🎯 Clean and modern responsive UI

---

## 🧠 Machine Learning Details

* Algorithm: **Linear Regression**
* Dataset: 816 records, 25 car brands
* Features used:

  * Company
  * Fuel Type
  * Year
  * Kilometers Driven
  * Car Age (derived)

### 📊 Model Performance

* R² Score: ~0.63
* MAE: ~₹1.57 Lakh
* RMSE: ~₹2.7 Lakh

Model training and evaluation handled in: 

---

## 🖥️ Tech Stack

### Backend

* Python
* Flask
* Scikit-learn
* NumPy, Pandas

### Frontend

* HTML, CSS, JavaScript
* Chart.js (for graphs)

Frontend UI file: 
Backend API: 

---

## ⚙️ How It Works

1. User selects:

   * Car company
   * Model
   * Year
   * Fuel type
   * Kilometers driven

2. Data is sent to Flask API

3. Model processes input using:

   * OneHotEncoding (categorical data)
   * Numerical features

4. Predicted price is returned and displayed with:

   * Estimated price
   * Price range
   * Depreciation
   * Comparison insights

---

## 🚀 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/car-price-predictor.git
cd car-price-predictor
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
python app.py
```

### 4️⃣ Open in browser

```
http://127.0.0.1:5000/
```

---

## 🔁 Retraining the Model

You can retrain the model anytime:

* Click **"Retrain model"** button in UI
  OR

```bash
python car_price_prediction.py
```

---

## 📂 Project Structure

```
├── app.py                      # Flask backend
├── car_price_prediction.py     # Model training script
├── car_price_model.pkl         # Saved ML model
├── Cleaned_Car_data.csv        # Dataset
├── templates/
│   └── index.html              # Frontend UI
```

---

## 🎯 Future Improvements

* Use advanced models (Random Forest, XGBoost)
* Add more features (transmission, owner type)
* Deploy on cloud (Render / AWS)
* Add login system & saved predictions
* Improve dataset size for better accuracy

---

## 🙌 Author

**Dhiraj Sarangi**
BCA Student | Aspiring MERN & ML Developer

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

