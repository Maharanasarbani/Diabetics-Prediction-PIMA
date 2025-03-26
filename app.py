import numpy as np
from flask import Flask,request,render_template # type: ignore
import pickle
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)
model=pickle.load(open("model.pkl",'rb'))

scaler = pickle.load(open("scaler.pkl", 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features=np.array(float_features)
    input_data_reshaped=features.reshape(1,-1)
    std_data=scaler.transform(input_data_reshaped)
    prediction=model.predict(std_data)
    ans=""
    if prediction[0]==0:
        ans="The person is not diabetic"
    else:
        ans="The person is diabetic"
    return render_template("index.html",prediction_text="{}".format(ans))

if __name__=="__main__":
    app.run(debug=True)