from flask import Flask, request,render_template, jsonify
import joblib
import pandas as pd

app= Flask(__name__)

@app.route('/',methods=['Get','POST'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['Get','POST'])
def predict():
    age = int(request.form['age'])
    euribor3m = float(request.form['euribor'])
    df = pd.DataFrame({'age': [age], 'euribor3m': [euribor3m]})
    df=pd.DataFrame(data_preprocessing(df))
    with open('model.pkl','rb') as file:
        model=joblib.load(file)
        prediction=model.predict(df)
        print(prediction)
        # print(prediction_test)
        # df['subscribed']=prediction_test
        return jsonify({'prediction': prediction.tolist()})

def data_preprocessing(df):
    with open('scaler.pkl','rb') as file:
        scaler=joblib.load(file)
        df=scaler.transform(df)
        # pd.DataFrame(df).to_csv(r"prediction.csv", index=False)
        return df

#below lines required on for development, remove when deploying    
# if __name__ =='__main__':
#     app.run(debug=True, port=5002)


