from django.shortcuts import render
import joblib
import os

# Paths to the model and data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'nlpapp', 'my_model_stock_price.pkl')
countvector_path = os.path.join(BASE_DIR, 'nlpapp', 'countvector.pkl')

# Load the model and CountVectorizer
model = joblib.load(model_path)
countvector = joblib.load(countvector_path)

def predict(request):
    result = None
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        
        # Transform input_text using the pre-fitted CountVectorizer
        transformed_input = countvector.transform([input_text])
        
        # Make prediction
        prediction = model.predict(transformed_input)
        
        if prediction[0] == 1:
            result = "The stock price increased"
        else:
            result = "The stock price decreased"
    
    return render(request, 'predict.html', {'result': result})
