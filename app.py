from flask import Flask, render_template, request 
import re
import nltk 

from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import pickle 

app = Flask(__name__)

# Load the pre-trained model
model1 = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))


nltk.download('stopwords')
nltk.download('wordnet')

# Function to process text
def process_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        sms_message = request.form['sms']
        processed_message = process_text(sms_message)
        
        processed_message_vectorized = vectorizer.transform([processed_message])
        
        # Make a prediction 
        prediction = model1.predict(processed_message_vectorized)[0]
        
        
    
    return render_template('index.html', prediction=prediction)



if __name__ == '__main__':
    app.run(debug=True)
