from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
import bcrypt
from bson import ObjectId
from flask_cors import CORS,cross_origin
from flask_mail import Mail, Message

app = Flask(__name__)
CORS(app, supports_credentials=True)


# Email Configuration
app.config['MAIL_SERVER'] = 'localhost'
app.config['MAIL_PORT'] = 1025
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

app.config['MONGO_URI'] = 'mongodb+srv://ashishgolla2003:NS011618@cluster0.ophbpqo.mongodb.net/project'
mongo = PyMongo(app)


# Check and create 'arole' collection
admin_collection_name = 'arole'
admin_collection = mongo.db[admin_collection_name]

# Check and create 'prole' collection
patient_collection_name = 'prole'
patient_collection = mongo.db[patient_collection_name]

'''*****************************************************Signup Code**************************************************************************'''
@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        confirmPassword = data.get('confirmPassword')
        role = data.get('role')

        if not username or not email or not password or not confirmPassword or not role:
            return jsonify({'status': False, 'msg': 'Incomplete data provided'}), 400

        collection = admin_collection if role == "admin" else patient_collection

        if collection.find_one({'email': email}):
            return jsonify({'status': False, 'msg': 'Email already exists'}), 400

        if password != confirmPassword:
            return jsonify({'status': False, 'msg': 'Password and confirm password do not match'}), 400

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        user_id = collection.insert_one({'username': username,'email': email, 'password': hashed_password, 'role': role}).inserted_id

        return jsonify({'status': True, 'msg': 'Registered successfully', 'user_id': str(user_id)}), 201

    except Exception as e:
        return jsonify({'status': False, 'msg': str(e)}), 500

'''*****************************************************Login Code**************************************************************************'''
@app.route('/login', methods=['POST'])
@cross_origin(supports_credentials=True)
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role')

        if not username or not email or not password or not role:
            return jsonify({'status': False, 'msg': 'Incomplete data provided'}), 400

        collection = admin_collection if role == "admin" else patient_collection

        user = collection.find_one({'email': email, 'role': role})

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
           
            return jsonify({'status': True, 'msg': 'Login successful'}), 200
            
        else:
            return jsonify({'status': False, 'msg': 'Invalid email or password'}), 401

    except Exception as e:
        return jsonify({'status': False, 'msg': str(e)}), 500



# Check and create 'aroledetail' collection
admin_data_name = 'aroledetail'
admin_data = mongo.db[admin_data_name]

# Check and create 'proledetail' collection
patient_data_name = 'proledetail'
patient_data = mongo.db[patient_data_name]

'''*****************************************************input data Code**************************************************************************'''
#post data to database
@app.route('/fetchinput', methods=['POST'])
@cross_origin(supports_credentials=True)
def receive_and_save_data():
    try:
        data = request.get_json()  # Get the JSON data from the request
        
        if not data:
            return jsonify({'message': 'No data received'}), 400

        # getting role from the data 
        role = data[0]
      
        collection = admin_data if role == "admin" else patient_data

        # Ensure that "Sno" is unique and serves as the primary key
        for record in data[1]:
            if "Sno" not in record:
                return jsonify({'message': 'Each record must have a "Sno" field'}), 400

            # Check if a record with the same "Sno" already exists in the database
            existing_record = collection.find_one({"Sno": record["Sno"]})
            if existing_record:
                return jsonify({'message': f'Duplicate record with Sno {record["Sno"]} exists'}), 400

        # Insert the received data into the MongoDB collection
        result = collection.insert_many(data[1])
        
        return jsonify({'message': 'Data received and saved successfully'}), 200
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500

'''*****************************************************Contact us Code**************************************************************************'''
@app.route('/contact', methods=['POST'])
def contact():
    try:
        data = request.get_json()

        # Send email with field names
        msg = Message('New Contact Form Submission', sender=data['email'], recipients=['nani011618@gmail.com'])
        msg.body = f"Name: {data['first_name']} {data['last_name']}\nEmail: {data['email']}\nMessage: {data['message']}\n\nRaw Form Data:\n{data}"

        mail.send(msg)

        return jsonify({'message': 'Message sent successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Check and create 'symptomdetail' collection
symptom_data_name = 'symptomdetail'
symptom_data = mongo.db[symptom_data_name]

'''************************************************** ML Code To predict Disease ***********************************************************'''
# Import the required libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

'''def train_and_evaluate_classifier(selected_algorithm):
    symptoms_list = []
    labels = []
    

    # Assuming admin_data is a collection or data source
    for doc in admin_data.find():
        symptoms = doc['symptoms']
        symptoms_list.extend(symptoms)
        diseases=doc['disease']
        labels.extend([diseases] * len(symptoms))
        

    
    print(labels)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(symptoms_list)

        

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    algorithms = {
        'NaiveBayes': MultinomialNB(),
        'SVM': SVC(kernel='linear')
    }

    try:
        clf = algorithms[selected_algorithm]
    except KeyError:
        return None, None, "Invalid algorithm"

    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, accuracy, None, vectorizer

def predict_disease_algorithm(symptoms, selected_algorithm, clf, vectorizer):
    try:
        # Ensure symptoms is a list of strings
        if not all(isinstance(symptom, str) for symptom in symptoms):
            raise ValueError("Symptoms should be a list of strings")

        # Vectorize the input symptoms
        symptoms_vectorized = vectorizer.transform(symptoms)

        # Make a prediction based on vectorized symptoms
        predicted_disease = clf.predict(symptoms_vectorized)[0]

        return predicted_disease
    except ValueError as e:
        # Log the exception details
        print("ValueError:", str(e))
        raise ValueError("Invalid input format")
    '''

    
'''
@app.route('/predictdisease', methods=['POST'])
def predict_disease_flask():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        algorithm = data.get('algorithm', 'NaiveBayes')  # Default to NaiveBayes if not specified

        clf, accuracy, error, vectorizer = train_and_evaluate_classifier(algorithm)

        if error:
            return jsonify({'error': error}), 400  # Return a 400 Bad Request status for an invalid algorithm

        result = predict_disease_algorithm(symptoms, algorithm, clf, vectorizer)

        return jsonify({'predicted_disease': result, 'accuracy': accuracy}), 200

    except ValueError as ve:
        return jsonify({'error': 'Invalid input format', 'details': str(ve)}), 400
    except Exception as e:
        # Log the exception details
        print("Exception:", str(e))

        # Return a more informative error response
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500'''


'''********************** Train the Model *******************************'''

def train_evaluate_predict(selected_algorithm, symptoms):
    try:
        # Features and target variable
        l1 = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
              'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
              'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
              'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
              'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
              'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
              'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
              'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
              'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
              'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
              'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
              'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
              'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
              'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
              'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
              'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
              'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
              'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
              'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
              'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
              'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
              'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
              'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
              'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
              'red_sore_around_nose', 'yellow_crust_ooze']
        l2=[]
        for x in range(0,len(l1)):
            l2.append(0)

        disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
                   'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
                   'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria',
                   'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
                   'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
                   'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism',
                   'Hypoglycemia', 'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
                   'Urinary tract infection', 'Psoriasis', 'Impetigo']

        # Load testing data
        tr = pd.read_csv(r"C:\Users\OS23H\OneDrive\Desktop\project\server\Testing_Predict.csv")
        tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
        'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
        'Migraine':11,'Cervical spondylosis':12,
        'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
        'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
        'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
        'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
        '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
        'Impetigo':40}},inplace=True)

        X_test = tr[l1]
        y_test = tr["prognosis"]
        np.ravel(y_test)

        # Load training data
        df = pd.read_csv(r"C:\Users\OS23H\OneDrive\Desktop\project\server\Training_Predict.csv")
        df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
        'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
        'Migraine':11,'Cervical spondylosis':12,
        'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
        'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
        'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
        'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
        '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
        'Impetigo':40}},inplace=True)


        # Convert labels to numeric type
        X = df[l1]
        y = df["prognosis"]
        np.ravel(y)        


        algorithms = {
            'NaiveBayes': MultinomialNB(),
            
        }

        clf = algorithms.get(selected_algorithm)

        if clf is None:
            return jsonify({'error': 'Invalid algorithm'}), 400

        clf.fit(X, np.ravel(y))

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred,normalize=False)
        accuracy = float(accuracy)
      
        
        for k in range(0,len(l1)):
             for z in symptoms:
                if(z==l1[k]):
                  l2[k]=1
        # Predict disease based on input symptoms
        predicted_disease = clf.predict([l2])[0]


        return jsonify({'predicted_disease': disease[predicted_disease], 'accuracy': accuracy}), 200

    except Exception as e:
        # Log the exception details
        print("Exception:", str(e))
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500

'''********************** Predict the Disease *******************************'''
@app.route('/predictdisease', methods=['POST'])
def predict_disease_flask():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging statement
        symptoms = data.get('symptoms', [])
        algorithm = data.get('algorithm', 'NaiveBayes')  # Default to NaiveBayes if not specified

        return train_evaluate_predict(algorithm, symptoms)

    except Exception as e:
        # Log the exception details
        print("Exception:", str(e))
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500



'''********************** Fetching symptoms  *******************************'''

@app.route('/getsymptoms', methods=['GET'])
def get_symptoms():
    try:
        # Retrieve all documents from the 'symptom_data' collection
        symptoms_cursor = symptom_data.find()

        # Convert MongoDB cursor to a list of dictionaries
        symptom_list = list(symptoms_cursor)

        # Convert ObjectId to string for each document
        for symptom in symptom_list:
            symptom['_id'] = str(symptom['_id'])

        # Return the list of symptoms as JSON response
        return jsonify(symptom_list), 200, {'Content-Type': 'application/json', 'Cache-Control': 'no-cache'}

    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error fetching symptoms: {str(e)}")

        # Return a more informative error response
        error_message = {'error': 'Internal Server Error', 'details': str(e)}
        return jsonify(error_message), 500


if __name__ == '__main__':
    app.run(debug=True)


