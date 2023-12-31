import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster_Reponse', con=engine)
    X = df['message']
    Y = df[['1', 'request-0', 'offer-0',
       'aid_related-0', 'medical_help-0', 'medical_products-0',
       'search_and_rescue-0', 'security-0', 'military-0', 'child_alone-0',
       'water-0', 'food-0', 'shelter-0', 'clothing-0', 'money-0',
       'missing_people-0', 'refugees-0', 'death-0', 'other_aid-0',
       'infrastructure_related-0', 'transport-0', 'buildings-0',
       'electricity-0', 'tools-0', 'hospitals-0', 'shops-0', 'aid_centers-0',
       'other_infrastructure-0', 'weather_related-0', 'floods-0', 'storm-0',
       'fire-0', 'earthquake-0', 'cold-0', 'other_weather-0',
       'direct_report-0']]
    return X, Y


def tokenize(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__max_features': [5000, 10000],
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }
    
    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)
    
    return cv


def train_model(X, Y):
    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train model
    model = build_model()
    model.fit(X_train, Y_train)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    categories = Y_test.columns.tolist()

    # Iterate through each category and call classification_report
    for idx, category in enumerate(categories):
        print("Category:", category)
        print(classification_report(Y_test[category], Y_pred[:, idx]))
        print("--------------------")


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    print("Model saved successfully!")


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        
        print('Training model...')
        model = train_model(X, Y)
        
        print('Evaluating model...')
        evaluate_model(model, X, Y)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('Trained model saved!')
    
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
