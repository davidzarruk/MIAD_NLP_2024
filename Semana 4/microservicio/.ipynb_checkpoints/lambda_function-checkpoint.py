import joblib
import os
import json
import numpy as np

def lambda_handler(event, context):

    clf = joblib.load(os.path.dirname(__file__) + '/phishing_clf_v0.pkl') 

    url = event.get('url')

    # Create a new dictionary to store features
    url_ = {'url': url}

    # Create features
    keywords = ['https', 'login', '.php', '.html', '@', 'sign']
    for keyword in keywords:
        url_['keyword_' + keyword] = int(keyword in url_['url'])

    url_['lenght'] = len(url_['url']) - 2
    domain = url_['url'].split('/')[2]
    url_['lenght_domain'] = len(domain)
    url_['isIP'] = int((url_['url'].replace('.', '') * 1).isnumeric())
    url_['count_com'] = url_['url'].count('com')
    url_.pop('url')
    
    # Make prediction
    p1 = clf.predict_proba(np.array(list(url_.values())).reshape(1, -1))[0,1]

    # Create a response body
    response_body = {
        "result": p1
    }
    
    # Create a response
    response = {
        "statusCode": 200,
        "body": json.dumps(response_body)
    }

    return response