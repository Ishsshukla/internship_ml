import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from fastapi import FastAPI
import requests

app = FastAPI()


response = requests.get('https://workshala-7v7q.onrender.com/internshipData')
data = response.json()  
df = pd.DataFrame(data)
df.isnull().sum()
df.head()

df_copy1 =df.drop(columns=['_id', 'time',  'company','type', 'state',
       'salary','work','description','title']).copy()

df_copy1['skills_text'] = df_copy1['skills'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)


vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(df_copy1['skills_text'])

cosine_sim = cosine_similarity(matrix , matrix)

def get_intership(skills):
    selected_skills = ', '.join(skills)

    new_vector = vectorizer.transform([selected_skills])

    cosine_sim_with_selected_skills = cosine_similarity(new_vector, matrix)

    sim_scores = list(enumerate(cosine_sim_with_selected_skills[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    indices = [score[0] for score in sim_scores[:5]]
    recommendations = []
    for i in indices:
       recommendations.append({"Internship" : df['jobProfile'].iloc[i] , "company" : df["company"].iloc[i] , "salary" : df["salary"].iloc[i] , "state" : df["state"].iloc[i] })
    
    return recommendations

app = FastAPI()

@app.get("/internship/{skills}")
def recommendation_func(skills : str):
    recommendations = get_intership([skills])
    return recommendations













    