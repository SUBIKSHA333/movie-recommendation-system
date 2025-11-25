import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Combine features
df["combined_features"] = (
    df["title"] + " " +
    df["genres"] + " " +
    df["keywords"]
)

# Vectorization
cv = CountVectorizer(stop_words="english")
vectors = cv.fit_transform(df["combined_features"])

# Similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_name):
    movie_name = movie_name.lower()
    if movie_name not in df['title'].str.lower().values:
        return ["Movie not found in database"]
    
    index = df[df['title'].str.lower() == movie_name].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_movies = []
    for i in distances[1:6]:
        recommended_movies.append(df.iloc[i[0]].title)

    return recommended_movies

# Test
print("Recommendations for Avatar:")
print(recommend("Avatar"))
