import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
import numpy as np
import ast
from collections import Counter

# Dosyaları oku
df = pd.read_csv("Content Based Filtering/archive (2)/data/data.csv")
df_genres = pd.read_csv("Content Based Filtering/archive (2)/data/data_w_genres.csv")

# Artists kolonunu sağlamca parse et
def parse_artists(x):
    if pd.isna(x):
        return ["unknown_artist"]
    try:
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            if x.startswith("["):
                return ast.literal_eval(x)
            else:
                return [x.strip('"').strip("'")]
        return [str(x)]
    except Exception:
        return ["unknown_artist"]

df['artists_parsed'] = df['artists'].apply(parse_artists)
df_genres['artists_parsed'] = df_genres['artists'].apply(parse_artists)

# Artist→Genre haritası oluştur (ilk genre'yı al)
def parse_genres(x):
    try:
        if isinstance(x, list):
            return x[0] if x else "unknown"
        if isinstance(x, str) and x.startswith("["):
            l = ast.literal_eval(x)
            return l[0] if l else "unknown"
        if isinstance(x, str):
            return x
        return "unknown"
    except Exception:
        return "unknown"

artist_to_genre = {}
for i, row in df_genres.iterrows():
    artist_list = row['artists_parsed']
    genre_val = parse_genres(row['genres'])
    for a in artist_list:
        if a not in artist_to_genre:
            artist_to_genre[a] = genre_val

# Şarkıya genre ekle (ilk bulunan artist)
def get_genre(artist_list):
    for a in artist_list:
        if a in artist_to_genre:
            return artist_to_genre[a]
    return "unknown"

df['genre'] = df['artists_parsed'].apply(get_genre)

# Sayısal feature'lar, key, mode
numeric_features = [
    'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity', 'key', 'mode'
]
df_numeric = df[numeric_features].fillna(0)
scaler = MinMaxScaler()
numeric_scaled = scaler.fit_transform(df_numeric)

# Artist ve genre encoding (top 1000 artist, top 100 genre)
artist_list = [artist for sublist in df['artists_parsed'] for artist in sublist]
top_artists = [a for a, _ in Counter(artist_list).most_common(1000)]
df['artists_filtered'] = df['artists_parsed'].apply(lambda lst: [a if a in top_artists else 'other_artist' for a in lst])

mlb_artist = MultiLabelBinarizer()
artist_encoded = mlb_artist.fit_transform(df['artists_filtered'])

# Genre encoding
# Genre encoding
top_genres = [g for g, _ in Counter(df['genre']).most_common(100)]
df['genre_filtered'] = df['genre'].apply(lambda g: g if g in top_genres else "other_genre")
mlb_genre = MultiLabelBinarizer()
genre_for_encoding = df['genre_filtered'].apply(lambda x: [x]).tolist()  # <- önemli satır!
genre_encoded = mlb_genre.fit_transform(genre_for_encoding)


# Vektörleri birleştir
features_all = np.hstack([numeric_scaled, artist_encoded, genre_encoded])

# Şarkı index fonksiyonu
def get_song_index(name, artist=None):
    songs = df[df['name'].str.lower() == name.lower()]
    if artist:
        songs = songs[songs['artists'].str.lower().str.contains(artist.lower())]
    if len(songs) == 0:
        print(f"Şarkı bulunamadı: {name} - {artist}")
        return None
    return songs.index[0]

# Kullanıcı şarkıları
user_songs = [
    ("Shape of You", "Ed Sheeran"), 
    ("Shake It Off", "Taylor Swift"),
]
user_indices = [get_song_index(name, artist) for name, artist in user_songs]
user_indices = [idx for idx in user_indices if idx is not None]

if len(user_indices) == 0:
    print("Hiçbir şarkı bulunamadı.")
else:
    user_vector = np.mean(features_all[user_indices], axis=0)
    similarities = cosine_similarity([user_vector], features_all)[0]
    for idx in user_indices:
        similarities[idx] = -1
    top_idx = similarities.argsort()[::-1][:10]
    print("\nEn benzer 10 şarkı önerisi:")
    for idx in top_idx:
        print(f"{df.iloc[idx]['name']} - {df.iloc[idx]['artists']} (genre: {df.iloc[idx]['genre']})")
