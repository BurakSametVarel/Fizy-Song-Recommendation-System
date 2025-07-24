from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import ast
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from flask import send_file
import io
import os
from flask import Flask, render_template, request, redirect, url_for, flash
import librosa
import numpy as np
import tempfile
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector



app = Flask(__name__)
app.secret_key = 'supersecret'  # Flash mesajlar için

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


import os

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "123456")
MYSQL_DB = os.environ.get("MYSQL_DB", "fizy")

# --------------- DB & TABLE OLUŞTUR ---------------
def create_db_and_table():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB}")
    cursor.close()
    conn.close()
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL
        )
    """)
    cursor.close()
    conn.close()

create_db_and_table()

def get_db():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )


def create_listened_table():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS listened_songs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) NOT NULL,
            song_name VARCHAR(255) NOT NULL,
            song_artist VARCHAR(255) NOT NULL,
            listened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    cur.close()
    db.close()

create_listened_table()


# --------------- VERİ YÜKLEME VE HAZIRLAMA ---------------

df = pd.read_csv("data/data.csv")
df_genres = pd.read_csv("data/data_w_genres.csv")

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

def get_genre(artist_list):
    for a in artist_list:
        if a in artist_to_genre:
            return artist_to_genre[a]
    return "unknown"

df['genre'] = df['artists_parsed'].apply(get_genre)

numeric_features = [
    'acousticness', 'danceability', 'energy', 'instrumentalness', 'popularity',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'key', 'mode'
]
df_numeric = df[numeric_features].fillna(0)
scaler = MinMaxScaler()
numeric_scaled = scaler.fit_transform(df_numeric)

artist_list = [artist for sublist in df['artists_parsed'] for artist in sublist]
top_artists = [a for a, _ in Counter(artist_list).most_common(1000)]
df['artists_filtered'] = df['artists_parsed'].apply(lambda lst: [a if a in top_artists else 'other_artist' for a in lst])

mlb_artist = MultiLabelBinarizer()
artist_encoded = mlb_artist.fit_transform(df['artists_filtered'])

top_genres = [g for g, _ in Counter(df['genre']).most_common(100)]
df['genre_filtered'] = df['genre'].apply(lambda g: g if g in top_genres else "other_genre")
mlb_genre = MultiLabelBinarizer()
genre_for_encoding = df['genre_filtered'].apply(lambda x: [x]).tolist()
genre_encoded = mlb_genre.fit_transform(genre_for_encoding)

features_all = np.hstack([numeric_scaled, artist_encoded, genre_encoded])

def get_song_index(name, artist=None):
    # Boş/None aramalar boş döner
    if not name or pd.isna(name):
        return None
    # Şarkı adını case-insensitive şekilde içinde geçenleri bul
    songs = df[df['name'].str.lower().str.contains(name.lower(), na=False)]
    if artist and not pd.isna(artist):
        songs = songs[songs['artists'].str.lower().str.contains(artist.lower(), na=False)]
    if len(songs) == 0:
        return None
    # En popüler olanı veya ilkini dönebiliriz
    # En popüler:
    songs = songs.sort_values("popularity", ascending=False)
    return songs.index[0]


def get_song_index_from_fullstr(full_str):
    # full_str örn: "Hips Don't Lie (feat. Wyclef Jean) - Shakira, Wyclef Jean"
    if " - " in full_str:
        name, _ = full_str.split(" - ", 1)
    else:
        name = full_str
    name = name.strip()
    # Önce tam eşleşen
    candidates = df[df['name'].str.lower() == name.lower()]
    if len(candidates) == 0:
        # Sonra içeren
        candidates = df[df['name'].str.lower().str.contains(name.lower(), na=False)]
    if len(candidates) == 0:
        return None
    # En popüler satırı döndür
    candidates = candidates.sort_values("popularity", ascending=False)
    return candidates.index[0]

def get_song_indices(song_lines):
    indices = []
    for line in song_lines:
        idx = get_song_index_from_fullstr(line)
        if idx is not None:
            indices.append(idx)
    return indices


def get_recommendations_multi(song_artist_lines, top_n=10):
    user_indices = get_song_indices(song_artist_lines)
    if not user_indices:
        return []

    # Kullanıcı sanatçılarını topla (hepsini küçük harfe çevir)
    user_artists = set()
    for idx in user_indices:
        for a in df.iloc[idx]['artists_parsed']:
            user_artists.add(a.strip().lower())

    user_vector = np.mean(features_all[user_indices], axis=0)
    similarities = []
    batch_size = 5000
    n = features_all.shape[0]
    for i in range(0, n, batch_size):
        batch = features_all[i:i+batch_size]
        sim = cosine_similarity([user_vector], batch)[0]
        similarities.extend(sim)
    similarities = np.array(similarities)
    for idx in user_indices:
        similarities[idx] = -1

    # Sıralı şekilde tüm benzerleri dolaş ve aynı sanatçıyı atla
    filtered_recommendations = []
    sorted_idx = similarities.argsort()[::-1]
    for i in sorted_idx:
        # Tüm sanatçılardan en az biri kullanıcı listesinde ise atla
        rec_artists = [a.strip().lower() for a in df.iloc[i]['artists_parsed']]
        if any(a in user_artists for a in rec_artists):
            continue
        filtered_recommendations.append({
            "name": df.iloc[i]['name'],
            "artists": df.iloc[i]['artists'],
            "genre": df.iloc[i]['genre'],
            "popularity": df.iloc[i]['popularity'],
            "year": df.iloc[i]['year']
        })
        if len(filtered_recommendations) >= top_n:
            break
    return filtered_recommendations


# ---- K-MEANS CLUSTER + PCA PIPELINE ----

# ---------- KMEANS + PCA ----------
def cluster_and_reduce(data, n_clusters=5):
    # Yalnızca sayısal kolonlar
    numeric_features = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'key', 'mode'
    ]
    X = data[numeric_features].fillna(0)

    # KMeans Pipeline: StandardScaler + KMeans
    cluster_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])
    cluster_pipeline.fit(X)
    cluster_labels = cluster_pipeline.named_steps['kmeans'].labels_
    data['cluster_label'] = cluster_labels

    # PCA Pipeline: StandardScaler + PCA
    pca_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2))
    ])
    embedding = pca_pipe.fit_transform(X)
    data['pca_x'] = embedding[:,0]
    data['pca_y'] = embedding[:,1]
    return data

def plot_clusters_plotly(data):
    fig = px.scatter(
        data, x='pca_x', y='pca_y', color='cluster_label',
        hover_data=['name', 'artists', 'genre', 'year'],
        title="Şarkıların K-means Kümeleme Görselleştirmesi (PCA ile)",
        template='plotly_dark', width=820, height=500  # Biraz daha küçük bırak
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='DarkSlateGrey')))
    html = fig.to_html(full_html=False)
    # Ortalamak için dış div: margin: 0 auto; display: block; ile!
    html = f'''
    <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
        <div style="width: 820px; margin: 0 auto;">{html}</div>
    </div>
    '''
    return html




# ----------------- FLASK ROUTES ------------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/recommender", methods=["GET", "POST"])
def recommender():
    if request.method == "POST":
        songs_raw = request.form.get("songs", "")
        song_lines = songs_raw.splitlines()
        recommendations = get_recommendations_multi(song_lines)
        return render_template("results.html", songs=song_lines, recommendations=recommendations)
    return render_template("index.html")

@app.route("/search_song", methods=["POST"])
def search_song():
    data = request.get_json()
    query = data.get("song", "")
    artist_query = data.get("artist", "")
    if not query:
        return jsonify(results=[])
    results = df[df['name'].str.lower().str.contains(query.lower(), na=False)]
    if artist_query:
        results = results[results['artists'].str.lower().str.contains(artist_query.lower(), na=False)]
    results = results.sort_values("popularity", ascending=False).head(10)
    out = []
    for _, row in results.iterrows():
        out.append({
            "name": row['name'],
            "artists": ", ".join(row['artists_parsed']),
            "full_str": f"{row['name']} - {', '.join(row['artists_parsed'])}"
        })
    return jsonify(results=out)



@app.route('/songs')
def songs():
    # K-means kümeleme + PCA
    clustered_df = cluster_and_reduce(df.copy(), n_clusters=5)
    cluster_html = plot_clusters_plotly(clustered_df)

    # İstatistikler
    stats = {
        'Toplam Şarkı': len(df),
        'Farklı Sanatçı': len(set(a for lst in df['artists_parsed'] for a in lst)),
        'Yıllar': f"{df['year'].min()} — {df['year'].max()}",
        'En Popüler Şarkı': df.loc[df['popularity'].idxmax()]['name'],
        'En Popüler Sanatçı': Counter([a for lst in df['artists_parsed'] for a in lst]).most_common(1)[0][0],
        'Ortalama Akustiklik': round(df['acousticness'].mean(), 3),
        'Ortalama Enerji': round(df['energy'].mean(), 3),
        'Ortalama Dans Edilebilirlik': round(df['danceability'].mean(), 3),
        'Ortalama Tempo': round(df['tempo'].mean(), 1),
        'Ortalama Popülerlik': round(df['popularity'].mean(), 2)
    }
    # Sadece ilk 100 şarkıyı tabloya gönderiyoruz
    song_rows = clustered_df.head(100).to_dict('records')

    return render_template('songs.html',
                           songs=song_rows,
                           stats=stats,
                           cluster_html=cluster_html)






import tempfile, os
from flask import request, render_template

@app.route('/feature-extraction', methods=['GET', 'POST'])
def feature_extraction():
    features = None
    error = None
    filename = None
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            flash('Dosya seçilmedi!', 'danger')
            return redirect(request.url)
        file = request.files['file']
        # Sadece mp3 veya wav dosyalarına izin verelim
        if not (file.filename.lower().endswith('.mp3') or file.filename.lower().endswith('.wav')):
            flash('Lütfen bir MP3 veya WAV dosyası yükleyin.', 'danger')
            return redirect(request.url)
        # Geçici dosya olarak kaydet
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            file.save(tmp.name)
            filename = file.filename
            try:
                # Dosyayı oku
                y, sr = librosa.load(tmp.name, sr=None)
                features = {
                    'Tempo': float(librosa.beat.tempo(y=y, sr=sr)[0]),
                    'MFCC Mean': float(np.mean(librosa.feature.mfcc(y=y, sr=sr))),
                    'Chroma Mean': float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr))),
                    'Spectral Centroid Mean': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                    'Spectral Bandwidth Mean': float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
                    'Spectral Rolloff Mean': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
                    'Zero Crossing Rate Mean': float(np.mean(librosa.feature.zero_crossing_rate(y=y))),
                    'RMS Mean': float(np.mean(librosa.feature.rms(y=y))),
                    'Duration (s)': float(librosa.get_duration(y=y, sr=sr))
                }
            except Exception as e:
                error = f"Özellik çıkarılırken hata oluştu: {e}"
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
    return render_template('feature_extraction.html', features=features, filename=filename, error=error)



@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        hashed_pw = generate_password_hash(password)
        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_pw))
            db.commit()
            cursor.close()
            db.close()
            flash("Kayıt başarılı, şimdi giriş yapabilirsin!", "success")
            return redirect(url_for('login'))
        except mysql.connector.errors.IntegrityError:
            error = "Bu kullanıcı adı zaten var!"
        except Exception as e:
            error = f"Kayıt sırasında hata: {e}"
    return render_template('register.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()
        cursor.close()
        db.close()
        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            
            return redirect(url_for('profile'))
        else:
            error = "Kullanıcı adı veya şifre yanlış"
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Çıkış yapıldı!', 'success')
    return redirect(url_for('login'))



@app.route('/listen', methods=['POST'])
def listen():
    if 'username' not in session:
        return jsonify({'error': 'Giriş yapmalısınız.'}), 401
    song_name = request.form['song_name']
    song_artist = request.form['song_artist']
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT INTO listened_songs (username, song_name, song_artist) VALUES (%s, %s, %s)",
                (session['username'], song_name, song_artist))
    db.commit()
    cur.close()
    db.close()
    return jsonify({'status': 'ok'})


@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))

    db = get_db()
    cur = db.cursor(dictionary=True)

    # Son 10 dinlenen şarkı
    cur.execute("""
        SELECT song_name, song_artist 
        FROM listened_songs 
        WHERE username=%s 
        ORDER BY listened_at DESC 
        LIMIT 10
    """, (session['username'],))
    listened_songs = cur.fetchall()
    cur.close()
    db.close()

    # Sana özel öneriler (son dinlenenlerden)
    if listened_songs:
        user_songs = [f"{row['song_name']} - {row['song_artist']}" for row in listened_songs]
        recommendations = get_recommendations_multi(user_songs)
    else:
        recommendations = []

    # En popüler 20 şarkıyı veri dosyasından çek
    top_songs = df.sort_values("popularity", ascending=False).head(20)
    populer_songs = []
    for _, row in top_songs.iterrows():
        # artists birden çoksa virgül ile birleştir
        if isinstance(row['artists'], list):
            artists = ', '.join(row['artists'])
        elif isinstance(row['artists'], str) and row['artists'].startswith('['):
            try:
                artists = ', '.join(ast.literal_eval(row['artists']))
            except:
                artists = row['artists']
        else:
            artists = row['artists']
        populer_songs.append({'name': row['name'], 'artists': artists})

    return render_template('profile.html',
        username=session['username'],
        listened_songs=listened_songs,
        recommendations=recommendations,
        populer_songs=populer_songs   # önemli: populer_songs gönder!
    )


@app.route('/delete_listened', methods=['POST'])
def delete_listened():
    if 'username' not in session:
        return jsonify({'error': 'Giriş yapmalısınız.'}), 401
    song_name = request.form['song_name']
    song_artist = request.form['song_artist']
    db = get_db()
    cur = db.cursor()
    # Son dinleneni (en yeni kaydı) sil
    cur.execute("""
        DELETE FROM listened_songs 
        WHERE username=%s AND song_name=%s AND song_artist=%s
        ORDER BY listened_at DESC LIMIT 1
    """, (session['username'], song_name, song_artist))
    db.commit()
    cur.close()
    db.close()
    return jsonify({'status': 'ok'})





# ----------------- MAIN ------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
