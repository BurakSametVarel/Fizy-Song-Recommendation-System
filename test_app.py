import pytest
import os
import mysql.connector
from app import app, create_db_and_table, create_listened_table

MYSQL_HOST = os.environ.get("MYSQL_HOST", "db")
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "BSVgs14531071.")
TEST_DB_NAME = os.environ.get("MYSQL_DB", "fizy_test")

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Test başlamadan test veritabanını oluşturur, bitince siler."""
    os.environ["MYSQL_DB"] = TEST_DB_NAME
    conn = mysql.connector.connect(
        host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {TEST_DB_NAME}")
    cursor.close()
    conn.close()
    create_db_and_table()
    create_listened_table()
    yield
    # Testler bitince veritabanı temizlenir.
    conn = mysql.connector.connect(
        host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
    cursor.close()
    conn.close()

@pytest.fixture
def client():
    """Her testte izole bir Flask client nesnesi sağlar."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page_ok(client):
    # Ana sayfa yükleniyor mu? (HTTP 200 ve temel içerik var mı)
    response = client.get('/')
    data = response.data.decode()
    assert response.status_code == 200
    assert "Spotify" in data or "Recommender" in data or "Şarkı" in data  # Ana sayfa içeriği kontrolü

def test_register_success(client):
    # Kullanıcı başarıyla kayıt olabiliyor mu?
    response = client.post('/register', data={'username': 'user1', 'password': 'pass1'}, follow_redirects=True)
    data = response.data.decode()
    assert ("Kayıt başarılı" in data) or ("şimdi giriş yapabilirsin" in data) or ("Başarıyla" in data)

def test_register_duplicate_username(client):
    # Aynı kullanıcı adıyla tekrar kayıt denendiğinde hata dönüyor mu?
    client.post('/register', data={'username': 'userdup', 'password': 'pass'}, follow_redirects=True)
    response = client.post('/register', data={'username': 'userdup', 'password': 'pass'}, follow_redirects=True)
    data = response.data.decode()
    assert "zaten var" in data or "var!" in data

def test_login_success(client):
    # Kayıtlı kullanıcı doğru şifreyle giriş yapabiliyor mu?
    client.post('/register', data={'username': 'user2', 'password': 'pass2'})
    response = client.post('/login', data={'username': 'user2', 'password': 'pass2'}, follow_redirects=True)
    data = response.data.decode()
    assert "Profil" in data or "profile" in data or "user2" in data

def test_login_wrong_password(client):
    # Kullanıcı yanlış şifreyle giriş yapamıyor ve hata mesajı alıyor mu?
    client.post('/register', data={'username': 'user3', 'password': 'pass3'})
    response = client.post('/login', data={'username': 'user3', 'password': 'wrongpass'}, follow_redirects=True)
    data = response.data.decode()
    assert "yanlış" in data or "error" in data

def test_login_nonexistent_user(client):
    # Hiç kayıt olmayan kullanıcı giriş yapamıyor mu?
    response = client.post('/login', data={'username': 'userx', 'password': 'passx'}, follow_redirects=True)
    data = response.data.decode()
    assert "yanlış" in data or "error" in data

def test_logout_success(client):
    # Kullanıcı çıktıktan sonra login ekranına yönlendiriliyor mu?
    client.post('/register', data={'username': 'logoutu', 'password': 'pass'})
    client.post('/login', data={'username': 'logoutu', 'password': 'pass'}, follow_redirects=True)
    response = client.get('/logout', follow_redirects=True)
    data = response.data.decode()
    assert "Çıkış yapıldı" in data or "login" in data or "Giriş" in data

def test_profile_requires_login(client):
    # Giriş yapılmadan /profile'a erişilirse login ekranına yönlendiriliyor mu?
    response = client.get('/profile', follow_redirects=True)
    data = response.data.decode()
    assert "Giriş" in data or "login" in data

def test_profile_access_and_recommendations(client):
    # Kullanıcı login olduktan sonra profile gidebiliyor, dinlenen ve önerilen şarkıları görebiliyor mu?
    client.post('/register', data={'username': 'profilci', 'password': 'abc123'})
    client.post('/login', data={'username': 'profilci', 'password': 'abc123'}, follow_redirects=True)
    client.post('/listen', data={'song_name': 'Shape of You', 'song_artist': 'Ed Sheeran'})
    response = client.get('/profile')
    data = response.data.decode()
    assert "Shape of You" in data  # Son dinlenen şarkı gösteriliyor mu?
    assert "Öneri" in data or "recommendation" in data or "Popüler" in data  # Öneri bölümü var mı?

def test_listen_and_delete_song(client):
    # Kullanıcı login olduktan sonra şarkı dinleyip, dinlediğini silebiliyor mu?
    client.post('/register', data={'username': 'musicman', 'password': '1111'})
    client.post('/login', data={'username': 'musicman', 'password': '1111'}, follow_redirects=True)
    response = client.post('/listen', data={'song_name': 'TestSong', 'song_artist': 'TestArtist'})
    data = response.data.decode()
    assert "ok" in data
    response = client.post('/delete_listened', data={'song_name': 'TestSong', 'song_artist': 'TestArtist'})
    data = response.data.decode()
    assert "ok" in data

def test_delete_nonexistent_listened_song(client):
    # Dinlenmemiş bir şarkıyı silmeye çalışınca sistem hata vermeden 'ok' dönüyor mu?
    client.post('/register', data={'username': 'deleteman', 'password': '4444'})
    client.post('/login', data={'username': 'deleteman', 'password': '4444'}, follow_redirects=True)
    response = client.post('/delete_listened', data={'song_name': 'RandomSong', 'song_artist': 'RandomArtist'})
    data = response.data.decode()
    assert "ok" in data

def test_listen_requires_login(client):
    # Kullanıcı login olmadan /listen endpointine erişirse 401 döner mi?
    response = client.post('/listen', data={'song_name': 'TestSong', 'song_artist': 'TestArtist'})
    assert response.status_code == 401

def test_delete_requires_login(client):
    # Kullanıcı login olmadan /delete_listened endpointine erişirse 401 döner mi?
    response = client.post('/delete_listened', data={'song_name': 'TestSong', 'song_artist': 'TestArtist'})
    assert response.status_code == 401

def test_logout_redirects_to_login(client):
    # Çıkış yapınca login ekranına döndürülüyor mu?
    response = client.get('/logout', follow_redirects=True)
    data = response.data.decode()
    assert "login" in data or "Giriş" in data

def test_song_search_api(client):
    # /search_song endpointine şarkı adı ile istek atınca ilgili sonuçlar dönüyor mu?
    response = client.post('/search_song', json={'song': 'Shape of You'})
    data = response.data.decode()
    assert response.status_code == 200
    assert "Shape of You" in data or "shape of you" in data

def test_song_search_empty_result(client):
    # Var olmayan bir şarkı arandığında sonuç dönmüyor mu?
    response = client.post('/search_song', json={'song': 'ASDJKLASDJKLASDJKL'})
    data = response.data.decode()
    assert response.status_code == 200
    assert "Shape of You" not in data and "Despacito" not in data

def test_recommender_post(client):
    # /recommender endpointine çoklu şarkı ismi ile istek atınca öneri listesi geliyor mu?
    data = {
        'songs': "Shape of You - Ed Sheeran\nDespacito - Luis Fonsi"
    }
    response = client.post('/recommender', data=data)
    data_str = response.data.decode()
    assert response.status_code == 200
    assert "Despacito" in data_str or "Shape of You" in data_str

def test_songs_page(client):
    # /songs sayfası başarıyla yükleniyor ve K-means/PCA cluster görseli geliyor mu?
    response = client.get('/songs')
    data = response.data.decode()
    assert response.status_code == 200
    assert "K-means" in data or "Kümeleme" in data or "Şarkı" in data

def test_register_blank_fields(client):
    # Boş kullanıcı adı veya şifre ile kayıt denendiğinde hata dönüyor mu?
    response = client.post('/register', data={'username': '', 'password': ''}, follow_redirects=True)
    data = response.data.decode()
    assert "hata" in data or "eksik" in data or "Kayıt" in data or "error" in data

def test_login_blank_fields(client):
    # Boş kullanıcı adı veya şifre ile login denendiğinde hata mesajı dönüyor mu?
    response = client.post('/login', data={'username': '', 'password': ''}, follow_redirects=True)
    data = response.data.decode()
    assert "yanlış" in data or "hata" in data or "error" in data
