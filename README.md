\=======================================================
FIZY SONG RECOMMENDATION SYSTEM
===============================

Developed during my internship at Turkcell (Fizy Department).
This project demonstrates how advanced data science and software engineering practices can be implemented in a real-world, corporate setting with a focus on modern recommendation algorithms and robust backend architecture.

**Project Overview**

Fizy Song Recommendation System is a comprehensive, scalable music recommendation engine built with Python (Flask), MySQL, and Docker. The system utilizes both **content-based filtering** and **collaborative filtering** approaches to deliver accurate and relevant song suggestions to users, leveraging a real-world dataset from Spotify (Kaggle).

**Key Algorithms & Techniques**

* **Content-Based Filtering:**
  Each song is represented by a rich set of features (acousticness, energy, danceability, valence, tempo, etc.). Songs are recommended by calculating cosine similarity between feature vectors of songs the user likes and all other songs in the dataset, excluding previously listened ones and songs by the same artist for diversity.

* **Collaborative Filtering:**
  The system records users’ listening histories in MySQL and uses these to model implicit preferences. By aggregating the preferences of users with similar tastes, the system can suggest tracks that align with collective patterns, not just individual feature similarity.

* **Clustering & Visualization:**
  K-Means clustering and PCA (Principal Component Analysis) are used for exploratory analysis and to visualize the song universe, helping to identify natural groupings and feature distributions among songs.

* **Audio Feature Extraction:**
  The application includes a module for extracting advanced features from uploaded MP3 or WAV files using Librosa, allowing integration of custom user audio data into the recommendation pipeline.

* **User Authentication & Profiles:**
  Secure user authentication (hashed passwords), listening history tracking, and profile-based personalized recommendations are all supported. Each user receives recommendations based on their recent listening activity.

**Dataset**

Source: Spotify Dataset on Kaggle
Description: Contains thousands of songs with features including artist, genre, valence, energy, danceability, popularity, release date, and more.

**System Architecture**

* **Backend:** Python (Flask)
* **Database:** MySQL (with Docker volume persistence)
* **Frontend:** HTML templates rendered by Flask (supports RESTful interaction)
* **Containerization:** Docker and Docker Compose for reproducible local/server deployment

**Deployment (Ubuntu Server)**

**Prerequisites:**

* Ubuntu 20.04/22.04 server
* Docker
* Docker Compose

**Step 1 - Clone the Repository**
git clone [https://github.com/YourUsername/Fizy-Song-Recommendation-System.git](https://github.com/YourUsername/Fizy-Song-Recommendation-System.git)
cd Fizy-Song-Recommendation-System

**Step 2 - Configure Environment**
Adjust any necessary environment variables (MySQL root password, DB name, etc.) in docker-compose.yml as needed.

**Step 3 - Place Dataset**
Download the Spotify Dataset from Kaggle and place the CSV files into the data/ directory.

**Step 4 - Build and Start**
docker-compose up --build -d

Flask API: [http://your-server-ip:5000](http://your-server-ip:5000)
MySQL: db:3306 (accessible from within Docker network)

**Step 5 - Logs & Troubleshooting**
docker-compose logs -f

**Step 6 - Stop the System**
docker-compose down

**Project Directory Structure**

Fizy-Song-Recommendation-System/

* app.py
* requirements.txt
* Dockerfile
* docker-compose.yml
* data/ (with spotify\_dataset.csv)
* templates/ (HTML UI)
* uploads/ (for user audio analysis)

**Usage & API**

* **User Registration/Login:**
  Users create an account, log in, and their listening history is securely tracked.
* **Song Recommendation:**
  Users can enter favorite songs or upload audio, and receive tailored recommendations combining both content and collaborative methods.
* **Audio Feature Extraction:**
  Upload your own song and get its extracted features for custom recommendation.
* **Visualization:**
  Explore the musical landscape with K-Means clusters and PCA projection.
* **REST API:**
  The backend is structured to allow easy integration with frontend apps or external services.

**Extensibility**

The system is designed for extensibility: new algorithms, more features, and third-party integrations (Spotify API, user rating, etc.) can easily be added.

**Acknowledgements**

This project was developed as part of my internship at Turkcell, in the Fizy Department.
Special thanks to my mentors and the Fizy team for their continuous guidance and technical support.

**License**

Licensed under the MIT License.

**Contact**

For technical questions, suggestions, or collaboration:
Name: Burak Samet Varel
Email: [varelburaksamet38@gmail.com](mailto:varelburaksamet38@gmail.com)
LinkedIn: [https://www.linkedin.com/in/burak-samet-varel-30b989227/](https://www.linkedin.com/in/burak-samet-varel-30b989227/)
GitHub: [https://github.com/BurakSametVarel](https://github.com/BurakSametVarel)

---

İstersen başlığı sadece büyük harflerle de bırakabilirim veya daha sade bir çizgiyle ayırabilirim. Hangisini tercih edersin?
