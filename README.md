Smart Song Recommendation System

Developed during my internship at Turkcell (Fizy Department).
This project demonstrates the application of data science and software engineering practices in a real-world, corporate setting.

Overview

Fizy Song Recommendation System is a scalable and hybrid music recommendation engine that combines content-based and collaborative filtering methods. The system uses real-world music data from the Spotify Dataset ([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)) available on Kaggle.

The project is fully containerized with Docker and is designed for seamless deployment on Ubuntu Server environments.

Features

* Content-Based Filtering (using song features and metadata)
* Collaborative Filtering (user preferences and listening history)
* RESTful API for recommendations
* MySQL database integration
* Dockerized for production and development
* Easily extensible and configurable

Dataset

Source: Spotify Dataset on Kaggle ([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets))
Description: Contains thousands of songs with features such as valence, energy, danceability, release date, and more.

Deployment (Ubuntu Server)

Prerequisites:

* Ubuntu 20.04/22.04 server
* Docker ([https://docs.docker.com/engine/install/ubuntu/](https://docs.docker.com/engine/install/ubuntu/))
* Docker Compose ([https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/))

1. Clone the Repository

git clone [https://github.com/YourUsername/Fizy-Song-Recommendation-System.git](https://github.com/YourUsername/Fizy-Song-Recommendation-System.git)
cd Fizy-Song-Recommendation-System

2. Configure Environment Variables

Edit the docker-compose.yml file if you need to set custom MySQL passwords or change ports.

3. Place Dataset

Download the Spotify Dataset ([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)) from Kaggle and place the CSV file(s) into the data/ directory of the project.

4. Build and Run with Docker Compose

docker-compose up --build -d

The Flask API will be available at: [http://your-server-ip:5000](http://your-server-ip:5000)
MySQL will run inside a container and is accessible as db:3306 from other containers.

5. Check Logs

To view logs for troubleshooting:

docker-compose logs -f

6. Stopping the Application

docker-compose down

Project Structure

Fizy-Song-Recommendation-System/

* app.py
* requirements.txt
* Dockerfile
* docker-compose.yml
* data/

  * spotify\_dataset.csv
* ...

Acknowledgements

This project was developed as part of my internship at Turkcell, in the Fizy Department.
Special thanks to my mentors and the Fizy team for their valuable support and guidance.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to open issues or contribute!

Contact

For any questions, feedback, or collaboration requests, please feel free to reach out:

Name: Burak Samet Varel
Email: [buraksametvarel@gmail.com](mailto:buraksametvarel@gmail.com)
LinkedIn: [https://www.linkedin.com/in/buraksametvarel/](https://www.linkedin.com/in/buraksametvarel/)
GitHub: [https://github.com/BurakSametVarel](https://github.com/BurakSametVarel)

