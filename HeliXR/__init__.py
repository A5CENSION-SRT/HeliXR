from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_socketio import SocketIO

app = Flask(__name__)

app.config['SECRET_KEY'] = 'c1877bdc3305c942f87b10f86e246167'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

app.mongo_db = None
app.mongo_collection = None

app.config['MONGO_URI'] = "mongodb+srv://snehalreddy:S0OcbrCRXJmAZrAd@sudarshan-chakra-cluste.0hokvj0.mongodb.net/sudarshan-chakra"
app.config['MONGO_DB_NAME'] = "radarDB"
app.config['MONGO_COLLECTION_NAME'] = "scans"

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager=LoginManager(app)
socketio = SocketIO(app)




from HeliXR import routes

# Create database tables for SQLAlchemy
with app.app_context():
    db.create_all()