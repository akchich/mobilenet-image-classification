from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
import os

import numpy as np
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image


app = Flask(__name__)

# Dossier où les images seront enregistrées
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Connexion MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["tp_tensorflow"]
collection = db["Image"]

# Chargement du modèle TensorFlow
model = MobileNetV2(weights="imagenet")


def donner_badge(confiance):
    """
    Cette fonction donne un badge selon le taux de confiance.
    C'est notre petite partie ludique/gamification.
    """
    if confiance >= 80:
        return "Très fiable"
    elif confiance >= 50:
        return "Moyennement fiable"
    else:
        return "À vérifier"


def classer_type(label):
    """
    MobileNetV2 donne une prédiction en anglais.
    Ici, on transforme la prédiction en grande catégorie :
    Animal, Véhicule, Plante, Humain ou Personnage fictif.
    """

    label = label.lower()

    animaux = [
        "dog", "cat", "lion", "tiger", "bear", "elephant", "horse",
        "zebra", "monkey", "bird", "fish", "shark", "whale", "snake",
        "frog", "rabbit", "hamster", "mouse", "fox", "wolf", "cow",
        "sheep", "goat", "pig", "hen", "duck", "eagle", "owl",
        "leopard", "jaguar", "panda", "koala"
    ]

    vehicules = [
        "car", "truck", "bus", "train", "bicycle", "bike", "motorcycle",
        "airplane", "plane", "boat", "ship", "scooter", "ambulance",
        "taxi", "jeep", "limousine", "tractor"
    ]

    plantes = [
        "plant", "flower", "tree", "mushroom", "daisy", "rose",
        "sunflower", "cactus", "palm", "corn", "acorn"
    ]

    humains = [
        "person", "man", "woman", "boy", "girl", "bridegroom",
        "groom", "baseball_player", "scuba_diver"
    ]

    personnages = [
        "yoda", "pikachu", "mario", "cartoon", "doll", "puppet",
        "toy", "comic"
    ]

    for mot in animaux:
        if mot in label:
            return "Animal"

    for mot in vehicules:
        if mot in label:
            return "Véhicule"

    for mot in plantes:
        if mot in label:
            return "Plante"

    for mot in humains:
        if mot in label:
            return "Humain"

    for mot in personnages:
        if mot in label:
            return "Personnage fictif"

    return "Autre"


def analyser_image(chemin_image):
    """
    Cette fonction prépare l'image, l'envoie au modèle TensorFlow,
    puis récupère la meilleure prédiction.
    """

    img = image.load_img(chemin_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)

    resultat = decode_predictions(predictions, top=1)[0][0]

    label = resultat[1]
    confiance = round(float(resultat[2]) * 100, 2)
    categorie = classer_type(label)
    badge = donner_badge(confiance)

    return label, categorie, confiance, badge


@app.route("/", methods=["GET", "POST"])
def index():
    resultat = None

    if request.method == "POST":
        fichier = request.files.get("image")

        if fichier and fichier.filename != "":
            nom_image = fichier.filename
            chemin = os.path.join(app.config["UPLOAD_FOLDER"], nom_image)
            fichier.save(chemin)

            taille = os.path.getsize(chemin)

            prediction_originale, categorie, confiance, badge = analyser_image(chemin)

            document = {
                "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "nom_image": nom_image,
                "taille_octets": taille,
                "chemin_image": chemin,
                "analyse": {
                    "prediction_originale": prediction_originale,
                    "type_reconnu": categorie,
                    "taux_reussite": confiance,
                    "badge": badge
                }
            }

            collection.insert_one(document)

            resultat = document

    historiques = list(collection.find().sort("_id", -1))

    return render_template(
        "index.html",
        resultat=resultat,
        historiques=historiques
    )


@app.route("/supprimer/<id>")
def supprimer(id):
    collection.delete_one({"_id": ObjectId(id)})
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)