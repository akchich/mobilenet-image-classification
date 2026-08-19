# MobileNet Image Classification App

Application web développée avec Flask permettant d'importer une image, de l'analyser avec un modèle MobileNetV2 pré-entraîné sur ImageNet, puis d'enregistrer le résultat de l'analyse dans MongoDB.

Le projet combine vision par ordinateur, intégration d'un modèle de Deep Learning, développement web avec Flask et persistance des données avec MongoDB.

## Apprentissages autour du projet

Ce projet m'a permis d'approfondir le fonctionnement des réseaux de neurones convolutifs (CNN) appliqués à la reconnaissance d'images.

Notions étudiées et manipulées autour du projet :

* représentation d'une image sous forme de pixels et de valeurs RGB ;
* convolution et utilisation de filtres pour détecter des caractéristiques ;
* feature maps ;
* fonction d'activation ReLU ;
* pooling et réduction de dimension ;
* principe de classification d'images ;
* Softmax et scores de confiance ;
* différence entre entraînement et inférence ;
* utilisation d'un modèle pré-entraîné ;
* rôle de TensorFlow dans l'exécution d'un modèle de Deep Learning ;
* limites d'un modèle lorsqu'il analyse des classes différentes de celles présentes dans son jeu d'entraînement.

Le modèle MobileNetV2 utilisé dans ce projet n'a pas été entraîné à partir de zéro. Il est chargé avec des poids pré-entraînés sur ImageNet.

## Fonctionnalités

L'application permet de :

* importer une image depuis une interface web ;
* enregistrer temporairement l'image sur le serveur ;
* redimensionner l'image au format attendu par MobileNetV2 ;
* préparer l'image avant son passage dans le modèle ;
* effectuer une prédiction avec MobileNetV2 ;
* récupérer la classe ayant le meilleur score ;
* afficher le taux de confiance associé ;
* regrouper certaines prédictions dans une catégorie simplifiée ;
* attribuer un badge selon le taux de confiance ;
* enregistrer les résultats dans MongoDB ;
* afficher l'historique des analyses ;
* supprimer une analyse enregistrée.

## Catégories utilisées dans l'application

L'application essaie de regrouper les prédictions obtenues dans les catégories suivantes :

* Animal
* Véhicule
* Plante
* Humain
* Personnage fictif
* Autre

Cette catégorisation est réalisée à partir du label retourné par MobileNetV2.

## Système de confiance

Un badge est attribué selon le score de confiance du modèle :

* **Très fiable** : confiance supérieure ou égale à 80 %
* **Moyennement fiable** : confiance entre 50 % et 80 %
* **À vérifier** : confiance inférieure à 50 %

Ce système permet de donner à l'utilisateur une indication simple sur le niveau de confiance de la prédiction.

## Architecture simplifiée

```text
Utilisateur
    ↓
Upload de l'image
    ↓
Flask
    ↓
Prétraitement
    ↓
MobileNetV2
    ↓
Prédiction
    ↓
Catégorisation + score de confiance
    ↓
MongoDB
    ↓
Affichage du résultat et de l'historique
```

## Technologies utilisées

### Intelligence artificielle

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy

### Développement web

* Flask
* HTML

### Base de données

* MongoDB
* PyMongo

## Fonctionnement de la partie IA

### 1. Chargement de l'image

L'image sélectionnée par l'utilisateur est chargée et redimensionnée en :

```text
224 × 224 pixels
```

Ce format correspond au format d'entrée utilisé par MobileNetV2 dans le projet.

### 2. Transformation en tableau numérique

L'image est transformée en tableau NumPy afin de pouvoir être manipulée par le modèle.

### 3. Ajout de la dimension batch

Le tableau représentant une image passe d'une forme similaire à :

```text
(224, 224, 3)
```

à :

```text
(1, 224, 224, 3)
```

La première dimension représente le nombre d'images envoyées au modèle.

### 4. Prétraitement

La fonction :

```python
preprocess_input()
```

prépare les valeurs de l'image selon le format attendu par MobileNetV2.

### 5. Inférence

Le modèle effectue ensuite une prédiction avec :

```python
model.predict()
```

L'application récupère la meilleure prédiction avec :

```python
decode_predictions()
```

Le résultat contient notamment :

* le label prédit ;
* le score de confiance.

## MobileNetV2

Le projet utilise :

```python
MobileNetV2(weights="imagenet")
```

Cela signifie que l'application charge un modèle MobileNetV2 déjà pré-entraîné sur ImageNet.

Le travail réalisé dans ce projet porte principalement sur :

* l'intégration du modèle dans une application ;
* la préparation des images ;
* l'exécution de l'inférence ;
* l'exploitation des prédictions ;
* la catégorisation des résultats ;
* la persistance des analyses dans MongoDB.

## Base de données MongoDB

Le projet utilise une base MongoDB locale.

```text
Base : tp_tensorflow
Collection : Image
```

Chaque analyse enregistrée contient notamment :

```text
date
nom de l'image
taille du fichier
chemin de l'image
prédiction originale
catégorie reconnue
taux de confiance
badge
```

Exemple simplifié :

```json
{
  "date": "12/06/2026 02:00:00",
  "nom_image": "image.jpg",
  "taille_octets": 32500,
  "analyse": {
    "prediction_originale": "lion",
    "type_reconnu": "Animal",
    "taux_reussite": 53.4,
    "badge": "Moyennement fiable"
  }
}
```

## Résultats et limites observées

Le projet a permis d'obtenir des prédictions cohérentes sur certaines images, notamment des animaux ou des plantes.

Des erreurs ont également été observées sur des images représentant des objets ou personnages qui ne correspondent pas directement aux classes connues par le modèle.

Par exemple, durant les tests :

* certaines fleurs ont été correctement reconnues ;
* certains animaux ont été correctement identifiés ;
* des objets particuliers ou personnages fictifs ont parfois été associés à une classe ImageNet différente.

Ces résultats montrent qu'un modèle pré-entraîné reste dépendant des catégories et des données utilisées lors de son entraînement.

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/akchich/mobilenet-image-classification.git
```

### 2. Entrer dans le dossier

```bash
cd mobilenet-image-classification
```

### 3. Créer un environnement virtuel

```bash
python -m venv venv
```

### 4. Activer l'environnement virtuel sous Windows

```bash
venv\Scripts\activate
```

### 5. Installer les dépendances

```bash
pip install flask pymongo tensorflow numpy pillow
```

## Configuration de MongoDB

MongoDB doit être installé et lancé localement.

L'application utilise actuellement :

```text
mongodb://localhost:27017/
```

## Lancement de l'application

Lancer l'application avec :

```bash
python app.py
```

Puis ouvrir :

```text
http://127.0.0.1:5000
```

dans un navigateur.

## Structure du projet

```text
mobilenet-image-classification/
│
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── uploads/
├── .gitignore
└── README.md
```

Le dossier `venv` n'est pas versionné sur GitHub.

## Améliorations possibles

Plusieurs améliorations pourraient être ajoutées :

* utiliser un fichier `.env` pour l'URI MongoDB ;
* ajouter un fichier `requirements.txt` ;
* améliorer la gestion des erreurs ;
* sécuriser davantage l'upload des fichiers ;
* vérifier le type et la taille des images importées ;
* utiliser une méthode HTTP plus adaptée pour la suppression ;
* améliorer la catégorisation des résultats ;
* afficher plusieurs prédictions au lieu d'une seule ;
* entraîner ou adapter un modèle à des catégories spécifiques ;
* améliorer l'interface utilisateur.

## Objectif du projet

L'objectif principal de ce projet était de comprendre comment intégrer un modèle de vision par ordinateur pré-entraîné dans une application web complète, depuis l'import de l'image jusqu'à l'enregistrement et l'affichage du résultat.

Le projet m'a également permis d'approfondir les concepts fondamentaux des CNN et de mieux comprendre les différences entre l'utilisation d'un modèle pré-entraîné, l'inférence et l'entraînement d'un modèle.
