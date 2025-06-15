
.. raw:: html

   <div style="display: flex; align-items: center; margin-bottom: 20px;">
     <img src="_static/images/logo.png" alt="Logo" width="80" style="margin-right: 20px;">
     <h1 style="margin: 0;">Fruit Ninja V8 – Détection d'Objets et Joueur Automatique Intelligent</h1>
   </div>

=====================================================================

L'intelligence artificielle progresse chaque jour un peu plus dans l'imitation des capacités humaines : comprendre le langage, voir le monde, raisonner, parler… Mais un aspect reste souvent dans l'ombre : le jeu.

Et pourtant, le jeu n'est pas un simple divertissement. C'est un terrain d'expérimentation, un espace où l'on teste les limites de nos algorithmes, de nos idées, de notre imagination.

Depuis les Jeux olympiques antiques jusqu'aux jeux vidéo d'aujourd'hui, le jeu a toujours été un miroir de l'intelligence humaine, un moteur d'innovation.

Dans cette optique, notre projet s'inscrit dans une aventure aussi technique que ludique : créer une IA capable de jouer à Fruit Ninja.
Un jeu rapide, visuel, imprévisible… idéal pour confronter l'intelligence artificielle au monde réel — ou presque.

Objectifs du Projet
------------------

* Entraîner un modèle de détection d'objets léger et en temps réel pour Fruit Ninja.
* Intégrer une stratégie de découpe automatique intelligente utilisant **l'algorithme A*** pour couper les fruits tout en évitant les bombes.
* Capturer les images d'écran, détecter les objets et contrôler les mouvements de découpe pour battre le **score le plus élevé possible** dans le jeu.

------

Structure du Projet
------------------

.. code-block:: bash

    fruit-ninja-v8/
    ├── src/                  
    │   ├── Fruit.py          # Définit la structure d'un objet
    │   ├── Astar.py          # Logique de pathfinding A* pour une découpe optimale
    │   ├── VideoRecorder.py  # Enregistre le jeu pour analyse future
    │   ├── ScreenCapture.py  # Capture l'écran de jeu en temps réel
    │   ├── main.py           # script principal
    |   └── requirement.txt   # les bibliothèques utilisées
    ├── Model YOLOv8n/
    │   ├── fruit_ninja_v8.pt       # Modèle YOLOv8-nano entraîné
    |   ├── Confusion Matrix.jpeg   # Matrice de Confusion
    |   ├── Metrics.jpeg            # Métriques
    |   ├── Validation.jpeg         # Quelques Images de Validation
    ├── README.md             # Aperçu du projet et instructions d'installation

------

Modèle IA : YOLOv8-Nano
----------------------

* **Architecture** : YOLOv8-nano (Ultralytics)
* **Classes** : Fruits (banane, pomme, etc.), Bombes
* **Taille d'entrée** : 640x640
* **Données d'entraînement** : ~1500 images capturées du gameplay
* **Format d'annotation** : Fichiers ``.txt`` style YOLOv5/8

Métriques d'Évaluation :
^^^^^^^^^^^^^^^^^^^^^^^

* **Précision, Rappel, mAP** presque 1.0 (vérifier Model YOLOv8n/Metrics.png)
* **Matrice de confusion** visualisée dans ``/metrics/``

IA de Découpe – Optimiseur de Chemin A*
--------------------------------------

Pour maximiser le score tout en minimisant le risque :

* Utilise **l'algorithme A*** pour calculer le chemin optimal à travers les groupes de fruits.
* Évite les bombes et planifie les coupes qui maximisent les combos et bonus.

Fonctionnement
-------------

1. **Capture d'Écran et Début d'Enregistrement si nécessaire** : Capture continuellement les images du jeu.
2. **Détection** : Passe l'image au modèle YOLOv8.
3. **Analyse** : Trouve les positions des fruits, évite les bombes.
4. **Contrôleur A*** : Calcule le meilleur chemin de balayage.
5. **Déclencheur d'Entrée** : Envoie le geste de découpe via l'entrée souris/tactile du système.

En Cours de Développement
------------------------

* Amélioration de la logique d'évitement des bombes
* Utilisation d'une Capture d'Écran plus rapide au lieu de mss

Exécuter le Projet
-----------------

Cloner le dépôt

.. code-block:: bash

    git clone https://github.com/MA-Zbida/Fruit-Ninja-AI.git
    cd Fruit-Nnja-AI

Installer les Dépendances et exécuter le script

.. code-block:: bash

    pip install -r src/requirements.txt
    python src/main.py

Contact
-------

Pour des questions ou une collaboration :

**[ABDERRAZAK KHALIL] – Étudiant en Ingénierie IA**

**[ZBIDA MOHAMMED AMINE] – Étudiant en Ingénierie IA**

Email : `[Abderrazak Khalil] <mailto:khalilabderrazak1@gmail.com>`_

Email : `[Mohamed Amine Zbida] <mailto:itzzbida@gmail.com>`_


.. toctree::
   :maxdepth: 2
   :caption: Contenus:

   installation
   usage
   modules
   index2
   index3
   index4
   index5