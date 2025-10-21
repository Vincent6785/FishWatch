# FishWatch  
  
## Description  
Un exemple d'entraînement spécifique du modèle YOLO pour la détection de poissons dans un aquarium.  
  
#### Besoin  
Dans mon aquarium, j’ai 11 poissons, dont 5 jeunes [scalaires](https://www.fishipedia.fr/fr/poissons/pterophyllum-scalare)

Ces scalaires formeront bientôt des couples et défendront leurs territoires.  
Or, je ne peux pas assurer une surveillance quotidienne. Il me faut une solution automatisée.
  
Par surveillance, j’entends uniquement :  
1. M’assurer de l’état de santé des 11 poissons. La médecine des poissons étant peu développée, je me contenterai d’un statut mort/vivant.  
2. Observer le comportement des scalaires pour répondre à la question : *des couples se forment-ils ?*
  
#### Solution métier
Pour s’assurer visuellement de l’état de santé d’un poisson, il faut détecter sa présence avec un haut niveau de certitude.  
Un poisson mort disparaît souvent en moins de 48 heures.  
La présence est donc un premier indicateur de santé. 
  
Pour constater la formation de couples, il faut détecter la localisation des scalaires avec précision.
C’est cette précision qui déterminera la qualité du rendu.

#### Solution technique
Il faut : 
1. Filmer l'aquarium.
2. Détecter, dans les images, les poissons et leurs localisations.

#### Capture d'images et dataset
La caméra que j'utilise filme en 720p à 30 fps.

Les images sont prises face à l'aquarium, pendant la plage horaire d'éclairage (15h - 21h30).  
Pour entraîner au mieux le modèle, j'ai utilisé la même caméra, disposée de la même manière pour les images d'entraînement que pour la surveillance réelle.

Pour créer le dataset, j'ai utilisé [Label Studio](https://labelstud.io/).  
Dans un premier temps, j'ai labellisé manuellement 500 images.  
Puis j'ai entraîné YOLOv8m spécifiquement sur ce dataset jusqu'au surapprentissage.

Avec ce premier modèle, j'ai pu pré-labelliser les images et ainsi gagner beaucoup de temps.

J'ai maintenant 1 500 images contenant entre 330 et 1480 labels par classe.
Plus précisément, concernant les scalaires, il y a entre 385 et 1468 labels par classe.

#### Entraînement
<img width="1134" height="963" alt="Capture d'écran 2025-10-18 124402" src="https://github.com/user-attachments/assets/52eea95d-dd95-41f6-9205-0c5f71716f3b" />

#### Bilan
Avant entrainement:

<img width="646" height="430" alt="Capture d'écran 2025-10-18 122941" src="https://github.com/user-attachments/assets/13d29d7a-8c24-4c6e-8599-86e1da1057dd" />
<img width="604" height="498" alt="Capture d'écran 2025-10-18 122835" src="https://github.com/user-attachments/assets/8c037b06-e1b8-497e-8231-2d4fe55b64ba" />

Après entrainement:

![exemple](https://github.com/user-attachments/assets/0ee5edc1-eca1-4efb-9a00-dd727593a5ca)

Nota bene: la classe Corydoras_Panda est mal nommée. Elle englobe en réalité deux Corydoras dont j'ai oublié l'espèce exacte.

#### Axes d'amélioration
  
###### Quick win
L’évaluation de l’état de santé repose sur la détection de la présence de chaque poisson.
Cette observation peut être affinée en considérant :  
- un poisson qui flotte à la surface,  
- ou, au contraire, qui reste immobile au fond de l'aquarium.

###### Heavy lift
Augmenter le nombre d'images du dataset. 1500 images, c'est trop peu.

Changer d'approche et entrainer un modèle pour reconnaitre une espece et non un individu. On perdrait en finesse mais gagnerait en adaptabilité en cas de changement d'aquarium.
