# FishWatch  
  
## Description  
Un exemple d'entraînement spécifique du modèle YOLO pour la détection de poissons dans un aquarium.  
  
#### Besoin  
Dans mon aquarium, j’ai 11 poissons, mais ici on se concentre sur 5 jeunes scalaires. (cf. https://www.fishipedia.fr/fr/poissons/pterophyllum-scalare)

Ces poissons formeront bientôt des couples et défendront leurs territoires.  
Or, je ne peux pas assurer une surveillance quotidienne. Il me faut une solution automatisée.
  
Par surveillance, j’entends uniquement :  
1. M’assurer de l’état de santé des poissons. La médecine des poissons étant peu développée, je me contenterai d’un statut mort/vivant.  
2. Observer le comportement des poissons pour répondre à la question : *des couples se forment-ils ?*
  
#### Solution métier
Pour s’assurer visuellement de l’état de santé d’un poisson, il faut détecter sa présence avec un haut niveau de certitude.  
Un poisson mort disparaît souvent en moins de 48 heures.  
La présence est donc un premier indicateur de santé.  
On peut affiner avec :  
- un poisson qui flotte à la surface,  
- ou, à l’inverse, qui reste au fond.  
  
Pour constater la formation de couples, il faut détecter la localisation des poissons avec précision.
C’est cette précision qui déterminera la qualité du rendu.

#### Solution technique
Il faut : 
1. Filmer l'aquarium.
2. Détecter, dans les images, les poissons et leurs localisations.

#### Capture d'images et dataset
La caméra que j'utilise filme en 720p à 30 fps.

Les images sont prises face à l'aquarium, pendant la plage horaire d'éclairage (15h - 21h30).  
Pour entraîner au mieux le modèle, j'ai utilisé la même caméra, disposée de la même manière pour les images d'entraînement que pour la surveillance réelle.

#### Entraînement

#### Bilan

#### Forces

#### Faiblesses

#### Axes d'amélioration
  
###### Quick win
L’évaluation de l’état de santé repose sur la détection de la présence de chaque poisson.
Cette observation peut être affinée en considérant :  
- un poisson qui flotte à la surface,  
- ou, au contraire, qui reste immobile au fond de l'aquarium.
  
###### Heavy lift



