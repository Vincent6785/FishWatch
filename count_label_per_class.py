import os
from collections import Counter

# --- Paramètres ---
LABELS_DIR = r"./dataset/labels/"

# --- Initialisation ---
counts = Counter({i: 0 for i in range(11)})  # IDs de 0 à 10 inclus

# --- Parcours des fichiers ---
for fname in os.listdir(LABELS_DIR):
    if fname.endswith(".txt"):  # seulement les fichiers YOLO
        with open(os.path.join(LABELS_DIR, fname), "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:  # éviter les lignes vides
                    try:
                        class_id = int(parts[0])
                        if 0 <= class_id <= 10:
                            counts[class_id] += 1
                    except ValueError:
                        pass  # ligne mal formée

# --- Résultat ---
print("Comptage des classes (0-10) :")
for cid in range(11):
    print(f"Classe {cid}: {counts[cid]} instances")
