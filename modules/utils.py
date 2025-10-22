import re

def extract_datetime_key(filename: str) -> str:
    """
    Extrait une clé de tri basée sur la date et l'heure contenues dans le nom de fichier.

    Exemple :
        "20241021_153045_video.avi" → "20241021153045"

    Args:
        filename: Nom du fichier vidéo.

    Returns:
        Une chaîne représentant la date et l'heure concaténées si trouvées,
        sinon le nom de fichier original (pour un tri par défaut).
    """
    pattern = r"(\\d{8})_(\\d{6})"
    match = re.search(pattern, filename)

    if not match:
        return filename

    date_part, time_part = match.groups()
    return f"{date_part}{time_part}"
