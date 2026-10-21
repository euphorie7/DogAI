from datasets import load_dataset
from pathlib import Path

# Colonne	                Contenu
# anchor	                Question sur une race
# positive	                Passage documentaire pertinent
# negative_1 à negative_5	Passages non pertinents pour la question


# Charger le dataset
dataset = load_dataset(
    "dorrito-dev/dog-breed-ntuple-retrieval"
)

# Extraire les textes positifs et supprimer les doublons
documents = set(dataset["train"]["positive"])

# Créer le dossier data
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

# Regrouper tous les textes dans un seul document
texte_complet = "\n\n---DOCUMENT---\n\n".join(sorted(documents))
print(texte_complet)

# Enregistrer dans un seul fichier
file_path = output_dir / "dogs.txt"
file_path.write_text(texte_complet, encoding="utf-8")