import asyncio
import os
import glob
import logging
import re
from datetime import datetime
from typing import List, Dict, Any

# Imports des librairies tierces
import fitz  # PyMuPDF
from bs4 import BeautifulSoup  # Pour le parsing HTML rapide
from sentence_transformers import SentenceTransformer
import chromadb

# --- CONFIGURATION DU LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION GLOBALE ---
# Ajustez selon votre RAM et VRAM (GPU)
MAX_CONCURRENT_FILES = 8      # Nombre de fichiers traités en parallèle (Extraction/Chunking)
BATCH_SIZE = 64               # Taille du batch pour l'embedding et l'indexation
DATE_STR = datetime.now().strftime("%d-%m-%y")

# --- DEFINITION DES CLASSES (Tirées du Notebook) ---

class Document:
    def __init__(self, text: str, doc_index: int, file_name: str, doc_type: str):
        self.text = text
        self.doc_index = doc_index
        self.date_index = DATE_STR
        self.file_name = file_name
        self.doc_type = doc_type
        # Génération de l'ID unique
        clean_text = re.sub(r'[^a-zA-Z0-9]', '', self.text)
        suffix = clean_text[:10]
        self.id = f"inalco{self.doc_index}{self.date_index}{suffix}"

class Chunk:
    def __init__(self, parent_doc_id: str, chunk_index: int, text: str, file_name: str, doc_type: str):
        self.parent_doc_id = parent_doc_id
        self.chunk_index = chunk_index
        self.text = text
        self.chunk_file = file_name
        self.doc_type = doc_type
        self.chunk_id = f"{parent_doc_id}_chk{chunk_index}"

# --- FONCTIONS SYNCHRONES (WORKERS) ---
# Ces fonctions sont bloquantes. Nous les exécuterons dans un ThreadPool via asyncio.

def sync_read_file(file_path: str) -> str:
    """
    Lit un fichier PDF ou HTML et retourne le texte brut.
    Utilise PyMuPDF pour les PDF et BeautifulSoup pour les HTML (plus rapide que unstructured).
    """
    try:
        text = ""
        if file_path.lower().endswith(".pdf"):
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
        elif file_path.lower().endswith(".html") or file_path.lower().endswith(".htm"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f, 'html.parser')
                text = soup.get_text(separator=" ", strip=True)
        elif file_path.lower().endswith(".txt"):
             with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        return text
    except Exception as e:
        logging.error(f"Erreur lors de la lecture de {file_path}: {e}")
        return ""

def sync_chunk_text(text: str, max_size=500) -> List[str]:
    """
    Logique de découpage du texte (reprise simplifiée du notebook).
    """
    list_chunks = []
    # Nettoyage basique
    text = text.replace("[SEP]", " ") 
    
    current_chunk = ""
    # On découpe par phrase pour éviter de couper les mots
    sentences = text.split('. ') 
    
    for sentence in sentences:
        sentence = sentence.strip() + ". "
        if len(current_chunk) + len(sentence) <= max_size:
            current_chunk += sentence
        else:
            if current_chunk:
                list_chunks.append(current_chunk.strip())
            current_chunk = sentence if len(sentence) <= max_size else sentence[:max_size] # Cas limite phrase très longue
            
    if current_chunk:
        list_chunks.append(current_chunk.strip())
        
    return list_chunks

def sync_create_chunks_objects(doc_obj: Document) -> List[Chunk]:
    """Coordonne le chunking et la création d'objets Chunk"""
    raw_chunks = sync_chunk_text(doc_obj.text)
    return [
        Chunk(doc_obj.id, i+1, txt, doc_obj.file_name, doc_obj.doc_type)
        for i, txt in enumerate(raw_chunks)
    ]

def sync_embed_and_index(batch_chunks: List[Chunk], model, collection):
    """Calcule les embeddings et insère dans ChromaDB"""
    texts = [c.text for c in batch_chunks]
    ids = [c.chunk_id for c in batch_chunks]
    
    # Métadonnées
    metadatas = [{
        "file_name": c.chunk_file,
        "doc_type": c.doc_type,
        "parent_id": c.parent_doc_id
    } for c in batch_chunks]

    # Vectorisation (peut utiliser le GPU si dispo)
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    # Ajout DB
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

# --- PIPELINE ASYNCHRONE ---

async def process_document_step(file_path: str, file_index: int, sem: asyncio.Semaphore) -> List[Chunk]:
    """
    Traite UN fichier : Lecture -> Objet Document -> Liste de Chunks.
    Utilise un sémaphore pour limiter le nombre de fichiers ouverts simultanément.
    """
    loop = asyncio.get_running_loop()
    
    async with sem:
        # 1. Extraction (IO/CPU intensive -> ThreadPool)
        raw_text = await loop.run_in_executor(None, sync_read_file, file_path)
        
        if not raw_text or len(raw_text.strip()) < 10:
            return []

        # Détermination du type
        fname = os.path.basename(file_path)
        ftype = "pdf" if fname.endswith(".pdf") else "html"
        
        # Création objet Document
        doc = Document(raw_text, file_index, fname, ftype)

        # 2. Chunking (CPU intensive -> ThreadPool)
        chunks = await loop.run_in_executor(None, sync_create_chunks_objects, doc)
        
        return chunks

async def pipeline_main(documents_folder: str):
    """
    Orchestrateur principal.
    """
    # 1. Initialisation des outils (Load Model & DB)
    logging.info("Chargement du modèle et de la base de données...")
    
    # Détection GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Utilisation du device : {device}")
    
    # On charge le modèle (opération synchrone unique au début)
    model = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
    
    # Client Chroma
    client = chromadb.PersistentClient(path="./chroma_db_async")
    collection = client.get_or_create_collection("inalco_final_db")

    # 2. Récupération des fichiers
    # On cherche récursivement ou juste dans le dossier
    file_patterns = [f"{documents_folder}/*.pdf", f"{documents_folder}/*.html", f"{documents_folder}/*.txt"]
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern))
    
    all_files = sorted(list(set(all_files))) # Déduplication
    logging.info(f"{len(all_files)} documents trouvés dans {documents_folder}")

    if not all_files:
        logging.warning("Aucun fichier trouvé. Vérifiez le chemin.")
        return

    # 3. Phase Parallèle : Extraction & Chunking
    logging.info("--- Démarrage Phase 1 : Extraction & Chunking (Async) ---")
    
    sem = asyncio.Semaphore(MAX_CONCURRENT_FILES)
    tasks = [process_document_step(f, i, sem) for i, f in enumerate(all_files)]
    
    # asyncio.gather lance toutes les tâches et attend les résultats
    # results sera une liste de listes de Chunks : [[Chunk, Chunk], [], [Chunk]...]
    results = await asyncio.gather(*tasks)
    
    # Aplatissement de la liste
    total_chunks = [chunk for sublist in results for chunk in sublist]
    logging.info(f"Extraction terminée. {len(total_chunks)} chunks générés au total.")

    if not total_chunks:
        return

    # 4. Phase Batch : Embedding & Indexation
    logging.info("--- Démarrage Phase 2 : Embedding & Indexation (Batch) ---")
    
    loop = asyncio.get_running_loop()
    total_batches = (len(total_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(total_chunks), BATCH_SIZE):
        batch = total_chunks[i : i + BATCH_SIZE]
        
        # On exécute l'embedding et le stockage dans le threadpool pour ne pas bloquer la loop
        # (Bien que séquentiel ici, cela permet de garder l'appli réactive si on avait une API web)
        await loop.run_in_executor(None, sync_embed_and_index, batch, model, collection)
        
        current_batch = (i // BATCH_SIZE) + 1
        print(f"Indexation batch {current_batch}/{total_batches}...", end='\r')

    print("") # Saut de ligne final
    logging.info("Traitement terminé avec succès !")

# --- POINT D'ENTRÉE ---

if __name__ == "__main__":
    # Simulation du dossier si exécuté en local sans Kaggle
    input_folder = "/kaggle/input/inalcofiles" # Remplacez par votre variable documents_folder réelle
    
    # Si vous êtes dans le notebook, vous passez la variable documents_folder définie plus haut
    # Exemple : input_folder = documents_folder 
    
    if os.path.exists(input_folder):
        asyncio.run(pipeline_main(input_folder))
    else:
        logging.error(f"Le dossier {input_folder} n'existe pas.")