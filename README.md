# InalcoSearch

### Prérequis :

#### 1. La base de données vectorielle ChromaDB (basée sur [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base))
```python
!sudo apt-get install megatools -q
!megadl "https://mega.nz/file/KV8kDJJC#as8NYEPKGI-C2My9B4PUYBcKp70QU7taMxcIMMgJpvA"
```

#### 2. L'ensemble des documents (c'est un dossier contenant 4863 fichiers) :

```python
import kagglehub

path = kagglehub.dataset_download("gwendaltsang/inalcofiles")
```

**Remarque** : la recherche sémantique fonctionnera quand même si vous avec la base de donnée vectorielle sans avoir les documents, mais le streamlit a la fonctionnalité d'afficher dynamiquement les PDFs sur la page.

