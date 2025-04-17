# AskTheMenu

A simple LLM-powered restaurant recommender using Sentence-BERT, clustering, and FAISS.

## Features
- Generates fake restaurant data
- Embeds restaurant descriptions using Sentence-BERT
- Clusters restaurants by semantic similarity
- Builds a FAISS index for fast vector search
- Ready for integration with LangChain or GPT-based reasoning

## Installation

```bash
git clone <repo-url>
cd askthemenu
pip install -r requirements.txt
```

## Usage

```bash
python run_all.py
```

All data will be generated and stored in the `data/` folder.

## Folder Structure
```
askthemenu/
├── data/
│   ├── restaurants.csv
│   ├── embeddings.npy
│   ├── restaurants_clustered.csv
│   └── faiss_index.index
├── src/
│   ├── generate_data.py
│   ├── embed.py
│   ├── cluster.py
│   └── build_index.py
├── requirements.txt
├── README.md
└── run_all.py
```
