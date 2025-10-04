# README

This repository contains a compact Retrieval‑Augmented Generation (RAG) sandbox built around a mini‑Wikipedia dataset. It lets you compare a naïve RAG baseline against small enhancements and different top‑k retrieval settings. The core assets are Jupyter notebooks in notebooks/: naive\_rag.ipynb (baseline), naive\_rag\_top3.ipynb and naive\_rag\_top5.ipynb (k‑variants), enhanced\_rag.ipynb (improvements), data\_exploratory.ipynb (EDA), and a starter template. Vector stores are created locally via FAISS or Milvus Lite, producing .db files alongside the notebooks. Source utilities live in src/ (e.g., evaluation.py for basic scoring/reporting); input text sits in data/passages.csv; experiment outputs may go to result/. Dependencies (e.g., sentence-transformers, transformers, torch, faiss-cpu, pymilvus, langchain) are captured in requirements.txt. Use the provided notebooks to ingest passages, embed with all-MiniLM-L6-v2, index into a vector store, retrieve top‑k contexts, and generate answers for quick comparisons.

Create a Conda environment in VS Code (line‑by‑line)

1. Create an environment (Python 3.10 recommended):

conda create \-n rag\_env python=3.10 \-y

2. Activate it:

conda activate rag\_env

3. In each notebook, select Kernel \-\> Python (rag\_env).  
     
4. Install dependencies from requirements.txt

Ensure you are in the repo root (where requirements.txt lives).

pip install \-r requirements.txt

This installs: numpy, pandas, torch, transformers, sentence-transformers, faiss-cpu, pymilvus/milvus-lite, langchain (+ splitters & openai wrapper), ragas, evaluate, nltk, and other utilities required by the notebooks and scripts.

If you have a CUDA GPU and want GPU PyTorch, install the matching wheel from pytorch.org before running pip install \-r requirements.txt, then run the pip install \-r to fetch the rest.

