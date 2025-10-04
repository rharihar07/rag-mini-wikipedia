import re
from typing import List, Iterable

# NLTK imports (corpora/models are downloaded by download_nltk_data)
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

class Setup:
    """Setup script for preparing the environment."""
    def __init__(self):
        self.nltk_packages = (
            "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "universal_tagset"
        )

    def download_nltk_data(self):
        import nltk
        for pkg in self.nltk_packages:
            nltk.download(pkg, quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.english_stopwords = set(stopwords.words("english"))

    def clean_text(self, text: str) -> str:
        """Clean the input string by converting it to lowercase, removing 's and apostrophe."""

        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r"'s\b", '', text)
        text = text.replace("'","")
        text = text.strip()
        return text

    def tokenize(self, cleaned_text: str) -> List[str]:
        """
        Tokenize the cleaned text, then split on non-alphanumeric and drop
        single-character fragments.
        """
        token = word_tokenize(cleaned_text)
        tokens: List[str] = []
        for t in token:
            parts = re.split(r"[^a-zA-Z0-9]", t)
            tokens.extend(parts)
        tokens = [tok for tok in tokens if len(tok) > 1]
        return tokens

    def lemmatize(self, tokens: Iterable[str], stopwords: Iterable[str] = ()) -> List[str]:
        """
        Lemmatize tokens using POS tags; filter out provided stopwords and
        one-character tokens.
        """
        # tag each token (as a single-word "sentence")
        tagged = [pos_tag([w])[0] for w in tokens if w]

        # map Penn tags -> WordNet tags
        wn_pairs = []
        for word, tag in tagged:
            if tag.startswith("J"):
                pos = wordnet.ADJ
            elif tag.startswith("V"):
                pos = "v"
            elif tag.startswith("R"):
                pos = "r"
            else:
                pos = "n"
            wn_pairs.append((word, pos))

        # lemmatize and filter
        lemmatizer = self.lemmatizer or WordNetLemmatizer()
        stop_set = set(stopwords) if stopwords else set()
        lemm = [lemmatizer.lemmatize(w, p) for (w, p) in wn_pairs]
        lemm = [w for w in lemm if (w not in stop_set) and (len(w) > 1)]
        return lemm

    def preprocess_text(self, text: str, stopwords: Iterable[str] = None) -> List[str]:
        """
        Full pipeline: clean -> tokenize -> lemmatize. Returns list of tokens.
        """
        if stopwords is None:
            stopwords = self.english_stopwords or set()
        cleaned = self.clean_text(text)
        toks = self.tokenize(cleaned)
        return self.lemmatize(toks, stopwords)
    
    def build_embedder(self, model_name: str = "all-MiniLM-L6-v2"):
        """Return a SentenceTransformer embedder (dim=384 for MiniLM)."""
        return SentenceTransformer(model_name)
    
    def init_milvus(self, db_path: str = "rag_wikipedia_mini.db", collection_name: str = "rag_mini", dim: int = 384, max_length: int = 8000):
        """
        Create/load a Milvus Lite DB exactly like your notebook cells:
        - create_schema() -> add_field(...)
        - create_collection(...)
        - prepare_index_params() / create_index(...)
        - load_collection(...)
        """
        from pymilvus import MilvusClient, DataType

    # 1) client
        client = MilvusClient(db_path)

    # 2) schema (id, passage, embedding)
        schema = MilvusClient.create_schema()
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=False,
        )
        schema.add_field(
            field_name="passage",
            datatype=DataType.VARCHAR,
            max_length=max_length,
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
        )

    # 3) create collection if missing
        try:
            client.describe_collection(collection_name)
        except Exception:
            client.create_collection(collection_name=collection_name, schema=schema)

    # 5) load
        client.load_collection(collection_name)
        return client
    
    def create_milvus_index(self, client, collection_name: str, field_name: str = "embedding", index_type: str = "IVF_FLAT", metric_type: str = "L2", nlist: int = 128):
        """Create a vector index on a field for a given collection."""
        from pymilvus import MilvusClient
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=field_name,
            index_type=index_type,
            metric_type=metric_type,
            params={"nlist": nlist},
        )
        client.create_index(collection_name=collection_name, index_params=index_params)

    def build_llm(self, model_name: str = "google/flan-t5-base"):
        """
        Return (tokenizer, model) for a FLAN-T5 checkpoint.
        """
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        return tok, mdl    
    
    def hyde(self, tokenizer, model, question: str, max_new_tokens: int = 120):
        import torch
        prompt = (
            "Write a concise, factual paragraph that would answer the question in an encyclopedia style.\n\n"
            f"Question: {question}\n\nParagraph:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)  # CPU tensors
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    
    def build_reranker(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name, device=device)

    def rerank(self, reranker, query: str, hits, top_k: int = 5):
        import numpy as np
        pairs, texts = [], []
        for h in hits:
            try:
                txt = h.entity.get("passage", "")
            except AttributeError:
                txt = h.get("entity", {}).get("passage", "")
            pairs.append((query, txt))
            texts.append(txt)
        scores = reranker.predict(pairs)  # runs on CPU
        order = np.argsort(-scores)[:top_k]
        return [texts[i] for i in order], [float(scores[i]) for i in order]

