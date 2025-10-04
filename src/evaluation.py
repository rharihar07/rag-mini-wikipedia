import numpy as np
from typing import Sequence


import evaluate

class Evaluation:
    """Simple evaluation metrics for QA outputs."""
    @staticmethod
    def score(df, id_col="id", pred_col="answer_generated", ref_col="answer", squad_v2=False):
        """
        Compute SQuAD-style Exact Match and F1 from a DataFrame with an 'ids' column.
        Returns: {'exact_match': float, 'f1': float} (percentage scores)
        """
        metric = evaluate.load("squad_v2" if squad_v2 else "squad")

        ids   = df[id_col].astype(str).tolist()
        preds = df[pred_col].fillna("").astype(str).tolist()
        refs  = df[ref_col].fillna("").astype(str).tolist()

        predictions = [{"id": i, "prediction_text": p} for i, p in zip(ids, preds)]
        references  = [{"id": i, "answers": {"text": [r], "answer_start": [0]}} for i, r in zip(ids, refs)]

        return metric.compute(predictions=predictions, references=references)
    


    def ragas_scores_df(
        df,
        question_col: str = "question",
        pred_col: str = "answer_generated",
        contexts_col: str = "context",        # can be list[str] or one big string
        ref_col: str = "answer",
        llm_model: str = "gpt-4o-mini",           
        emb_model: str = "text-embedding-3-small",
        ):
        """
        Compute RAGAS metrics on a DataFrame.
        Returns a dict: {'faithfulness': ..., 'answer_relevancy': ..., 'context_precision': ..., 'context_recall': ...}
        """

        import os,re
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        import openai

        os.environ["OPENAI_API_KEY"] = ""

    
        llm = ChatOpenAI(model=llm_model)
        openai_client = openai.OpenAI() 
        embeddings = OpenAIEmbeddings(model=emb_model)
        # llm_langchain = ChatOpenAI(model="gpt-4o", temperature=0)  # judge LLM
        # openai_client = openai.OpenAI()                            # OpenAI SDK client
        # embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")   
    # 3) Ensure contexts is a list[str]
        def to_list(x):
            if isinstance(x, list):
                return [str(t) for t in x if str(t).strip()]
            if isinstance(x, str):
            # If you concatenated with numbered blocks like "[1] ...\n\n[2] ...", split those.
                parts = [p.strip() for p in re.split(r"\n?\[\d+\]\s*", x) if p.strip()]
                return parts if parts else [x.strip()]
            return [str(x)]

        ds = Dataset.from_dict({
            "question":      df[question_col].astype(str).tolist(),
            "answer":        df[pred_col].fillna("").astype(str).tolist(),
            "contexts":      df[contexts_col].apply(to_list).tolist(),
            "ground_truth":  df[ref_col].fillna("").astype(str).tolist(),
        })

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        result = ragas_evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=embeddings)

    # result is a dict-like in current ragas
        return result