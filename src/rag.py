"""
Normal RAG pipeline implementation
"""

import time
import torch
import numpy as np
from typing import List, Tuple, Dict
from .models import get_rag_model, get_embedder, get_device


def run_normal_rag(docs: List[str], question: str, top_k: int) -> Tuple[str, Dict]:
    """
    Normal RAG pipeline:
    1. Embed docs and question
    2. Find top-K similar docs
    3. Stuff into prompt
    4. Generate answer
    """
    if not docs or not question:
        return "Please provide documents and a question.", {}
    
    rag_model, rag_tokenizer = get_rag_model()
    embedder = get_embedder()
    device = get_device()
    
    if rag_model is None or rag_tokenizer is None or embedder is None:
        return "Error: Models not loaded. Please restart the app.", {}
    
    start_time = time.time()
    
    # Step 1: Embed documents and question
    retrieval_start = time.time()
    doc_embeddings = embedder.encode(docs, normalize_embeddings=True, convert_to_numpy=True)
    question_embedding = embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
    
    # Step 2: Compute similarities (dot product for normalized embeddings)
    similarities = np.dot(doc_embeddings, question_embedding)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    retrieval_time = time.time() - retrieval_start
    
    # Step 3: Build prompt with top-K docs
    selected_docs = [docs[i] for i in top_indices]
    selected_scores = [float(similarities[i]) for i in top_indices]
    
    context_text = "\n\n".join([f"Doc{i+1}: {doc}" for i, doc in enumerate(selected_docs)])
    
    prompt = f"""Answer ONLY using the provided context. If the answer is not in the context, say "I don't know."

[CONTEXT]
{context_text}

[QUESTION]
{question}

[ANSWER]
"""
    
    # Step 4: Generate answer
    generation_start = time.time()
    inputs = rag_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    with torch.no_grad():
        outputs = rag_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=rag_tokenizer.eos_token_id
        )
    
    answer = rag_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    generation_time = time.time() - generation_start
    total_time = time.time() - start_time
    
    # Prepare evidence
    evidence = {
        "mode": "Normal RAG",
        "explanation": f"We retrieved Top-{top_k} docs and pasted them into the prompt.",
        "selected_docs": selected_docs,
        "scores": selected_scores,
        "indices": top_indices.tolist(),
        "prompt_length": len(prompt),
        "retrieval_time": f"{retrieval_time:.3f}s",
        "generation_time": f"{generation_time:.3f}s",
        "total_time": f"{total_time:.3f}s"
    }
    
    return answer, evidence

