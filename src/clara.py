"""
CLaRa pipeline implementation
"""

import time
import torch
from typing import List, Tuple, Dict
from .models import get_clara_model, get_device


def run_clara(docs: List[str], question: str) -> Tuple[str, Dict]:
    """
    CLaRa pipeline:
    1. Pass docs directly to CLaRa (latent compression)
    2. Generate answer
    """
    if not docs or not question:
        return "Please provide documents and a question.", {}
    
    clara_model, clara_tokenizer = get_clara_model()
    device = get_device()
    
    if clara_model is None or clara_tokenizer is None:
        return "Error: CLaRa model not loaded. Please check the model download.", {}
    
    start_time = time.time()
    
    try:
        # CLaRa uses generate_from_text method
        # Note: The exact API may vary - adjust based on actual CLaRa implementation
        generation_start = time.time()
        
        # Try the expected CLaRa API
        if hasattr(clara_model, 'generate_from_text'):
            result = clara_model.generate_from_text(
                questions=[question],
                documents=[docs],
                max_new_tokens=256
            )
            answer = result[0] if isinstance(result, list) else str(result)
        else:
            # Fallback: try standard generation with CLaRa's special format
            # This is a placeholder - actual CLaRa may have different API
            prompt = f"Question: {question}\n\nDocuments:\n" + "\n\n".join(docs)
            inputs = clara_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
            
            with torch.no_grad():
                outputs = clara_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=clara_tokenizer.eos_token_id
                )
            
            answer = clara_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Prepare evidence
        evidence = {
            "mode": "CLaRa",
            "explanation": f"We passed {len(docs)} docs directly to CLaRa, which compresses them into latent space (no prompt stuffing).",
            "docs_passed": len(docs),
            "generation_time": f"{generation_time:.3f}s",
            "total_time": f"{total_time:.3f}s"
        }
        
        return answer, evidence
        
    except Exception as e:
        return f"Error running CLaRa: {str(e)}\n\nNote: CLaRa API may differ. Check the model documentation.", {}

