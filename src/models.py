"""
Model loading and management
"""

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import SentenceTransformer


# Global model storage
clara_model = None
clara_tokenizer = None
rag_model = None
rag_tokenizer = None
embedder = None
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def load_models():
    """Load all models once at startup"""
    global clara_model, clara_tokenizer, rag_model, rag_tokenizer, embedder
    
    print("Loading models...")
    
    # Load CLaRa model
    print("Downloading CLaRa model...")
    try:
        snapshot_path = snapshot_download(
            repo_id="apple/CLaRa-7B-Instruct",
            allow_patterns=["compression-16/*"],
            local_dir="./clara_cache"
        )
        clara_dir = "./clara_cache/compression-16"
        print(f"Loading CLaRa from {clara_dir}...")
        clara_model = AutoModel.from_pretrained(
            clara_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None
        )
        clara_tokenizer = AutoTokenizer.from_pretrained(clara_dir, trust_remote_code=True)
        if clara_tokenizer.pad_token is None:
            clara_tokenizer.pad_token = clara_tokenizer.eos_token
        print("✓ CLaRa loaded")
    except Exception as e:
        print(f"Warning: Could not load CLaRa model: {e}")
        print("You may need to check the model repository or use a different approach.")
    
    # Load RAG LLM
    print("Loading RAG LLM (Qwen2.5-3B)...")
    try:
        rag_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        rag_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None
        )
        print("✓ RAG LLM loaded")
    except Exception as e:
        print(f"Warning: Could not load Qwen model: {e}")
        print("Falling back to a smaller model...")
        try:
            rag_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
            rag_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None
            )
            print("✓ Fallback RAG LLM loaded")
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")
    
    # Load embedder
    print("Loading sentence transformer embedder...")
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    print("✓ Embedder loaded")
    
    print("All models loaded!")


def get_clara_model():
    """Get CLaRa model and tokenizer"""
    return clara_model, clara_tokenizer


def get_rag_model():
    """Get RAG model and tokenizer"""
    return rag_model, rag_tokenizer


def get_embedder():
    """Get embedding model"""
    return embedder


def get_device():
    """Get current device"""
    return device

