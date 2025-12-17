# CLaRa vs Normal RAG Demo

A simple, educational demo that compares two approaches to document-based question answering:

- **CLaRa** ("latent doc compression"): Documents are compressed into latent space without prompt stuffing
- **Normal RAG**: Retrieve top-K document chunks → stuff them into the prompt → LLM generates answer

## Repository Structure

```
clara-vs-rag-demo/
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── models.py          # Model loading and management
│   ├── rag.py             # Normal RAG pipeline implementation
│   ├── clara.py           # CLaRa pipeline implementation
│   ├── utils.py           # Utility functions (parsing, formatting)
│   └── app.py             # Gradio web application
├── tests/                  # Test files
│   ├── __init__.py
│   └── test_frontend.py   # Frontend/utility function tests
├── examples/               # Example documents
│   ├── README.md
│   └── sample_docs.txt    # Sample documents for testing
├── app.py                 # Main entry point
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── LICENSE               # MIT License
└── .gitignore           # Git ignore rules
```

### Code Organization

- **`src/models.py`**: Handles loading of all models (CLaRa, RAG LLM, embeddings)
- **`src/rag.py`**: Implements the Normal RAG pipeline (embed → retrieve → prompt → generate)
- **`src/clara.py`**: Implements the CLaRa pipeline (latent compression)
- **`src/utils.py`**: Helper functions for document parsing and output formatting
- **`src/app.py`**: Gradio UI setup and main application logic
- **`app.py`**: Simple entry point that launches the application

## What This Demo Shows

This demo helps you understand the key difference between these two approaches:

- **Normal RAG** = Prompt Stuffing: We retrieve the most relevant documents using embeddings, then paste them directly into the LLM's prompt. This is simple but can hit token limits and may not scale well.

- **CLaRa** = Latent Compression: Documents are passed directly to CLaRa, which compresses them into a latent representation. No prompt stuffing occurs—the model handles documents internally through its compression mechanism.

## Quickstart

### Prerequisites

- Python 3.8+
- CUDA-capable GPU recommended (but CPU will work, just slower)
- ~15-20GB disk space for model downloads

### Installation

```bash
pip install -r requirements.txt
python app.py
```

The app will:
1. Download and load models (first run may take 10-15 minutes)
2. Start a Gradio web interface at `http://localhost:7860`

### Usage

1. **Paste Documents**: Enter your documents in the textarea, separated by `---`
   ```
   Document 1 text here...
   ---
   Document 2 text here...
   ---
   Document 3 text here...
   ```

2. **Enter Question**: Type your question in the question box

3. **Choose Mode**: 
   - **Normal RAG**: Uses embedding-based retrieval + prompt stuffing
   - **CLaRa**: Uses latent document compression

4. **Adjust Top-K** (RAG mode only): Select how many documents to retrieve (1-5)

5. **Click Run**: See the answer, evidence, and timing information

## Example

**Documents:**
```
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.
---
The Statue of Liberty is a neoclassical sculpture on Liberty Island in New York Harbor. It was a gift from France to the United States and was dedicated in 1886.
---
The Great Wall of China is a series of fortifications made of stone, brick, and other materials, generally built along an east-to-west line across the historical northern borders of China.
```

**Question:** "Which monument was a gift from France?"

**Normal RAG Output:**
- Retrieves top-K documents based on similarity
- Shows which docs were selected and their similarity scores
- Shows prompt length (characters)
- Shows retrieval + generation timing

**CLaRa Output:**
- Shows number of documents passed to the model
- Shows generation timing
- No prompt stuffing—docs compressed latently

## Models Used

- **CLaRa**: `apple/CLaRa-7B-Instruct` with `compression-16` checkpoint
- **RAG LLM**: `Qwen/Qwen2.5-3B-Instruct` (or fallback to `Phi-3-mini-4k-instruct`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`

## Technical Details

### Normal RAG Pipeline

1. Embed all documents and the question using sentence transformers
2. Compute cosine similarity (dot product of normalized embeddings)
3. Select top-K documents
4. Build prompt: System instruction + context docs + question
5. Generate answer using the RAG LLM

### CLaRa Pipeline

1. Pass documents directly to CLaRa model
2. CLaRa compresses documents into latent space
3. Generate answer using compressed representation
4. No prompt stuffing occurs

## Running Tests

Run the test suite to verify everything works:

```bash
# Install pytest if not already installed
pip install pytest

# Run tests
pytest tests/ -v
```

The tests cover:
- Document parsing functionality
- Output formatting for both RAG and CLaRa modes
- Edge cases (empty docs, single doc, etc.)

## Running on Google Colab

If you don't have a local GPU, you can run this on Google Colab:

1. Upload the files to Colab
2. Install dependencies: `!pip install -r requirements.txt`
3. Run: `!python app.py`
4. Use the public URL provided by Gradio

## Development

### For Developers

This repository is structured to be educational and easy to understand:

1. **Modular Design**: Each component (models, RAG, CLaRa) is in its own file
2. **Clear Separation**: Business logic separated from UI code
3. **Testable**: Utility functions can be easily tested
4. **Extensible**: Easy to add new features or modify existing ones

### Key Learning Points

- See how RAG works: embedding → retrieval → prompt construction → generation
- Understand CLaRa's approach: direct document compression without prompt stuffing
- Compare the two approaches side-by-side with timing and evidence

### Contributing

Feel free to:
- Add more example documents
- Improve the UI/UX
- Add more comprehensive tests
- Optimize model loading
- Add support for more models

## Notes

- First run will download models (~10-15GB total)
- GPU recommended for reasonable performance
- CLaRa model loading may require checking the official Apple CLaRa repository for exact API usage
- This is an educational demo, not production-grade code

## License

This demo uses publicly available models. Please check individual model licenses:
- CLaRa: Check Apple's license
- Qwen: Apache 2.0
- Sentence Transformers: Apache 2.0
