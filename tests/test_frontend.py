"""
Simple frontend tests for the demo application
"""

import pytest
from src.utils import parse_docs, format_output


def test_parse_docs():
    """Test document parsing"""
    # Test basic parsing
    docs_text = "Doc 1\n---\nDoc 2\n---\nDoc 3"
    docs = parse_docs(docs_text)
    assert len(docs) == 3
    assert docs[0] == "Doc 1"
    assert docs[1] == "Doc 2"
    assert docs[2] == "Doc 3"
    
    # Test with empty docs
    docs_text = "Doc 1\n---\n\n---\nDoc 3"
    docs = parse_docs(docs_text)
    assert len(docs) == 2
    assert "Doc 1" in docs
    assert "Doc 3" in docs
    
    # Test single doc
    docs_text = "Single document"
    docs = parse_docs(docs_text)
    assert len(docs) == 1
    assert docs[0] == "Single document"


def test_format_output_rag():
    """Test output formatting for RAG mode"""
    answer = "The answer is Paris."
    evidence = {
        "mode": "Normal RAG",
        "explanation": "We retrieved Top-2 docs and pasted them into the prompt.",
        "selected_docs": ["Doc 1", "Doc 2"],
        "scores": [0.95, 0.87],
        "prompt_length": 500,
        "retrieval_time": "0.123s",
        "generation_time": "1.456s",
        "total_time": "1.579s"
    }
    
    output = format_output(answer, evidence)
    
    assert "Answer:" in output
    assert "Paris" in output
    assert "Normal RAG" in output
    assert "Top-2" in output
    assert "500" in output
    assert "0.123s" in output


def test_format_output_clara():
    """Test output formatting for CLaRa mode"""
    answer = "The answer is Paris."
    evidence = {
        "mode": "CLaRa",
        "explanation": "We passed 3 docs directly to CLaRa.",
        "docs_passed": 3,
        "generation_time": "2.345s",
        "total_time": "2.345s"
    }
    
    output = format_output(answer, evidence)
    
    assert "Answer:" in output
    assert "Paris" in output
    assert "CLaRa" in output
    assert "3" in output
    assert "2.345s" in output


def test_format_output_empty():
    """Test output formatting with empty evidence"""
    answer = "Test answer"
    evidence = {}
    
    output = format_output(answer, evidence)
    assert output == answer


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

