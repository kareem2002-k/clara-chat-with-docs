"""
Utility functions
"""

from typing import List, Dict


def parse_docs(raw: str) -> List[str]:
    """Parse documents separated by '---'"""
    docs = [doc.strip() for doc in raw.split("---")]
    docs = [doc for doc in docs if doc]  # Remove empty docs
    return docs


def format_output(answer: str, evidence: Dict) -> str:
    """Format the output for display"""
    if not evidence:
        return answer
    
    output = f"**Answer:**\n{answer}\n\n"
    output += "---\n\n"
    output += f"**Mode:** {evidence.get('mode', 'Unknown')}\n\n"
    output += f"**Explanation:**\n{evidence.get('explanation', '')}\n\n"
    
    if evidence.get('mode') == 'Normal RAG':
        output += "**Retrieved Documents:**\n"
        for i, (doc, score) in enumerate(zip(evidence.get('selected_docs', []), evidence.get('scores', []))):
            output += f"\n**Doc {i+1}** (similarity: {score:.4f}):\n{doc[:200]}...\n"
        output += f"\n**Prompt Length:** {evidence.get('prompt_length', 0):,} characters\n"
        output += f"**Retrieval Time:** {evidence.get('retrieval_time', 'N/A')}\n"
        output += f"**Generation Time:** {evidence.get('generation_time', 'N/A')}\n"
    else:
        output += f"**Documents Passed:** {evidence.get('docs_passed', 0)}\n"
        output += f"**Generation Time:** {evidence.get('generation_time', 'N/A')}\n"
    
    output += f"\n**Total Time:** {evidence.get('total_time', 'N/A')}\n"
    
    return output

