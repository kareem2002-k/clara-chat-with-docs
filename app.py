"""
Main entry point for the application
"""

from src.app import create_app

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
