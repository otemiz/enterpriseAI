import gradio as gr
from rag import RAG
from dspy.utils import download


# "Example question: How many are working duplicate tasks? Who are they? What are these ta How can they optimize this"

# download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl")

# Gradio interface
def main():
    rag = RAG()
    interface = gr.Interface(
        fn=rag,
        inputs=gr.Textbox(label="Enter your question", placeholder="Type your question here..."),
        outputs=[
            gr.Textbox(label="Reasoning"),
            gr.Textbox(label="Response")
        ],
        title="RAG Question Answering",
        description="Ask a question, and the RAG module will provide an answer based on the corpus."
    )
    interface.launch()

if __name__ == "__main__":
    main()
