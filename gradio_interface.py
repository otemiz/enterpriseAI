from LLM import *
import gradio as gr

def query_rag(question):
    "Example question: How many are working duplicate tasks? Who are they? What are these ta How can they optimize this"
    try:
        response = rag(question)
        return response["reasoning"], response["response"], sum([x['cost'] for x in lm.history if x['cost'] is not None]) 
    except Exception as e:
        return f"An error occurred: {e}"

# Gradio interface
def main():
    interface = gr.Interface(
        fn=query_rag,
        inputs=gr.Textbox(label="Enter your question", placeholder="Type your question here..."),
        outputs=[
            gr.Textbox(label="Reasoning"),
            gr.Textbox(label="Response"),
            gr.Textbox(label="Cost")
        ],
        title="RAG Question Answering",
        description="Ask a question, and the RAG module will provide an answer based on the corpus."
    )
    interface.launch()

if __name__ == "__main__":
    main()