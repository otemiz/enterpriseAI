#from atlassian import Jira
import dspy
import ujson
from dspy.utils import download
from openai import OpenAI

download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl")

max_characters = 6000  # for truncating >99th percentile of documents
topk_docs_to_retrieve = 5  # number of documents to retrieve per search query

with open("dataset/jira_dataset/deneme.jsonl") as f:
    #for line in f:
    #    print(line)
    corpus = [ujson.loads(line)['text'][:max_characters] for line in f]
    print(f"Loaded {len(corpus)} documents. Will encode them below.")

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=topk_docs_to_retrieve)

class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)
    
rag = RAG()
rag(question="How many people are working on A* Related tasks? Who are they? What are the tasks' status")
dspy.inspect_history()

cost = sum([x['cost'] for x in lm.history if x['cost'] is not None]) 
print(cost)

#lm = dspy.LM('openai/gpt-4o-mini')
#dspy.configure(lm=lm)

#qa = dspy.Predict('question: str -> response: str')
#response = qa(question="what are high memory and low memory on linux?")

#print(response.response)