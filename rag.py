import os
import dspy
import ujson
import openai
from dspy.retrieve.chromadb_rm import ChromadbRM


class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')
        self.lm = dspy.LM('openai/gpt-4o-mini')
        self.embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
        self.max_characters = 6000
        self.topk_docs_to_retrieve = 5
        self.file_path = "dataset/jira_dataset/deneme.jsonl"
        self.initEmbedding()
        dspy.configure(lm=self.lm)

    def updateEmbedding(self, corpus):
        self.search_object = dspy.retrievers.Embeddings(embedder=self.embedder, corpus=corpus, k=self.topk_docs_to_retrieve)

    def initEmbedding(self):
        self.last_modified = os.path.getmtime(self.file_path)
        with open(self.file_path) as f:
            corpus = [ujson.loads(line)['text'][:self.max_characters] for line in f]
            print(f"Loaded {len(corpus)} initial documents. Will encode them below.")
            self.updateEmbedding(corpus)

    def detectFileChanges(self):
        current_modified = os.path.getmtime(self.file_path)
        if current_modified != self.last_modified:
            print("Database has changed!")
            self.last_modified = current_modified
            return True
        return False

    def runEmbedding(self):
        file_change = self.detectFileChanges()
        if (file_change):
            with open(self.file_path) as f:
                corpus = [ujson.loads(line)['text'][:self.max_characters] for line in f]
                print(f"Loaded {len(corpus)} updated documents. Will encode them below.")
                self.updateEmbedding(corpus)

    def forward(self, question):
        self.runEmbedding()
        context = self.search_object(question).passages
        result =  self.respond(context=context, question=question)
        return result["rationale"], result["response"]
    