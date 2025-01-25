import argparse
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


# Instantiate a persistent chroma client in the persist_directory.
# This will automatically load any previously saved collections.
# Learn more at docs.trychroma.com
client = chromadb.PersistentClient(path="chroma_storage")
embedding_function = OpenCLIPEmbeddingFunction()

single_collection = client.get_collection(name="single_collection", embedding_function=embedding_function)
#text_collection = client.get_collection(name="text_collection", embedding_function=embedding_function)

retrieved = single_collection.query(query_texts=["sunny weather"],include=['uris', 'data', 'documents'], n_results=3)

print(retrieved)
