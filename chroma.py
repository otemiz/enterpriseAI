import os
import chromadb
from chromadb.utils import embedding_functions
from dspy.retrieve.chromadb_rm import ChromadbRM
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def add_collections(folder_path, collection):
    try:
        # Check if the folder exists
        if not os.path.isdir(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return
        
        # Loop through all files in the folder
        count = 1
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # Check if it's a file (not a folder)
            if os.path.isfile(file_path):
                try:
                    # Open the file and read its content
                    with open(file_path, 'r', encoding='utf-8') as file:
                        collection.add(
                            documents=[file.read()],
                            ids=[str(count)]
                        )
                        count += 1
                except Exception as e:
                    print(f"Error reading file '{file_name}': {e}")
            else:
                print(f"Skipping '{file_name}' as it is not a file.")
    except Exception as e:
        print(f"An error occurred: {e}")

chroma_client = client = chromadb.PersistentClient(path="./vector_dataset/persistent/")
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get('OPENAI_API_KEY'),
    model_name="text-embedding-ada-002"
)
default_ef = embedding_function
collection = chroma_client.get_or_create_collection(name="everything", embedding_function=default_ef)
add_collections("./vector_dataset/", collection)

rm = ChromadbRM(collection_name='everything', persist_directory="./vector_dataset/persistent/", embedding_function=default_ef, k=1)
print(rm('Covid19'))