import os
import argparse
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def load_data(
    documents_directory: str = "documents",
    collection_name: str = "hackathon_collection",
    persist_directory: str = "chroma_storage",
) -> None:
    # Read all files in the data directory
    documents = []
    metadatas = []
    files = os.listdir(documents_directory)
    for filename in files:
        with open(f"{documents_directory}/{filename}", "r") as file:
            for line_number, line in enumerate(
                tqdm((file.readlines()), desc=f"Reading {filename}"), 1
            ):
                # Strip whitespace and append the line to the documents list
                line = line.strip()
                # Skip empty lines
                if len(line) == 0:
                    continue
                documents.append(line)
                metadatas.append({"filename": filename, "line_number": line_number})

    # Instantiate a persistent chroma client in the persist_directory.
    # Learn more at docs.trychroma.com
    client = chromadb.PersistentClient(path=persist_directory)

    # If the collection already exists, we just return it. This allows us to add more
    # data to an existing collection.
    openai_embedding = OpenAIEmbeddingFunction(
    api_key=os.environ.get('OPENAI_API_KEY'),
    model_name="text-embedding-3-small")
    #model_name="text-embedding-ada-002")
    collection = client.get_or_create_collection(name=collection_name, embedding_function=openai_embedding)

    # Create ids from the current count
    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    # Load the documents in batches of 100
    batch_size = 100
    for i in tqdm(
        range(0, len(documents), batch_size), desc="Adding documents", unit_scale=batch_size
    ):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],  # type: ignore
        )

    new_count = collection.count()
    print(f"Added {new_count - count} documents")


if __name__ == "__main__":
    load_data(
        documents_directory="documents",
        collection_name="hackathon_collection",
        persist_directory="chroma_storage"
    )
