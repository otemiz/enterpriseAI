import os
import argparse
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

def load_data(
    documents_directory: str = "documents",
    collection_name: str = "hackathon_collection",
    persist_directory: str = "chroma_storage",
) -> None:
    # Read all files in the data directory
    documents = []
    metadatas = []
    text_file_dir = os.path.join(documents_directory, "text")
    text_files = os.listdir(text_file_dir)
    image_file_dir = os.path.join(documents_directory, "image")
    image_files = os.listdir(image_file_dir)
    
    for filename in text_files:
        # Text files
        print(filename)
        with open(os.path.join(text_file_dir, filename), "r") as file:
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
    embedding_function = OpenCLIPEmbeddingFunction()
    image_loader = ImageLoader()
    single_collection = client.get_or_create_collection(name="single_collection", embedding_function=embedding_function, data_loader=image_loader)

    # Create ids from the current count
    count = single_collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    ## Load text documents in batches of 100
    batch_size = 1
    for i in tqdm(
        range(0, len(documents), batch_size), desc="Adding documents", unit_scale=batch_size
    ):
        single_collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],  # type: ignore
        )

    new_count = single_collection.count()
    print(f"Added {new_count - count} documents")

    # Get the uris to the images
    image_uris = sorted([os.path.join(image_file_dir, image_name) for image_name in os.listdir(image_file_dir)])
    ids = [str(new_count + i + 1) for i in range(len(image_uris))]
    single_collection.add(ids=ids, uris=image_uris)
    print(f"Added {single_collection.count()-new_count} images")

if __name__ == "__main__":
    load_data(
        documents_directory="documents",
        collection_name="hackathon_collection",
        persist_directory="chroma_storage"
    )
