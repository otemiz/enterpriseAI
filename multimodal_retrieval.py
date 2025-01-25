import os
from datasets import load_dataset
from matplotlib import pyplot as plt
import base64

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

from interface_gpt import get_chatGPT_response, get_chatGPT_response_image

# dataset = load_dataset(path="detection-datasets/coco", split="train", streaming=True)


IMAGE_FOLDER = "images"
N_IMAGES = 20

# # Write the images to a folder
# dataset_iter = iter(dataset)
# os.makedirs(IMAGE_FOLDER, exist_ok=True)
# for i in range(N_IMAGES):
#     image = next(dataset_iter)['image']
#     axes[i].imshow(image)
#     axes[i].axis("off")

#     image.save(f"images/{i}.jpg")

# plt.tight_layout()
# plt.show()


client = chromadb.Client()

embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()

collection = client.create_collection(
    name='multimodal_collection', 
    embedding_function=embedding_function, 
    data_loader=image_loader)

# Get the uris to the images
image_uris = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER)])
ids = [str(i) for i in range(len(image_uris))]

collection.add(ids=ids, uris=image_uris)

n_results = 3

# For plotting
plot_cols = n_results
plot_rows = 1
fig, axes = plt.subplots(plot_rows, plot_cols)
axes = axes.flatten()


query = input("Query: ")
if len(query) == 0:
    print("Please enter a question. Ctrl+C to Quit.\n")

print(f"\nThinking using gpt...\n")

# Querying
retrieved = collection.query(query_texts=[query], include=["uris"], n_results=n_results)

for i, img_path in enumerate(retrieved['uris'][0]):
    img = plt.imread(img_path)
    axes[i].imshow(img)
    axes[i].axis("off")

plt.tight_layout()
plt.show()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

model_name = "gpt-4o-mini"

# Get the response from GPT
response = get_chatGPT_response_image(encode_image(retrieved['uris'][0][0]), model_name, query)
print(response)