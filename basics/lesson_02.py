from decouple import config
import os

from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.multi_modal_llms.generic_utils import (
    load_image_urls,
)

os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY")


image_urls = [
    "https://www.dropbox.com/scl/fi/v0l8a35fx62qnuoefdu6f/2019_Porsche_911_Carrera.jpg?rlkey=cjg374asg3u1u9deujhk2vsdo&raw=1",
]

image_documents = load_image_urls(image_urls=image_urls)

gemini_prov_vision = GeminiMultiModal(
    model_name="models/gemini-pro-vision"
)

response = gemini_prov_vision.complete(
    prompt="What is the name of the vehicle in the image?",
    image_documents=image_documents
)

print(response)