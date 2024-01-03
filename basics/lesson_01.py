import google.generativeai as genai
from decouple import config


GOOGLE_API_KEY = config("GOOGLE_API_KEY")

genai.configure(
    api_key=GOOGLE_API_KEY,
    client_options={"api_endpoint": "generativelanguage.googleapis.com"}
)

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)