from pydantic import BaseModel

from llama_index.multi_modal_llms import GeminiMultiModal
from llama_index.program import MultiModalLLMCompletionProgram
from llama_index.output_parsers import PydanticOutputParser
from llama_index import SimpleDirectoryReader

from decouple import config

GOOGLE_API_KEY = config("GOOGLE_API_KEY")
MODEL_NAME = "models/gemini-pro-vision"

class FootballPlayer(BaseModel):
    """Data model of description of football player"""
    nationality: str
    date_of_birth: str
    place_of_birth: str
    height: float
    weight_in_kilograms: float
    
    
prompt_template_str = """\
    Give me a summary of the person in the image\
    and return your respones with a json format\
"""

def structured_response_gemini(
    output_class: FootballPlayer,
    image_documents: list,
    prompt_template_str: str,
    model_name: str = MODEL_NAME
):
    gemini_llm  = GeminiMultiModal(
        api_key=GOOGLE_API_KEY,
        model_name=model_name
    )
    
    
    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=output_class),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=gemini_llm,
        verbose=True
    )
    
    response = llm_program()
    
    return response

# Load images
image_documents = SimpleDirectoryReader(input_dir="./images/").load_data()

# get response
for image_doc in image_documents:
    structured_respose = structured_response_gemini(
        output_class=FootballPlayer,
        image_documents=[image_doc],
        prompt_template_str=prompt_template_str,
        model_name=MODEL_NAME
    )
    
    print(structured_respose)