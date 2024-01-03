from pydantic import BaseModel
import secrets

from llama_index.multi_modal_llms import GeminiMultiModal
from llama_index.program import MultiModalLLMCompletionProgram
from llama_index.output_parsers import PydanticOutputParser
from llama_index import SimpleDirectoryReader

from decouple import config

import pandas as pd

import streamlit as st

GOOGLE_API_KEY = config("GOOGLE_API_KEY")
MODEL_NAME = "models/gemini-pro-vision"


class PersonAttributes(BaseModel):
    """Data model of description of person"""
    name: str
    nationality: str
    date_of_birth: str
    place_of_birth: str
    latitude_of_place_of_birth: float
    longitude_of_place_of_birth: float
    height: float
    weight_in_kilograms: float


prompt_template_str = """\
    Give me a summary of the person in the image\
    and return your respones with a json format\
"""


def structured_response_gemini(
    output_class: PersonAttributes,
    image_documents: list,
    prompt_template_str: str,
    model_name: str = MODEL_NAME
):
    gemini_llm = GeminiMultiModal(
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


def get_details_from_multimodal_gemini(uploaded_image):
    """get response"""
    for image_doc in uploaded_image:
        data_list = []
        structured_respose = structured_response_gemini(
            output_class=PersonAttributes,
            image_documents=[image_doc],
            prompt_template_str=prompt_template_str,
            model_name=MODEL_NAME
        )

        for r in structured_respose:
            data_list.append(r)

        data_dict = dict(data_list)

        return data_dict


uploaded_file = st.file_uploader(
    "Choose An Image File",
    accept_multiple_files=False,
    type=["png", "jpg"]
)


if uploaded_file is not None:
    st.toast("File uploaded successfully")
    byte_data = uploaded_file.read()
    st.write("Filename: ", uploaded_file.name)

    with st.spinner("Loading, please wait"):
        if uploaded_file.type == "image/jpeg":
            file_type = "jpg"
        else:
            file_type = "png"

        # save file
        filename = f"{secrets.token_hex(8)}.{file_type}"

        with open(f"./images/{filename}", "wb") as fp:
            fp.write(byte_data)

        file_path = f"./images/{filename}"

        # load images
        image_documents = SimpleDirectoryReader(
            input_files=[file_path]
        ).load_data()

        response = get_details_from_multimodal_gemini(
            uploaded_image=image_documents
        )

        with st.sidebar:
            st.image(image=file_path, caption=response.get("name", "Unknwon"))
            st.markdown(f"""
                    :green[Name]: :red[{response.get("name", "Unknwon")}]\n
                    :green[Nationality]: :violet[{response.get("nationality", "Unknwon")}]\n
                    :green[Date Of Birth]: :gray[{response.get("date_of_birth", "Unknwon")}]\n
                    :green[Place Of Birth]: :orange[{response.get("place_of_birth", "Unknwon")}]\n
                    :green[Height]: :red[{response.get("height", "Unknwon")}]\n
                    :green[Weight In Kilograms]: :red[{response.get("weight_in_kilograms", "Unknwon")}]\n
                        """)

        df = pd.DataFrame(
            {"latitude": response.get("latitude_of_place_of_birth", 0.0),
             "longitude": response.get("longitude_of_place_of_birth", 0)},
            index=[0]
        )

        st.map(df, latitude="latitude", longitude="longitude")
