import streamlit as st
from PIL import Image

import easyocr

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Clarifai

from langchain.embeddings.openai import OpenAIEmbeddings

import re
import openai
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ReadOnlySharedMemory
from langchain.tools import StructuredTool
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.agents import initialize_agent, AgentType, load_tools
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')



wrapper = DuckDuckGoSearchAPIWrapper(region="in-en", max_results=10)
ddgsearch = DuckDuckGoSearchResults(api_wrapper=wrapper)

def duck_search_sale(query):
    duck_search_results = ddgsearch(query)
    duck_search_results = duck_search_results.lower()
    return duck_search_results


def extract_text_using_easyocr(image_path, language='en'):
    try:
        # Initialize the EasyOCR reader with the specified language
        reader = easyocr.Reader([language])

        # Read text from the image
        result = reader.readtext(image_path)

        # Extract and return the text from the result
        extracted_text = ' '.join([item[1] for item in result])

        return extracted_text
    except Exception as e:
        return str(e)
    
clarifai_llm = Clarifai(
    pat=pat_key, user_id= "meta", app_id="Llama-2", model_id = "llama2-70b-chat"
)

clarifai_llm_2=Clarifai(
    pat=pat_key, user_id= "clarifai", app_id="ml", model_id = "llama2-13b-chat-alternative"
)

clarifai_openai= Clarifai(
    pat=pat_key, user_id= "openai", app_id="chat-completion", model_id = "GPT-4"
)

template = """
<s>[INST] <<SYS>>

You are an expert dermatologist. You have to provide relevant answer to the user's 'Query' that you have recieved under Paragraph section below. You exactly only reply to the 'Query' provided below under Paragraph section without any hallucinations and false information.
Remember, You are aware that context for this 'Query' is being extracted from image by using OCR tool and as such there may be some errors in the extraction.\
OCR Tool will attempt to account for some words being swapped with similar-sounding words or may also be irregular or incomplete.\
You have to understand this 'Query' interms of dermatology field.
<</SYS>>

Paragraph
Query: {query}
 [/INST]

"""


prompt = PromptTemplate(template=template, input_variables=["query"])

llm_chain_ocr = LLMChain(prompt=prompt, llm=clarifai_llm)



#TOOL
desc_ocr="""

Use this tool exclusively when OCR extracted texts from image is provided as input. String begins with 'This is the OCR oputput'. This tool accepts single parameter called user query

"""

class OCRTool(BaseTool):
  name='OCR Output'
  description=desc_ocr

  def _run(self,query:str)-> str:
    data=llm_chain_ocr.run(query=query)
    return data

  def _arun(self,symbol):
    raise NotImplementedError("This tool doesnt support async")

ocrtool=OCRTool()

template = """
<s>[INST] <<SYS>>

You are an expert dermatologist. You have to provide relevant answer to the user based on the 'Context' and 'Query' you recieved. Both 'Query' and 'Context' is under the Paragraph section below. You exactly only reply to the 'Query' provided below under Paragraph section without any hallucinations and false information.
Remember, You are aware that 'Context' is being extracted from the internet ot the web search and as such there may be some errors.\
You have to understand this 'Context' interms of dermatology field. Please include the extensive knowledge that you posses from your training in dermatology while giving recommendation or while answering users 'Query' inaddition to the 'Context'
<</SYS>>

Paragraph
Query: {query}
Context: {context}
 [/INST]

"""

prompt = PromptTemplate(template=template, input_variables=["query","context"])

llm_chain_search = LLMChain(prompt=prompt, llm=clarifai_llm)

desc_search="""

Use this tool to search the internet when only the query is given. This tool accepts single parameter called user query

"""

class SearchTool(BaseTool):
  name='General Search'
  description=desc_ocr

  def _run(self,query:str)-> str:
    data=llm_chain_search.run(query=query,context=duck_search_sale(query))
    return data

  def _arun(self,symbol):
    raise NotImplementedError("This tool doesnt support async")

searchtool=SearchTool()

sys_msg ="""
<s><<SYS>>
You are an expert assistant chatbot in dermatology trained by 'DermaVision AI' for assisting users in their skin health related search queries.\
You can use these tools 'OCR Output', 'General Search'  wisely for the queries. 'OCR Output' is used when the raw text from OCR is passed as an input along with the query. 'General search' is used when standalone question is asked on the skin health.
You are constantly learning and training. You are capable of answering all dermatology related queries effectively. You never hallucinate answers, you always give authentic answers to best of your ability without any false information.
If user says Hi, respond with Hello! How can I assist you Today?
You always give indepth answers to users with detailed explanations step by step.
 Do not answer any private, general questions other than dermatology related user queries
 You should ask users necessary follow up questions before proceeding to use tools.
<</SYS>>
""" 

#Chain 2 is the agent

tools = [

    Tool(name = "OCR Output",
         func = ocrtool._run,
         description = desc_ocr,
         return_direct = False

    ),

    Tool(name = "General Search",
         func = searchtool._run,
         description = desc_search,
         return_direct = False

    )]

conversational_memory = ConversationBufferWindowMemory(
        memory_key = "chat_history",
        k = 6,
        return_messages=True,
)

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=clarifai_openai,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory,
    handle_parsing_errors=True,

)


new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools

)

agent.agent.llm_chain.prompt = new_prompt



# Streamlit app layout
st.title("DermVision AI")

# Text input
query = st.text_input("Ask all your doubts here:", "")

# Image input
uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])

# Button to call agent.run() function
if st.button("Ask Agent"):
    # Check if an image was uploaded
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        string = extract_text_using_easyocr(image)
        string_ocr = f"This is the OCR output : {string}.{query}"
    else:
       string_ocr = query

    # Call agent.run() function
    results = agent.run(string_ocr)
    st.write(results)

