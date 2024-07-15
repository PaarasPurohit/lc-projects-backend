import os
import json
from constants import gemini_api_key

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain

# Configure Google API key
GOOGLE_API_KEY = gemini_api_key
genai.configure(api_key=GOOGLE_API_KEY)

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about the celestial object by the name of {name}"
)

# Memory
object_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
stats_memory = ConversationBufferMemory(input_key='name', memory_key='stats_history')
additional_info_memory = ConversationBufferMemory(input_key='name', memory_key='info_history')

# OPENAI LLMS
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
chain = LLMChain(
    llm=llm, prompt=first_input_prompt, verbose=True, output_key='object_info', memory=object_memory)

# Second Prompt Template
second_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="What are the statistics for the celestial object {name}"
)

chain2 = LLMChain(
    llm=llm, prompt=second_input_prompt, verbose=True, output_key='stats', memory=stats_memory)

# Third Prompt Template
third_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Important additional information for celestial object {name}"
)

chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='additional_info', memory=additional_info_memory)

parent_chain = SequentialChain(
    chains=[chain, chain2, chain3], input_variables=['name'], output_variables=['object_info', 'stats', 'additional_info'], verbose=True)

def get_celestial_object_info(name):
    result = parent_chain({'name': name})
    
    output_json = {
        "object-information": result['object_info'],
        "statistics": result['stats'],
        "additional-information": result['additional_info'],
        "object-name-buffer": object_memory.buffer,
        "statistics-buffer": stats_memory.buffer,
        "additional-information-buffer": additional_info_memory.buffer
    }
    
    return output_json
