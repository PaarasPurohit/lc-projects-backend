from constants import gemini_api_key

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain

GOOGLE_API_KEY = gemini_api_key
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

examples = [
    {
        "description":"Outstanding service!",
        "review":"5"
    },
    {
        "description":"Not very good service...",
        "review":"1"
    },
    {
        "description":"I had an okay time.",
        "review":"3"
    }
]

example_formatter_template = """Description: {description}
Review: {review}
"""

example_prompt = PromptTemplate(
    input_variables=["description", "review"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Based on the input, give a review. It should be a number from 1 to 5 and accurately reflect the description input\n",
    suffix="Review: {input}\nDescription: ",
    input_variables=["input"],
    example_separator="\n",
)

chain = LLMChain(
    llm = llm,
    prompt = few_shot_prompt
)

def get_predicted_review(input):
    result = chain({'input': input})
    
    output_json = {
        "description": result['input'],
        "predicted_review": result['text'],
    }
    
    return output_json