# from langchain import PromptTemplate
# from langchain import LLMChain
# # from langchain.llms import LLMChain
# # from langchain.llms import CTransformers
# from langchain_community.llms import CTransformers 

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import CTransformers

from src.helper import *

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# instructions = "Convert the following text from English to Hindi: \n\n {text}"
instructions = "Give a proper summary of a : \n\n {text}"

SYSTEM_PROMPT = B_SYS + CUSTOM_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instructions + E_INST

prompt = PromptTemplate(template=template, input_variables=["text"])

llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type='llama',
                    config={'max_new_tokens':256,
                            'temperature': 0.01})

llm_chain=LLMChain(prompt=prompt, llm=llm)

result = llm_chain.invoke("Harry Porter")
result = llm_chain.invoke("where are you?")
# print(llm_chain.invoke("where are you?"))
print(result['text'])