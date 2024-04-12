from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from src.helper import *
from flask import Flask, render_template, jsonify, request


app = Flask(__name__)

# Load the PDF file
loader = DirectoryLoader('data/',
                         glob="*.pdf",
                         loader_cls=PyPDFLoader)

documents = loader.load()

# Split Text into Chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                             chunk_overlap=50)

text_chunks = text_splitter.split_documents(documents)

# Load the Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':'cpu'})

# Convert the Text Chunks into Embeddings and Create a FAISS Vectore Store
vector_store = FAISS.from_documents(text_chunks, embeddings)

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':256,
                            'temperature':0.01})

qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])

chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                   return_source_documents=False,
                                    chain_type_kwargs={'prompt':qa_prompt})

# user_input= "Tell me about Rainfall Measurement of the paper."

# result=chain.invoke({'query':user_input})
# print(f"Answer: {result['result']}")



@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())


@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    
    if request.method=="POST":
        user_input=request.form['question']
        print(user_input)
        
        result = chain.invoke({'query':user_input})
        print(f"Answer:{result['result']}")
    
    return jsonify({"response":str(result['result'])})


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)