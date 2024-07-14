# RUN: chainlit run model.py

# 01: CONFIGURE
MODEL_ID="/work/u00cjz00/slurm_jobs/github/models/Llama-2-7b-chat-hf"
DB_FAISS_PATH = 'vectorstore/db_faiss'

# 02: Load LIBRARY
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import transformers
import torch
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer

# 03: custom_prompt_template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# 04: Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

# 05: Loading the model
#def load_llm2():
#    # Load the locally downloaded model here
#    llm = CTransformers(
#        model = "/work/u00cjz00/slurm_jobs/github/models/Taiwan-LLaMa-13b-1.0.Q8_0.gguf",
#        model_type="llama2",
#        max_new_tokens = 512,
#        temperature = 0.5
#    )
#    return llm

def load_llm():
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
    pipeline=transformers.pipeline(
        "text-generation",
        model=MODEL_ID,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
        )
    llm=HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0})
    
    return llm


# 05: QA Model Function
def qa_bot():
#    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
#                                       model_kwargs={'device': 'cpu'})
    embeddings = HuggingFaceEmbeddings(model_name='/work/u00cjz00/nvidia/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# 06: output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# 07: chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\n\n\n資料來源, Sources:" + str(sources)
    else:
        answer += "\n\n\nNo sources found"

    await cl.Message(content=answer).send()
    
    
    
