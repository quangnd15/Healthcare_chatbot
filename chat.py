import random
import torch 
from pyvi import ViTokenizer
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from qdrant_client import QdrantClient
import gradio as gr

from constants import INTRODUCE_CTX, prompt


embedding_model= SentenceTransformer('med_embedding')
embedding_model.to(dtype= torch.float32)
client= QdrantClient(path= "database")

llm= Llama(model_path= 'qwen_med_sft/qwen_sft_med_q4_k_m.gguf', n_gpu_layers= -1, n_ctx= 4096, chat_format= 'qwen', flash_attn= True)
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

def remove_threshold_search(hits, threshold= 0.7, return_text= False): 
    if not return_text:
        return [i for i in hits if i.score > threshold]
    else: 
        return [i.payload['text'] for i in hits if i.score > threshold]

def chat_llm(query, context): 
    response= llm.create_chat_completion(
        messages= [
            {'role': 'system', 'content': prompt.format(query, context)}
        ]
    )
    return response['choices'][0]['message']['content'].strip('### Answer:').strip()

def reranking(question, contexts):
    reranked_indices = []
    for passage in contexts:
        score = reranker.compute_score([[question, passage]])
        reranked_indices.append((passage, score))
    
    reranked_indices.sort(key=lambda x: x[1], reverse=True)
    return reranked_indices[0][0]

def search(question, threshold= 0.7, debug= True): 
    query_embedding= embedding_model.encode(ViTokenizer.tokenize(question).lower(), 
                                convert_to_numpy= True,  convert_to_tensor= False, show_progress_bar= True).tolist()

    hits= client.search(
        collection_name= "healthcare", 
        query_vector= query_embedding, 
        limit= 10
    )
    if debug:
        for i in hits:
            print(i)
    
    return remove_threshold_search(hits, threshold, return_text= True)


def retrieval(question, threshold_search= 0.7, debug= True): 
    doc_query= search(question, threshold_search, debug)
    
    if len(doc_query)==0:
        return ''
    else: 
        final_doc= reranking(question, doc_query)
        return final_doc
def chat(message, threshold_search= 0.7, debug= True): 
    doc_query= retrieval(message, threshold_search, debug)
    
    if doc_query == '':
        return chat_llm(message, INTRODUCE_CTX)
    
    return chat_llm(message, doc_query)

def respond(message, history):

    bot_message= chat(message)
    return bot_message

mychatbot= gr.Chatbot(
    value=[[None, random.choice(['hello', 'xin chào', 'chào bạn nha'])]],
    bubble_full_width= False, 
    show_label= False, 
    show_copy_button= True, 
    height= 550, 
    likeable=True, 
)

demo = gr.ChatInterface(fn= respond, chatbot= mychatbot, title= "Doctor Assistant")

demo.launch(show_api= False, share= True)