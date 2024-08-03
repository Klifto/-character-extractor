from gliner import GLiNER
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from difflib import SequenceMatcher
from langchain.llms import OpenAI
import os 
from langchain_openai import AzureChatOpenAI


from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
import textwrap
import faiss
from dotenv import load_dotenv

from prompts import character_description_prompt

load_dotenv()

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
    return wrap_text_preserve_newlines(llm_response['result'])

def load_embeddings(store_name, path):
    """
    Función apra cargar los embeddings generados con anterioridad

    Args:
        store_name: nobre del archivo que contiene lso embeddings
        path: ruta donde se encuentra el archivo
    """
    with open(f"{path}/faiss_{store_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
    return VectorStore

def load_data(path: str, chunk_size: int, chunk_overlap: int):
    """
    Función que carga los pdfs de un directorio y los divide siguiendo los criterios que vienen definidos por parámetros

    Args:
        path: ruta de la carpeta que donde están los pdfs que desamos procesar
        chunk_size: tamaño del chunk con el que dividir los documentos
        chunk_overlap: número de elemntos que se solaparán en cada división de chunks
    """
    loader = DirectoryLoader(path, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
                                               chunk_size=chunk_size, 
                                               chunk_overlap=chunk_overlap)

    texts = text_splitter.split_documents(documents)
    print(f"Texto cargado correctametne y dividido en {len(texts)} partes.")
    return texts

def main():
    text = load_data(path=f'.//books', chunk_size=1000, chunk_overlap=0)
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    labels = ["person"]
    
    characters_set = set()
    for index ,chunck in enumerate(text):
        print(f"Element {index} of {len(text)}.") 
        ner_results = model.predict_entities(chunck.page_content, labels)
        for ner_result in ner_results:
            if ner_result["score"] > 0.8:
                characters_set.add(ner_result["text"])
    
    print(characters_set)
    new_characters_set = set()

    for i in characters_set:
        i = i.replace(' -\n', "")
        i = i.replace("\n", "")
        i = i.replace("  ", " ")
        new_characters_set.add(i.lower())
    with open("./characters_list/characters.txt", "w", encoding='utf-8') as f:
        for i in new_characters_set: 
            f.write(i + '\n')

def process_characters():
    with open("./characters_list/characters.txt", "r", encoding='utf-8') as f:
        elements = f.read()
    elements = elements.replace("\n", ", ")
    #elements = [elements,[]]
    #find_similars(elements_list=elements)
    llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2024-05-01-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )
    messages = [
    (
        "system",
        "You are a helpful assistant your job is to take a list of characters from a novel and cluster these list with all the different names that every character has in the list.",
    ),
    ("human", f"Cluster the following list of characters: {elements}"),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)
    with open("./characters_list/characters_final.txt", "w", encoding='utf-8') as f:
        for i in ai_msg.content.split('\n'): 
            f.write(i + '\n')
    return elements

def generate_descriptions():
    #with open("characters_final.txt", "r", encoding='utf-8') as f:
    with open("./characters_list/characters_test.txt", "r", encoding='utf-8') as f:
        elements = f.read()
    
    llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2024-05-01-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )

    Embedding_store_path = f".//Embedding_store"
    try:
        #Intentamos cargar los embeddings en caso de tenerlos calculados de forma previa
        db_instructEmbedd = load_embeddings(store_name='instructEmbeddings', path=Embedding_store_path)
    except Exception as e:
        print(f"No se pudieron cargar los embeddings pasamos a generarlos. {e}")
    list_elements = elements.split("\n")
    charater_descriptions = []
    retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 30}) #Definicion buscador

    #Cargamos un modelo jusnto con el retirever
    qa_chain_instrucEmbed = RetrievalQA.from_chain_type(llm=llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)
    #with open("characters_final_solution.txt", "w", encoding='utf-8') as f:
    with open("./characters_list/characters_final_solution_test.txt", "w", encoding='utf-8') as f:
        for i in list_elements: 
            f.write(i + '\n')
            query=character_description_prompt + f"{i}"
            llm_response = qa_chain_instrucEmbed(query)
            f.write('\n' + process_llm_response(llm_response) + '\n')
    return None

def find_similars(elements_list):
    if elements_list[0]:
        similar_elements = []
        core_element = elements_list[0][0]
        similar_elements.append(core_element)
        for position, rest_elemet in enumerate(elements_list[0][1:]):
            if similar(core_element, rest_elemet) > 0.6:
                similar_elements.append(rest_elemet)
                elements_list[0].pop(position)
        elements_list.append(similar_elements)
        elements_list[0].pop(0)
        find_similars(elements_list)
    else:
        return elements_list

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

if __name__=="__main__":
    #main()
    #process_characters()
    generate_descriptions()