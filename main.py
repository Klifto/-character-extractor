from gliner import GLiNER
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from difflib import SequenceMatcher
from langchain_openai import AzureChatOpenAI


from langchain.chains import RetrievalQA
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
import textwrap
from dotenv import load_dotenv

from prompts import character_description_prompt,system_message,system_message_metric
import nltk
from nltk.corpus import stopwords

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
    Función para cargar los embeddings generados con anterioridad

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

def store_embeddings(docs, embeddings, store_name, path):
    """
    Función para generar y almacenar los embedings generados

    Args:
        docs: documentos sobre los que generamos embedings
        embeddings: modelo a usar para la generación de embeddings
        store_name: nombre del archivo pickle en el que guardaremos los embeddings
        path: ruta donde se guardarán los embeddings
    """
    vectorStore = FAISS.from_documents(docs, embeddings)

    with open(f"{path}/faiss_{store_name}.pkl", "wb") as f:
        pickle.dump(vectorStore, f)
    
    return None

def extract_characters():
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

    nltk.download('stopwords')
    stop_wrods = stopwords.words('english')

    for i in characters_set:
        if i.lower() in stop_wrods:
            pass
        else:
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
        system_message,
    ),
    ("human", f"Extract a list with all the different names for each character if the list of names is: {elements}"),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)
    with open("./characters_list/characters_final.txt", "w", encoding='utf-8') as f:
        for i in ai_msg.content.split('\n'): 
            f.write(i + '\n')
    return elements

def generate_descriptions():
    with open("./characters_list/characters_test.txt", "r", encoding='utf-8') as f:
        elements = f.read()
    
    llm = AzureChatOpenAI(
        azure_deployment="chatgpt16",
        api_version="2024-05-01-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )

    Embedding_store_path = f".//Embedding_store_harry"
    try:
        #Intentamos cargar los embeddings en caso de tenerlos calculados de forma previa
        db_instructEmbedd = load_embeddings(store_name='instructEmbeddings', path=Embedding_store_path)
    except Exception as e:
        instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"})
        texts = load_data(path=f'.//books', chunk_size=1000, chunk_overlap=200)
        print(f"No se pudieron cargar los embeddings pasamos a generarlos. {e}")
        store_embeddings(texts, 
                  instructor_embeddings, 
                  store_name='instructEmbeddings', 
                  path=Embedding_store_path)
        db_instructEmbedd = load_embeddings(store_name='instructEmbeddings', path=Embedding_store_path)
    list_elements = elements.split("\n")
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
            try:
                llm_response = qa_chain_instrucEmbed(query)
                f.write('\n' + process_llm_response(llm_response) + '\n')
            except Exception as e:
                print(f"Fail to generate respond reason: {e}")    
    return None

def get_performance():
    with open("./characters_list/characters_final_solution_test.txt", "r", encoding='utf-8') as f:
        elements = f.read()
    with open("./characters_list/wiki_harry_potter_list.txt", "r", encoding='utf-8') as f:
        elements_true = f.read()

    llm = AzureChatOpenAI(
        azure_deployment="chatgpt16",
        api_version="2024-05-01-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )
    
    messages = [
    (
        "system",
        system_message_metric,
    ),
    ("human", f"Give us the performance if the generated list description is [{elements}] and the true list description is [{elements_true}]. Do it step by step."),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)
    return None

if __name__=="__main__":
    #extract_characters()
    #process_characters()
    #generate_descriptions()
    get_performance()