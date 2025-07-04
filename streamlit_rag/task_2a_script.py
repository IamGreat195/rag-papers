import os
import PyPDF2
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.load import dumps, loads
from operator import itemgetter
import streamlit as st



@st.cache_resource(show_spinner=False) # we are caching the vector store to avoid reloading it every time we click somewhere in the website
def load_vector_store():
    # File mapping[we are just mapping the file names to their titles so that we can use them later in the metadata of the documents]
    filename_to_title = {  
        "attention is all you need.pdf": "Attention Is All You Need",
        "BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf": "BERT",
        "GPT-3 Language Models are Few-Shot Learners.pdf": "GPT-3",
        "Contrastive Language-Image Pretraining with Knowledge Graphs.pdf": "CLIP",
        "LLaMA Open and Efficient Foundation Language Models.pdf": "LLaMA"
    } 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # we are splitting the text into chunks of 1000 characters with an overlap of 200 characters
    document_chunks = []

    def extract_text(pdf_path):
        text = ""
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file) # Read the PDF file
            # Loop through each page in the PDF and extract text
            for page in pdf_reader.pages:
                text += page.extract_text() 
        return text

    for filename in os.listdir('research_papers'):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join('research_papers', filename)
            full_text = extract_text(pdf_path)
            title = filename_to_title.get(filename, "Unknown Paper") # getting the corresponding title from the dictionary mapping
            chunks = text_splitter.split_text(full_text)
            for chunk in chunks:
                document_chunks.append(Document(page_content=chunk, metadata={"source": title})) # we are creating a Document object for each chunk with the title as metadata

    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') # using a pre-trained sentence transformer model for embeddings

    vector_store = Chroma.from_documents( # creating a vector store from the documents for efficient retrieval
        documents=document_chunks,
        embedding=embedding_model,
        persist_directory="chroma_db" # persist is needed to save the vector store to disk so that we can load it later without reprocessing the documents
    )

    return vector_store





# %% [markdown]
# # Retrieval

# %% [markdown]
# Lets set up the langchain for analysis

# %%
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'

# %% [markdown]
# Chat model set up

# %%
from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyDfeVwwdJojhi6Ose_GSKg-Eb_g0zpNR48"


llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0) # i am using gemini-2.0-flash model for my chat model

# %% [markdown]
# # Generation

# %% [markdown]
# RAG Fusion

# %% [markdown]
# For generation let us use Rag fusion which is basically an advanced version of multi query and here we will generate 6 queries and instead of just directly de duplicating and joining the various documents we get using our 6 prompts we apply a rank to them using rrf and take in only the highest ranked documents

# %%
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

vector_store = load_vector_store() # we load the vector store from the cached function
retriever = vector_store.as_retriever() # we create a retriever from the vector store to retrieve documents based on the generated queries


template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (6 queries):""" # rag fusion template to generate multiple queries based on a single input query

prompt_rag_fusion = ChatPromptTemplate.from_template(template) # we create a chat prompt template from the rag fusion template

generate_queries = (
    prompt_rag_fusion # we pass the template into gemini, remove the text wrapper for chats and split the 5 questions formatting them with new lines
    | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)


# %%
from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion



# %%
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

# RAG
template = """Answer the following question based on this context: 

{context}

Question: {question}
""" # rag template to answer the question based on the context retrieved from the vector store

prompt = ChatPromptTemplate.from_template(template)


final_rag_chain = ( # we create a final rag chain that takes the context and question as input and returns the answer
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)




# streamlit app
st.set_page_config(page_title="RAG Fusion QA", page_icon="🔍")
st.title("🔍 Ask Questions About ML Papers")

question = st.text_input("Enter your question:")

if question:
    with st.spinner("Retrieving and answering..."):
        retrieval_chain_rag_fusion.invoke({"question": question})
        answer = final_rag_chain.invoke({"question": question})
        st.markdown("### Answer:")
        st.success(answer)



