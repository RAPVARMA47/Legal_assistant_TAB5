
import base64
import streamlit as st 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI  
import os
import time
from langchain_community.chat_models import ChatOllama  


# Set page config
st.set_page_config(page_title="Legal Assistant", page_icon="⚖️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff;
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4682B4;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #E6F3FF;
    }
    [id="ec04b9b3"]{
        margin-top : -100px;
        font-size:30px;
        margin-left : 90px;
        
    }
    
    .st-emotion-cache-ocqkz7 {
    display: flex;
    flex-wrap: wrap;
    -webkit-box-flex: 1;
    flex-grow: 1;
    -webkit-box-align: stretch;
    align-items: stretch;
    gap: 1rem;
    width: 1100px;
    }
    .st-b6 {
    border-bottom-color: rgb(32 122 199);
    }


    .st-b5 {
        border-top-color: rgb(32 122 199);
    }

    .st-b4 {
        border-right-color: rgb(32 122 199);
    }

    .st-b3 {
        border-left-color: rgb(32 122 199);
    }

    """, unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Title and description
st.title("🏛️ Advanced Legal Assistant for Women and Children")
st.markdown("Welcome to your AI-powered legal research companion. Ask any legal question, and get comprehensive answers backed by relevant case law.")

# Initialize Hugging Face embeddings model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings()

core_embeddings_model = load_embeddings()

@st.cache_resource
def load_vector_db():
    return FAISS.load_local("INDEX", core_embeddings_model,allow_dangerous_deserialization=True)

vector_db = load_vector_db()

# Define the prompt template for the LLM
prompt_template ="""

{question}
---
1. Legal Principle:

- **Insert Legal Principle Title Here:**  
  Provide a brief, clear description of the main legal rule, doctrine, or statute that applies to the lawyer's question. This section should explain the foundational legal principles underpinning the issue at hand, including key elements of the rule or doctrine.
---
2. Case Reference:
- **Case Name:** ***Insert Case Name Here***  
- **Citation:** *Insert Full Legal Citation Here*  
- **Hyperlink to Case:** [Provide Hyperlink to the Case for Reference]  

- **Relevance:**  
  Explain why this case is relevant to the question being asked. How does the case support or illustrate the legal principle discussed? Select a case directly applicable to the lawyer's jurisdiction if possible, and provide a brief explanation of its importance in this legal context.
---
3. Implications:
- Discuss the broader significance of the legal principle in the context of the law. What does this principle mean for current and future legal cases? How does it affect the legal landscape? Consider the practical implications for both the current case and similar cases going forward. 
  Include potential consequences, precedents set by the principle, or conflicts with other areas of law.
---
4. Court Decision:
- **Court's Ruling:**  
  Summarize the court’s ruling in the cited case, emphasizing how the court applied the legal principle to the specific facts of the case. 
  Focus on the key reasoning or interpretations made by the court that are pertinent to the lawyer’s query.
- **Key Reasoning:**  
  Highlight any significant legal interpretations, doctrines, or principles that the court relied on when reaching its decision. 
  This should connect back to the lawyer’s question and provide practical insights into how the legal principle is applied.
---
5. Summary:
- Provide a concise overall summary that integrates the legal principle, case reference, implications, and court decision.
  This section should distill the information into a clear and actionable insight that directly addresses the lawyer's question. Ensure that the summary provides a practical takeaway for the lawyer’s case preparation or strategy.
---
6. Source:
- Provide the source(s) from which the legal information was retrieved (e.g., legal database, case law citation, statutory references). Include any valid hyperlinks for further reference.
{context}
"""
api_key=st.secrets["OPENAI_API_KEY"]
# Initialize the language model (ChatOpenAI)
@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

llm = load_llm()

# Create a retriever from the vector store
retriever = vector_db.as_retriever()

# Set up prompt template for LLM chain
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

# Document combination chain setup
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)
combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
)

# Set up RetrievalQA chain
qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    retriever=retriever,
    return_source_documents=True,
)
def set_query(query):
    st.session_state.query_input = query
    st.rerun()

# Initialize session state for the query if it doesn't exist
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# The text input field, populated with the query stored in session state
query = st.text_input("Enter your query:", value=st.session_state.query_input)

st.write("Sample Prompts:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("What are the main ways the Act prevents sexual harassment?"):
        set_query("What are the main ways the Act prevents sexual harassment?")
with col2:
        
    if st.button("Penalties in the Child labour prohibition and regulation act"):
        set_query("Penalties in the Child labour prohibition and regulation act")

with col3:
            
    if st.button("How does the Act help women who face sexual harassment at work?"):
        set_query("How does the Act help women who face sexual harassment at work?")

with col4:

    if st.button("Legitimacy of children born of child marriages."):
        set_query("Legitimacy of children born of child marriages.")
        

# Process button
if st.button("Get Legal Insights"):
    if query:
        with st.spinner("Analyzing your question..."):
            start_time = time.time()
            try:
                # Invoke the retriever with the user query
                response = qa.invoke({"query": query})
                
                # Display response
                st.markdown("### Legal Analysis", unsafe_allow_html=True)
                st.markdown(response["result"], unsafe_allow_html=True)
                
                # Display processing time
                st.info(f"Processing time: {time.time() - start_time:.2f} seconds")
                
                # Display sources
                st.markdown("### Sources", unsafe_allow_html=True)
                unique_sources = set()
                top_n = 3  # Set how many top documents to print
                unique_sources = set()  # Track unique sources
                document_count = 0
                result_sources = []  # List to store the sources

                base_url = r"C:\Users\aipro\Documents\Fw_ Meeting with Promptora AI\Laws"

                for doc in response['source_documents']:
                    source = doc.metadata['source']
                    
                    if source not in unique_sources:
                        unique_sources.add(source)
                        document_count += 1
                        
                        # Extract just the filename without the path
                        filename = os.path.basename(source)
                        
                        # Replace .txt with .pdf
                        filename = filename.replace('.txt', '.pdf')
                        
                        # Create a hyperlink with "click here" text
                        result_sources.append(f"[{filename}]")
                        
                        if document_count == top_n:
                            break

                # Join the sources into a newline-separated string
                result_source_string = "\n\n".join(result_sources)
                print(result_source_string)
                st.markdown(f"result:{result_source_string}",unsafe_allow_html=True)
                            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question before submitting.")

