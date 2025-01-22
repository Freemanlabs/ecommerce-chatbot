import dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools import check_order_status, save_user_info

dotenv.load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# embeddings = OpenAIEmbeddings()

loader = TextLoader(file_path="data/return_policy.txt")
docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=20)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
doc_splits = text_splitter.split_documents(docs)

vector_store = Chroma.from_documents(documents=doc_splits, collection_name="rag_chroma", embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

retrieve_return_policy = create_retriever_tool(
    retriever,
    "return_policy",
    "Search and return the relevant information about return policy",
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None, max_retries=2)
# llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2)
