import os
import chromadb
import streamlit as st
import tiktoken
from llm import LLMClient
from dotenv import load_dotenv
from llama_index.core.node_parser.text.token import TokenTextSplitter
from style import header_content
from util import init_session_state, RagEmbedder, \
    RagRanker, export_chat_to_pdf, display_chat_history, \
    query_component, upload_handler

# download environment variables
load_dotenv()
docs_dir = os.environ['DOCUMENTS_DIR']
collection_name = os.environ['COLLECTION_NAME']
current_collection_name = os.environ["CURRENT_COLLECTION_NAME"]
emb_model_name = os.environ["EMB_MODEL_NAME"]
tokens_per_chunk = int(os.environ["TOKENS_PER_CHUNK"])
cross_encoder_name = os.environ["CROSS_ENCODER_NAME"]
port = os.environ["PORT"]
host = os.environ["HOST"]

print(emb_model_name)
print(cross_encoder_name)

# init the session state
init_session_state(st)

st.session_state.uploaded_files = []

if not st.session_state.llm_client:
    st.session_state.llm_client = LLMClient()

if not st.session_state.cross_encoder:
    st.session_state.cross_encoder = RagRanker(cross_encoder_name)

if not st.session_state.emb_fun:
    st.session_state.emb_fun = RagEmbedder(emb_model_name)

if not st.session_state.chroma_client:
    st.session_state.chroma_client = chromadb.HttpClient(host=host, port=port)

if not st.session_state.current_chroma_client:
    st.session_state.current_chroma_client = chromadb.Client()

if not st.session_state.chroma_collection:
    st.session_state.chroma_collection = st.session_state.chroma_client.get_or_create_collection(name=collection_name,
                                                                                                 embedding_function=st.session_state.emb_fun)

def uploaded_files_state(st):
    st.session_state.is_new_upload = True

if not st.session_state.token_splitter:
    st.session_state.token_splitter = TokenTextSplitter(
        separator=". ",
        chunk_size=200,
        chunk_overlap=20,
        backup_separators=[
            "\n\n\n", "\n\n", "\r\n", "\r", "\t", "! ", "? ", ": ", "; ", ", ",
        ],
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )

if not st.session_state.user_name:
    st.session_state.user_name = "User"

########################| Main Content |#############################
st.markdown(header_content(), unsafe_allow_html=True)

########################| Side Bare |#############################
st.sidebar.title('JURID.IA')
st.sidebar.markdown("***")
st.sidebar.markdown('Votre premier assistant juridique au Maroc 🇲🇦')
st.sidebar.markdown("***")
with st.container(border=True):
    is_upload = st.sidebar.toggle('Upload Files ?')
st.sidebar.markdown("***")
file_types = ["pdf", "txt", "docx"]
uploaded_files = st.sidebar.file_uploader("Déposez vos fichiers ici",
                                          type=file_types,
                                          on_change=(lambda: uploaded_files_state(st)),
                                          accept_multiple_files=True)

st.sidebar.markdown("***")
st.sidebar.write('Développé avec ❤️ par [Ben Alla Ismail](https://www.linkedin.com/in/omar-el-adlouni/)')


########################| Add Uploaded files |#############################
if uploaded_files:
    upload_handler(st, uploaded_files, current_collection_name)

########################| Display Chat History |#############################
if st.session_state.chat_history:
    display_chat_history(st, uploaded_files)

########################| From Database Files |#############################
if st.session_state.llm_client and (st.session_state.chroma_collection or st.session_state.current_chroma_collection):
    query_component(st, is_upload)

##########################################################
if st.button("Export Chat to PDF"):
    if st.session_state.chat_history:
        export_chat_to_pdf(st.session_state.chat_history)
        st.success("The chat has been exported to PDF. Check your downloads folder.")
    else:
        st.info("You did not start you chat yet :)")
