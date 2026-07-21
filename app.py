import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="IntelliDocs AI",
    page_icon="📄",
    layout="wide"
)


# ---------------- SESSION STATE ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "file_name" not in st.session_state:
    st.session_state.file_name = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "pages" not in st.session_state:
    st.session_state.pages = 0


# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("📄 IntelliDocs AI")

    st.write("### AI PDF Assistant")

    st.write(
        """
Powered by:

✅ Ollama  
✅ Llama 3.2  
✅ LangChain  
✅ FAISS  
✅ HuggingFace Embeddings
"""
    )

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()


    st.divider()


    if st.session_state.file_name:

        st.success("🟢 PDF Ready")

        st.write(
            f"""
📄 File:

{st.session_state.file_name}


📃 Pages:

{st.session_state.pages}


📦 Chunks:

{len(st.session_state.chunks)}
"""
        )

    else:

        st.info(
            "Upload a PDF to start"
        )


# ---------------- TITLE ---------------- #

st.title("📄 IntelliDocs AI")

st.write(
    "Chat with your PDF using Local AI + RAG"
)


# ---------------- MODEL ---------------- #

llm = OllamaLLM(
    model="llama3.2"
)


# ---------------- UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)



# ---------------- PDF PROCESSING ---------------- #

if uploaded_file:


    if st.session_state.file_name != uploaded_file.name:


        st.session_state.messages = []

        st.session_state.vector_store = None

        st.session_state.file_name = uploaded_file.name



    if st.session_state.vector_store is None:


        try:

            with st.spinner(
                "📚 Reading PDF..."
            ):


                pdf = PdfReader(
                    uploaded_file
                )


                st.session_state.pages = len(pdf.pages)


                text = ""


                for page in pdf.pages:

                    content = page.extract_text()

                    if content:

                        text += content



            splitter = RecursiveCharacterTextSplitter(

                chunk_size=500,

                chunk_overlap=100

            )


            chunks = splitter.split_text(
                text
            )


            st.session_state.chunks = chunks



            with st.spinner(
                "🧠 Creating AI Knowledge Base..."
            ):


                embeddings = HuggingFaceEmbeddings(

                    model_name=
                    "sentence-transformers/all-MiniLM-L6-v2"

                )


                st.session_state.vector_store = FAISS.from_texts(

                    texts=chunks,

                    embedding=embeddings

                )


            st.success(
                "✅ PDF processed successfully"
            )


        except Exception as e:

            st.error(
                f"Error processing PDF: {e}"
            )



# ---------------- CHAT HISTORY ---------------- #

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )



# ---------------- CHAT INPUT ---------------- #

question = st.chat_input(
    "Ask something about your PDF..."
)



if question:


    if st.session_state.vector_store is None:


        st.warning(
            "Please upload a PDF first"
        )


    else:


        st.session_state.messages.append(

            {
                "role": "user",
                "content": question
            }

        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )



        with st.chat_message(
            "assistant"
        ):


            with st.spinner(
                "🤖 Searching and thinking..."
            ):


                docs = (
                    st.session_state
                    .vector_store
                    .similarity_search(
                        question,
                        k=3
                    )
                )


                context = "\n\n".join(

                    [
                        doc.page_content
                        for doc in docs
                    ]

                )


                prompt = f"""

You are an AI PDF assistant.

Answer only using the context.

If the answer is not present,
say:
"I couldn't find this information in the PDF."


Context:

{context}


Question:

{question}


Answer:

"""


                answer = llm.invoke(
                    prompt
                )



            st.markdown(
                answer
            )


            st.divider()


            st.write(
                "📚 Sources Used:"
            )


            for i, doc in enumerate(docs):

                with st.expander(
                    f"Source {i+1}"
                ):

                    st.write(
                        doc.page_content
                    )



        st.session_state.messages.append(

            {
                "role": "assistant",
                "content": answer
            }

        )


