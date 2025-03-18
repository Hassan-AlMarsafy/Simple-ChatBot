# Chatbot with LangChain and Groq API

## Overview
This project is a conversational chatbot built using [LangChain](https://www.langchain.com/) and the [Groq API](https://groq.com/). It utilizes document retrieval techniques and vector databases to provide context-aware responses.
chatbot-1 is for practice and understanding how the code works in contrast with the llm. chatbot-2 is a more refined version with an already persisted chroma vectordb.

## Features
- Uses **LangChain** for LLM-based conversational AI.
- Integrates with **Groq API** for generating responses.
- Supports **PDF document loading and retrieval**.
- Uses **HuggingFace embeddings** and **Chroma vectorstore** for efficient document search.
- Includes **memory storage** for conversation context.

## Installation
### Prerequisites
Ensure you have Python 3.8+ and install the required dependencies:

```bash
pip install langchain langchain_groq langchain_huggingface chromadb pypdf
```

## Setup
### Environment Variables
You need a Groq API key. Set it up by running:

```python
import getpass
import os

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
```

### Run the Chatbot
1. **Load PDF documents**:
    ```python
    from langchain.document_loaders import PyPDFLoader

    loaders = [
        PyPDFLoader("rfc7540.pdf"),
        PyPDFLoader("rfc7541.pdf")
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    ```

2. **Split Documents**:
    ```python
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    ```

3. **Create Vector Database**:
    ```python
    from langchain.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    
    persist_directory = 'docs/chroma/'
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    ```

4. **Set Up Retrieval and Memory**:
    ```python
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    retriever = vectordb.as_retriever()
    
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    ```

5. **Ask Questions**:
    ```python
    question = "Is probability needed to understand HTTP?"
    result = qa({"question": question})
    print(result["answer"])
    ```

## Usage
- The chatbot loads PDF documents, processes them, and creates an efficient retrieval system.
- You can ask it questions, and it will respond with context-aware answers.
- The conversation memory allows it to remember previous interactions.

## Future Improvements
- Add more document formats (e.g., DOCX, TXT).
- Improve retrieval accuracy with advanced embedding models.
- Deploy as an API or web-based chatbot.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing
Feel free to submit issues or pull requests to improve this chatbot!

---
Made with ❤️ using LangChain and Groq API.

