## RAG with Openai LLMs Project Overview
This project is a document-based question-and-answer (Q&A) system built using **Chainlit**, **LangChain**, and **OpenAI** (or Azure OpenAI). Users can upload `.pdf` or `.docx` files, and the chatbot will process the text, store it in a vectorized format, and use a language model to answer questions based on the uploaded documents. The system focuses on document comprehension, leveraging embedding models to retrieve relevant information and provide answers with sources.

![image](https://github.com/user-attachments/assets/f8a0d1b3-f888-436c-b9f3-862c091013ba)



## Theoretical Explanation of Key Functions and Calls

### 1. **Environment Setup and Configuration Loading**

- **`load_dotenv()`**:
   - This function loads environment variables from a `.env` file. It allows the system to access critical configuration values like API keys, model settings, and application parameters, without hardcoding sensitive data into the source code.
   - The `.env` file might contain variables such as the `OPENAI_API_KEY` or `AZURE_OPENAI_MODEL` which are used later in the code to interact with the API for generating embeddings and completions.
   
   **Theoretical Perspective**:
   - This technique ensures security and portability. Sensitive configurations are kept outside the codebase, which is crucial for both development and production environments.

### 2. **`on_chat_start()` - File Upload Handling**

- **Asynchronous File Handling**:
   ```python
   files = await cl.AskFileMessage(content=f"Please upload up to {max_files} `.pdf` or `.docx` files to begin.").send()
   ```
   - This function waits for the user to upload `.pdf` or `.docx` files. It uses Chainlit's `AskFileMessage` to asynchronously collect files uploaded by the user.
   - After the files are uploaded, the function sends a message to inform the user that the files are being processed.

   **Theoretical Perspective**:
   - Asynchronous programming allows the server to handle multiple tasks (like file uploads) concurrently without blocking other operations. This is essential for web applications where multiple users might interact simultaneously.

### 3. **File Reading and Text Extraction**

- **`PdfReader` and `Document` Classes**:
   ```python
   reader = PdfReader(bytes)
   doc = Document(bytes)
   ```
   - Depending on the file type (`.pdf` or `.docx`), the corresponding library (PyPDF or python-docx) is used to extract text. For `.pdf` files, the `PdfReader` extracts text page by page. For `.docx` files, the `Document` object retrieves text from paragraphs.
   - These extracted texts are then stored for further processing.

   **Theoretical Perspective**:
   - File I/O with different document types needs specific parsing mechanisms due to their different internal structures. For example, `.pdf` files have structured text inside pages, whereas `.docx` files organize content into paragraphs. This differentiation allows the system to handle various formats efficiently.

### 4. **Text Chunking**

- **RecursiveCharacterTextSplitter**:
   ```python
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=text_splitter_chunk_size, chunk_overlap=text_splitter_chunk_overlap)
   texts = text_splitter.split_text(text)
   ```
   - After reading the document, the text is split into smaller chunks using a text splitter. The size of each chunk is controlled by parameters like `chunk_size` and `chunk_overlap`.
   - Chunking is essential because language models typically have input length limits, and smaller pieces of text are easier to manage for both embedding and retrieval.

   **Theoretical Perspective**:
   - Text chunking ensures that the vector embeddings can accurately represent pieces of the document. Without this step, long documents would exceed token limits of the language model, making it impossible to process them in one go. Overlapping chunks also allow for more accurate contextual understanding by ensuring continuity between chunks.

### 5. **Embedding Creation and Vector Store**

- **Embedding Models**:
   ```python
   embeddings = AzureOpenAIEmbeddings()  # or OpenAIEmbeddings()
   ```
   - Embedding models like `AzureOpenAIEmbeddings` or `OpenAIEmbeddings` convert each text chunk into high-dimensional vectors. These embeddings capture semantic meaning, allowing the system to retrieve relevant chunks when a question is asked.
   - The embeddings are stored in a vector database (Chroma) for efficient similarity search during the Q&A process.

   **Theoretical Perspective**:
   - Embeddings transform text into vector space, where semantically similar texts are close to each other. This is key in document retrieval, as it allows the system to find the most relevant parts of the document based on the user’s query.

### 6. **Retrieval-Based Q&A Chain**

- **RetrievalQAWithSourcesChain**:
   ```python
   chain = RetrievalQAWithSourcesChain.from_chain_type(
       llm=llm,
       chain_type="stuff",
       retriever=db.as_retriever(),
       return_source_documents=True
   )
   ```
   - This chain links the vector store (Chroma) with the language model (either OpenAI or Azure OpenAI). The retriever fetches the most relevant document chunks based on the query, and the model generates a response using the retrieved context.
   - The chain also ensures that the sources of the answer are returned to the user, which helps with transparency and validation.

   **Theoretical Perspective**:
   - The retrieval-based approach combines the power of embeddings and language models. Instead of forcing the model to process an entire document (which may be too large), relevant sections are retrieved first, making the process more efficient and scalable. Providing sources increases user trust and the system’s credibility.

### 7. **Answer Generation and User Interaction**

- **`on_message()`**:
   ```python
   response = await chain.acall(message.content, callbacks=[cb])
   ```
   - When the user asks a question, this function calls the retrieval chain asynchronously. The model retrieves relevant document chunks, generates an answer, and appends the sources that were used to answer the question.
   - The message is then displayed back to the user, providing both the answer and the document references.

   **Theoretical Perspective**:
   - This interaction showcases a powerful aspect of human-computer interaction: query understanding and source-based response. By grounding answers in the source material, the chatbot enhances user confidence in the results, particularly for legal, educational, or academic use cases.

### 8. **Logging and Debugging**

- **Logger Configuration**:
   ```python
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   ```
   - A logger is set up to track events like file uploads, text extraction, and the answering process. This helps monitor the flow of data through the system and diagnose any potential issues.
   
   **Theoretical Perspective**:
   - Logging is an essential part of maintaining robust and debuggable software. Especially in systems that involve asynchronous operations and API calls, logging ensures that any faults or delays can be easily traced and resolved.

## Summary of Function Calls

1. **Initialization**: Environment variables are loaded using `load_dotenv()` and other initial parameters like logging and configuration are set up.
2. **`on_chat_start()`**: Handles file uploads and reads the content of the uploaded `.pdf` or `.docx` files. It extracts text and prepares it for processing.
3. **Text Processing**: Using `RecursiveCharacterTextSplitter`, the text is split into chunks, and embeddings are generated with `AzureOpenAIEmbeddings` or `OpenAIEmbeddings`.
4. **Vector Store Creation**: The text chunks and their embeddings are stored in Chroma, which acts as a retriever for relevant document pieces.
5. **`on_message()`**: Handles user queries, retrieves relevant text from the document, and generates an answer with sources using `RetrievalQAWithSourcesChain`.

This flow ensures that documents are processed efficiently, and users receive accurate, context-aware answers with references to the original sources.
