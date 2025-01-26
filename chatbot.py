from flask import Flask, request, jsonify, render_template_string
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# HTML template for the upload form
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>PDF Chat Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { margin-top: 20px; }
        .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .form-group { margin-bottom: 15px; }
        input[type="file"] { margin-bottom: 10px; }
        button { padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        #response { white-space: pre-wrap; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <h1>PDF Chat Interface</h1>

    <div class="section">
        <h2>1. Upload PDF</h2>
        <form id="uploadForm">
            <div class="form-group">
                <input type="file" id="pdfFile" accept=".pdf" required>
            </div>
            <button type="submit">Upload PDF</button>
        </form>
        <div id="uploadStatus"></div>
    </div>

    <div class="section">
        <h2>2. Chat with PDF</h2>
        <div class="form-group">
            <input type="text" id="messageInput" style="width: 80%; padding: 8px;" placeholder="Type your message here...">
            <button onclick="sendMessage()">Send</button>
        </div>
        <div id="chatResponse"></div>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = async function(e) {
            e.preventDefault();
            const file = document.getElementById('pdfFile').files[0];
            const formData = new FormData();
            formData.append('file', file);

            document.getElementById('uploadStatus').innerHTML = 'Uploading...';

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                if (response.ok) {
                    document.getElementById('uploadStatus').className = 'success';
                    document.getElementById('uploadStatus').innerHTML = result.message;
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                document.getElementById('uploadStatus').className = 'error';
                document.getElementById('uploadStatus').innerHTML = 'Error: ' + error.message;
            }
        };

        async function sendMessage() {
            const message = document.getElementById('messageInput').value;
            if (!message) return;

            const chatResponse = document.getElementById('chatResponse');
            chatResponse.innerHTML = 'Processing...';

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message: message })
                });
                const result = await response.json();

                if (response.ok) {
                    chatResponse.innerHTML = '<strong>Response:</strong><br>' + result.response +
                        '<br><br><strong>Sources:</strong><br>' + result.sources.join('<br><br>');
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                chatResponse.className = 'error';
                chatResponse.innerHTML = 'Error: ' + error.message;
            }

            document.getElementById('messageInput').value = '';
        }
    </script>
</body>
</html>
'''
# Set your OpenAI API key
# OPENAI_API_KEY = "your api key"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize global variables
conversation_chain = None
retriever = None
chat_history = []

# Configure upload folder
UPLOAD_FOLDER = '/home/duaaali/mysite/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def initialize_bot(pdf_path):
    global conversation_chain, retriever
    logger.debug(f"Initializing bot with PDF: {pdf_path}")

    try:
        # Load and split the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.debug(f"Loaded {len(documents)} pages from PDF")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        logger.debug(f"Split documents into {len(splits)} chunks")

        # Create embeddings and vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()
        logger.debug("Created vectorstore and retriever")

        # Create LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        logger.debug("Created LLM")

        # Create the conversation chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on the provided context. if the question is not present in the document please say I can't answer this based on the provided document."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])

        def combine_documents(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        conversation_chain = (
            {
                "context": retriever | combine_documents,
                "chat_history": lambda x: [
                    HumanMessage(content=q) if i % 2 == 0 else AIMessage(content=a)
                    for i, (q, a) in enumerate(chat_history)
                ],
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        logger.debug("Created conversation chain successfully")

        # Test if conversation_chain was properly set
        if conversation_chain is None:
            raise ValueError("conversation_chain is None after initialization")

        return True

    except Exception as e:
        logger.error(f"Error initializing bot: {str(e)}", exc_info=True)
        raise

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_pdf():
    logger.debug("Received upload request")

    if 'file' not in request.files:
        logger.error("No file in request")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("Empty filename")
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.endswith('.pdf'):
        try:
            # Save PDF in the uploads directory
            pdf_path = os.path.join(UPLOAD_FOLDER, "temp.pdf")
            file.save(pdf_path)
            logger.debug(f"Saved PDF to {pdf_path}")

            # Initialize the chatbot
            success = initialize_bot(pdf_path)

            if success:
                logger.debug("Bot initialized successfully")
                return jsonify({'message': 'PDF uploaded and processed successfully'}), 200
            else:
                logger.error("Bot initialization failed")
                return jsonify({'error': 'Failed to initialize chatbot'}), 500

        except Exception as e:
            logger.error(f"Error during upload: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    else:
        logger.error("Invalid file type")
        return jsonify({'error': 'Please upload a PDF file'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    logger.debug("Received chat request")
    global conversation_chain, chat_history

    # Debug print to check conversation_chain state
    logger.debug(f"conversation_chain state: {conversation_chain is not None}")

    if conversation_chain is None:
        logger.error("No conversation chain available")
        return jsonify({'error': 'Please upload a PDF first'}), 400

    data = request.json
    if not data or 'message' not in data:
        logger.error("No message in request")
        return jsonify({'error': 'No message provided'}), 400

    try:
        question = data['message']
        logger.debug(f"Processing question: {question}")

        result = conversation_chain.invoke(question)
        logger.debug("Got response from conversation chain")

        # Get the sources
        sources = retriever.get_relevant_documents(question)
        logger.debug(f"Retrieved {len(sources)} relevant documents")

        # Update chat history
        chat_history.append((question, result.content))

        return jsonify({
            'response': result.content,
            'sources': [doc.page_content for doc in sources]
        }), 200

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# This is required for PythonAnywhere WSGI
application = app
