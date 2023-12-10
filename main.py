# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`
import datetime
from firebase_functions import https_fn
# from firebase_admin import storage
from firebase_admin import initialize_app
import tiktoken
# from werkzeug import secure_filename
import os

# import gradio as gr
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.tools import ElevenLabsText2SpeechTool

from langchain.document_loaders import PyPDFLoader
# Loaders
from langchain.schema import Document

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Model
from langchain.chat_models import ChatOpenAI

# Embedding Support
# from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain

# Data Science
import numpy as np
from sklearn.cluster import KMeans

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
# from langchain import PromptTemplate
import os
from flask import Flask, render_template, request
from tempfile import NamedTemporaryFile
from google.cloud import storage
initialize_app()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "serviceAccount2.json"
storage_client = storage.Client()

# storage_client = storage._StorageClient(credentials="serviceAccount2.json", 
#                                         project="tutorial-image-4ab83", default_bucket="tutorial-image-4ab83.appspot.com")

# storage_client = storage.Client(
#     credentials=os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
# )

@https_fn.on_request(secrets=["OPENAI_API_KEY", "ELEVEN_API_KEY"],memory= "1GB")
def upload_pdf_make_me_a_podcast(req: https_fn.Request) -> https_fn.Response:
    # Validate if file was sended in the request
    print('req', req)
    if 'file' not in request.files:
        return https_fn.Response({"pdf no pdf uploaded, you have to upload a pdf file in the formData "})
    uploaded_file = request.files['file']
    print('uploaded_file', uploaded_file)
    
    if uploaded_file:
        # Crear un archivo temporal
        temp_file = NamedTemporaryFile(delete=False)
        try:
            # Guardar el contenido del archivo en el archivo temporal
            temp_file.write(uploaded_file.read())
            
            # Obtener la ruta del archivo temporal
            temp_file_path = temp_file.name
            print(os.environ["OPENAI_API_KEY"])
            print(os.environ["ELEVEN_API_KEY"] )
            openai_api_key = os.environ["OPENAI_API_KEY"]
            def num_tokens_from_string(string: str, encoding_name: str) -> int:
                encoding = tiktoken.get_encoding(encoding_name)
                num_tokens = len(encoding.encode(string))
                return num_tokens
            llm = OpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"] )
            def sumarizeBook():
                # Load the book
                llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
                loader = PyPDFLoader(temp_file_path)
                pages = loader.load()

                # Cut out the open and closing parts
                # pages = pages[26:277]

                # Combine the pages, and replace the tabs with spaces
                text = ""

                for page in pages:
                    text += page.page_content
                    
                text = text.replace('\t', ' ')

                num_tokens = llm.get_num_tokens(text)
                if num_tokens > 4000:
                    text = text[:4000]

                print (f"This book has {num_tokens} tokens in it")
                text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)

                docs = text_splitter.create_documents([text])

                num_documents = len(docs)

                print (f"Now our book is split up into {num_documents} documents")

                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

                vectors = embeddings.embed_documents([x.page_content for x in docs])


                # Assuming 'embeddings' is a list or array of 1536-dimensional embeddings

                # Choose the number of clusters, this can be adjusted based on the book's content.
                # I played around and found ~10 was the best.
                # Usually if you have 10 passages from a book you can tell what it's about
                num_clusters = 1

                # Perform K-means clustering
                kmeans = KMeans(n_clusters=num_clusters, random_state=10).fit(vectors)

                kmeans.labels_

                # Find the closest embeddings to the centroids

                # Create an empty list that will hold your closest points
                closest_indices = []

                # Loop through the number of clusters you have
                for i in range(num_clusters):
                    
                    # Get the list of distances from that particular cluster center
                    distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
                    
                    # Find the list position of the closest one (using argmin to find the smallest distance)
                    closest_index = np.argmin(distances)
                    
                    # Append that position to your closest indices list
                    closest_indices.append(closest_index)

                selected_indices = sorted(closest_indices)
                selected_indices


                llm3 = ChatOpenAI(temperature=0,max_tokens=1000, openai_api_key=openai_api_key,    model='gpt-3.5-turbo'   )

                map_prompt = """
                You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
                Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
                Your response should be at least three paragraphs and fully encompass what was said in the passage.

                ```{text}```
                FULL SUMMARY:
                """
                map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

                map_chain = load_summarize_chain(llm=llm3,                             chain_type="stuff",       prompt=map_prompt_template)
                selected_docs = [docs[doc] for doc in selected_indices]

                # Make an empty list to hold your summaries
                summary_list = []

                # Loop through a range of the lenght of your selected docs
                for i, doc in enumerate(selected_docs):
                    
                    # Go get a summary of the chunk
                    chunk_summary = map_chain.run([doc])
                    
                    # Append that summary to your list
                    summary_list.append(chunk_summary)
                    
                    print (f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")


                summaries = "\n".join(summary_list)

                # Convert it back to a document
                summaries = Document(page_content=summaries)

                print (f"Your total summary has {llm.get_num_tokens(summaries.page_content)} tokens")

                llm4 = ChatOpenAI(temperature=0, max_tokens=3000,model='gpt-3.5-turbo', request_timeout=120                )
                
                combine_prompt = """
                You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
                Your goal is to give a verbose summary of what happened in the story.
                The reader should be able to grasp what happened in the book.

                ```{text}```
                VERBOSE SUMMARY:
                """
                combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

                reduce_chain = load_summarize_chain(llm=llm4, chain_type="stuff", prompt=combine_prompt_template,)

                output = reduce_chain.run([summaries])
                print (output)
                return output
            resumen = sumarizeBook()
            # print(resumen)
            # cut length resumen 
            # resumen = resumen[:1000]
            text_to_speak = resumen[0:400]
            # print(text_to_speak)
            tts = ElevenLabsText2SpeechTool()
            tts.name
            speech_file = tts.run(text_to_speak)
            print('speech_file', speech_file)
            bucket_name = 'tutorial-image-4ab83.appspot.com'

            # [START storageThumbnailGeneration]
            bucket = storage_client.bucket(bucket_name)
            # now =1702237661679
            now = datetime.datetime.now().timestamp()
            in_six_days = datetime.datetime.now() + datetime.timedelta(days=6)
            path_storage = f'audio_{now}.mp3'
            # upload audio speech_file to firestorage
            blob = bucket.blob(path_storage)
            blob.upload_from_filename(speech_file)
            signedUrls = blob.generate_signed_url( expiration=in_six_days,
            api_access_endpoint="https://storage.googleapis.com",
            method="GET",
            content_md5=None,
            content_type=None,
            response_disposition=None,
            response_type=None,
            generation=None,
            headers=None,
            query_parameters=None,
            client=None,
            credentials=None,
            version=None,
            service_account_email=None,
            access_token=None,
            virtual_hosted_style=False,
            bucket_bound_hostname=None,
            scheme="http");
            print('signedUrls', signedUrls)
            # delete from temporal folder
            if os.path.exists(speech_file):
                os.remove(speech_file)
            return https_fn.Response({ signedUrls: signedUrls})
        finally:
            print('temp_file', temp_file)
            # Cerrar y eliminar el archivo temporal al finalizar
            temp_file.close()
            os.unlink(temp_file.name)
    return https_fn.Response({"error with uploaded PDF"})


