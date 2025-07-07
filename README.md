# Rag-papers

-> Using Reciprocal rank fusion to answer questions related to 5 research papers that are loaded in as vector stores

Ok, so all of our research papers will be stored in the research_papers directory, this includes:
*        "Attention Is All You Need",
*        "BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf"
*        "GPT-3 Language Models are Few-Shot Learners.pdf"
*        "Contrastive Language-Image Pretraining with Knowledge Graphs.pdf"
*        "LLaMA Open and Efficient Foundation Language Models.pdf"

We'll extract the text from these research papers, split them into chunks and then store them as vector store embeddings to access later

we'll also be using langchain for the analysis our rag model [to look into the different document chunks and the differnet ways our question gets modified]

Embeddings model: Hugging face

Chat model: gemini

# Reciprocal Rank Fusion
For generating the answers to the queries we'll basically use reciprocal rank fusion method, which is basically a more advanced version of multi query where the intial query sent by the user is rephrased and split into 5 similar questions by the language model which are then used to retrieve similar looking document chunks from the vector store

Heres where reciprocal rank fusion diverges from simple multi query method, the retrieved chunks are then ranked based on the RRF formula, higher rrf score meaning more relevant and then again for the final time we pass in the top ranked context chunks to the model with the original question and get the answer

![image](https://github.com/user-attachments/assets/a73e9c7a-5fae-4b30-aa4c-6bda1e120a9b)

where r is the rank and k is a constant(typically 60)

# Usage
We'll be using simple streamlit framework to display the website and get user input

![image](https://github.com/user-attachments/assets/f3a72a77-29c9-458b-b221-b2503167c1ec)

we can ask any question relating to the research papers and it will apply rag to retrieve the information needed

![image](https://github.com/user-attachments/assets/71a4bfc1-abfc-4b98-95c1-eab24ad22319)

# Analysis
For analysis we'll be using langsmith, it basically gives an entire overlook on the different questions the origianl question is rephrased to and also the final answer and the document chunks it gets it from

![image](https://github.com/user-attachments/assets/af570ecb-f8ab-4cc9-a154-89d609ed6e07)

Splitting into 6 similar questions:
![image](https://github.com/user-attachments/assets/16b7c3d8-4b76-4a5d-9953-f959f311d07e)

The chunks retrieved from one of those questions:
![image](https://github.com/user-attachments/assets/05bbbbb3-179f-4261-837a-701fd762fabe)

Applying RRF:
![image](https://github.com/user-attachments/assets/4d88d830-d2ca-46bb-b9a9-98cfd4c3f165)

Final Output:
![image](https://github.com/user-attachments/assets/9f9eea47-a030-40d8-ba87-c95da055d358)





