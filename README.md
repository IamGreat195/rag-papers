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
