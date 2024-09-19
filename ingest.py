from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = "top-movies-gemini-embedding-004"
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def ingest(file_path: str):
    loader = CSVLoader(
        file_path,
        source_column="id",
        metadata_columns=["id", "title"],
        content_columns=["overview"]
    )
    docs = loader.load()
    print(f"Loading {len(docs)} documents to Pinecone...")

    chunk_size = 128
    for i in range(0, len(docs), chunk_size):
        batch = docs[i: i + chunk_size]
        print(f"Loading {len(batch)} ({i + 1}-{i + len(batch)}) documents")
        PineconeVectorStore.from_documents(batch, embeddings, index_name=INDEX_NAME)

    print(f"Loaded a total of {len(docs)} documents to Pinecone.")


if __name__ == "__main__":
    file_path = "resources/movies_complete.csv"
    ingest(file_path)
