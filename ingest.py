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
    PineconeVectorStore.from_documents(docs, embeddings, index_name=INDEX_NAME)
    print(f"Loaded {len(docs)} documents to Pinecone.")


if __name__ == "__main__":
    file_path = "resources/movies_100_only.csv"
    ingest(file_path)
