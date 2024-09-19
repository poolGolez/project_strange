from langchain_community.document_loaders import CSVLoader


def ingest(file_path: str):
    loader = CSVLoader(
        file_path,
        source_column="id",
        metadata_columns=["id", "title"],
        content_columns=["overview"]
    )
    docs = loader.load()

    print("Done.")


if __name__ == "__main__":
    ingest("resources/movies_100_only.csv")
