from langchain_core.documents import Document


PAGE_CONTENT = "page_content"
METADATA = "metadata"

def serialize_context(context : list[Document]) -> list[dict]:
    serialized = []
    for doc in context:
        doc_serialized = {}
        doc_serialized[PAGE_CONTENT] = doc.page_content
        doc_serialized[METADATA] = doc.metadata
        serialized.append(doc_serialized)
    return serialized


def deserialize_context(serialized : list[dict]) -> Document:
    documents = []
    for doc_serialized in serialized:
        if PAGE_CONTENT not in doc_serialized or METADATA not in doc_serialized:
            raise Exception("Invalid serialization, cannot form a document")
        documents.append(Document(doc_serialized[PAGE_CONTENT], doc_serialized[METADATA]))
    return documents