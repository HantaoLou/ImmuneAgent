from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.documents import Document


def _html_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def load_url_docs(url: str) -> list[Document]:
    """ """
    loader = RecursiveUrlLoader(url, extractor=_html_extractor)
    docs = loader.load()
    return docs


if __name__ == "__main__":
    url = "https://foldxsuite.crg.eu/products"
    docs = load_url_docs(url)
    print([d.metadata for d in docs])
    from kb.cli.filter import filter_by_entropy
    from kb.config.config import get_text_splitter

    s = get_text_splitter()
    splitted = s.split_documents(docs)
    doc_and_entropies = filter_by_entropy(splitted, 0)
    for doc, entropy in doc_and_entropies:
        print(f"{entropy} - {doc.page_content[:300]}")
