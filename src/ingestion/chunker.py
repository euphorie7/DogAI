def chunk_documents(
    documents : list[str],
    chunk_size : int = 500,
    overlap: int = 100
) -> list[str]:

    if chunk_size <=0 or overlap <=0 or overlap>=chunk_size :
        raise ValueError("PArametres de chunking invalides !")

    chunks = []

    for document in documents : 

        start = 0 

        while start < len(documents):
            end = start + chunk_size

            chunk = document[start:end]
            chunks.append(chunk)

            start+= chunk_size - overlap
    
    return chunks