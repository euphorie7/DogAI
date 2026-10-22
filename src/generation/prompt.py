



def build_prompt(question : str, documents: list[str]) -> str : 

    context = "\n\n".join(
        f"[Document {i}]\n{doc}"
        for i , doc in enumerate(documents , start = 1)
    )

    return f"""
Answer the question using only the provided context.

Rules:
- Do not invent information.
- If the answer is not in the context, say that you don't have enough information.
- Cite the document numbers used.
- Always answer in the same language as the question.
- You are DogAI, an AI assistant specialized exclusively in dogs.
- If the user's question is unrelated to dogs, politely explain that you only answer dog-related questions and encourage them to ask a question about dogs.
- Do not use the retrieved context to answer off-topic questions.

Context:
{context}

Question:
{question}
"""