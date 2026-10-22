from src.retrieval.retrieval import retrieve_documents
from src.generation.generation import generate_answer

def main():

    print("Welcome to DogAI!")

    while True:
        # 1. User question
        question = input("\nAsk a question (or 'exit' to quit): ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        # 2. Semantic retrieval
        results = retrieve_documents(
            question=question,
            top_k=2
        )

        # 3. Extract relevant documents
        documents = results["documents"][0]

        if not documents:
            print("No relevant documents found.")
            continue

        # 3. LLM Generation
        answer = generate_answer(question, documents)

        print(f"\nDogAI: {answer}")

        # # 3. Display retrieved documents
        # print("\n--- Retrieved Documents ---")

        # for index, (document, distance) in enumerate(
        #     zip(
        #         results["documents"][0],
        #         results["distances"][0]
        #     ),
        #     start=1
        # ):
        #     print(f"\nDocument {index}")
        #     print(f"Distance: {distance:.4f}")
        #     print(f"Content: {document}")


if __name__ == "__main__":
    main()