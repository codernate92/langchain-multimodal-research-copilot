#init.py

from utils import get_qa_chain, pretty_print_answer

def main():
    qa = get_qa_chain()
    print("Multimodal Copiliot(CLI Mode)")
    print("Type 'exit' to quit.")
    while True:
        query = input("❓ Ask your question: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        result = qa({"query": query})
        pretty_print_answer(result)
        print("-" * 40)

if __name__ == "__main__":
    main()   #Call the main function to start the CLI