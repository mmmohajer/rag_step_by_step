from services.helpers import BOLD, RESET, done, step
from services.rag import Rag


rag = Rag()

print(f"\n{BOLD}Building the knowledge base{RESET}", flush=True)

step(1, "Gathering sources")
data = rag.gather_info()
done(f"{len(data)} documents collected")

step(2, "Splitting text into chunks")
chunks = rag.split(data)
done(f"{len(chunks)} chunks built")

step(3, "Embedding chunks")
embedded = rag.embed_chunks(chunks)
done(f"{len(embedded)} embeddings created")

step(4, "Storing embeddings")
rag.store(embedded)
done(f"{len(rag.stored)} chunks stored")

print(f"\n{BOLD}Ready.{RESET} Ask a question, or type {BOLD}exit(){RESET} to quit.\n")

while True:
    question = input(f"{BOLD}You{RESET}  ").strip()
    if question == "exit()":
        print("\nGoodbye.\n")
        break
    if not question:
        continue

    step(1, "Embedding your question")
    query_embedding = rag.embed_query(question)
    done("question embedded")

    step(2, "Finding similar chunks")
    related_chunks = rag.find_similar(query_embedding)
    done(f"{len(related_chunks)} chunks matched")

    step(3, "Building the prompt")
    messages = rag.build_messages(question, related_chunks)
    done("prompt ready")

    step(4, "Generating the answer")
    response = rag.generate_response(messages)
    print(f"\n{BOLD}Answer{RESET}\n{response}\n")
