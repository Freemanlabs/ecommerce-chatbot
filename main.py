import os
import uuid

from langgraph.checkpoint.memory import MemorySaver

from graph import build_graph

os.system("clear")


if __name__ == "__main__":
    workflow = build_graph()
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    # Specify a thread
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    def stream_graph_updates(user_input: str):
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config):
            for key, value in event.items():
                if (key != "rewriter") and len(value["messages"][-1].content) > 0:
                    if key == "tools" and value["messages"][-1].name == "return_policy":
                        continue
                    print("Agent:", value["messages"][-1].content)
                pass

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
