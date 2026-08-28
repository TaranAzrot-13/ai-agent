import os
import datetime
from dotenv import load_dotenv
from groq import Groq
from ddgs import DDGS
import json

load_dotenv()  # load api key

client = Groq()  # initialize client

memory = [
    {
        "role": "system",
        "content": (
            "You are JARVIS, a highly intelligent and slightly sarcastic AI assistant. "
            "When using the search_web tool, you MUST use short, concise keywords "
            "(e.g., 'F1 latest race winner') rather than long conversational sentences."
        ),
    }
]

print("Waking up...")


def get_current_time():  # function to get the current time and date
    return datetime.datetime.now().strftime("%I:%M %p on %B %d, %Y")


def search_web(query):
    try:
        # Initialize DDGS and strictly use keyword arguments for the text method
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords=query, max_results=3))

        if not results:
            return "No search results found."

        # Format the top 3 results for JARVIS
        formatted = ""
        for r in results:
            formatted += f"Title: {r.get('title', 'No Title')}\nSnippet: {r.get('body', 'No Snippet')}\n\n"
        return formatted

    except Exception as e:
        return f"Search error: {str(e)}"


jarvis_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current real-world time and date.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the live web for real-time news, current events, facts, and live data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search terms to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

while True:
    user_text = input("\nYou: ")

    if user_text.lower() == "exit":
        print("Exiting, have a nice day!")
        break

    memory.append({"role": "user", "content": user_text})

    # Inner recursive loop: lets JARVIS chain as many tools as he needs per user query
    while True:
        chat_completion = client.chat.completions.create(
            messages=memory,
            model="qwen/qwen3.8-27b",
            tools=jarvis_tools,
        )

        ai_reply = chat_completion.choices[0].message

        # If no tool calls requested, print response and break the inner loop
        if not ai_reply.tool_calls:
            final_output = ai_reply.content
            print(f"\nJARVIS: {final_output}")
            memory.append({"role": "assistant", "content": final_output})
            break

        # If a tool call is requested, execute it
        tool_call = ai_reply.tool_calls[0]
        print(
            f"\n[SYSTEM]: JARVIS requested to use a tool --> {tool_call.function.name}()"
        )

        memory.append(ai_reply)

        if tool_call.function.name == "get_current_time":
            tool_result = get_current_time()
        elif tool_call.function.name == "search_web":
            args = json.loads(tool_call.function.arguments)
            search_query = args.get("query")
            print(f"[SYSTEM]: Searching the live web for --> '{search_query}'")
            tool_result = search_web(search_query)

        memory.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": tool_result,
            }
        )

# print("Active models for this account: \n")  SCRIPT TO VERIFY THE AVAILABLE MODELS
# models = client.models.list()

# for model in models.data:
#     print(f"- {model.id}")
