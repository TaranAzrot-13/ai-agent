import os
import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # load api key

client = Groq()  # initialize client

memory = [
    {"role": "system", "content": "You are JARVIS, a highly intelligent and slightly sarcastic AI assistant."}
]

print("Waking up...")


def get_current_time():  # function to get the current time and date
    return datetime.datetime.now().strftime("%I:%M %p on %B %d, %Y")


jarvis_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current real-world time and date.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

while True:
    user_text = input("\nYou: ")  # user input

    if user_text.lower() == "exit":  # exit condition
        print("Exiting, have a nice day!")
        break

    # add user input to model memory
    memory.append({"role": "user", "content": user_text})

    chat_completion = client.chat.completions.create(

        messages=memory,
        model="qwen/qwen3.8-27b",  # model used
        tools=jarvis_tools,  # give the model access to the tools
        tool_choice="auto",  # allow the model to choose which tool to use
    )

    ai_reply = chat_completion.choices[0].message

    if ai_reply.tool_calls:  # check if the model requested to use a tool
        tool_call = ai_reply.tool_calls[0]
        print(
            f"\n[SYSTEM]: JARVIS requested to use a tool --> {tool_call.function.name}()")
        memory.append(ai_reply)  # save the tool call to memory

        # execute the tool function and get the result
        if tool_call.function.name == "get_current_time":
            time_result = get_current_time()
            print(
                f"\n[SYSTEM]: JARVIS executed the tool and got the result --> {time_result}")
            # add the tool result to memory
            memory.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": time_result
            })

            final_completion = client.chat.completions.create(
                messages=memory,
                model="qwen/qwen3.8-27b",
                tools=jarvis_tools,
                tool_choice="auto"
            )

            final_reply = final_completion.choices[0].message.content
            print("\n")
            print(f"\nJARVIS: {final_reply}")  # print out the response
            # save model response to memory
            memory.append({"role": "assistant", "content": final_reply})
    else:
        ai_reply = ai_reply.content
        print("\n")  # print a new line for better readability
        print(f"\nJARVIS: {ai_reply}")  # print out the response
        # save model response to memory
        memory.append({"role": "assistant", "content": ai_reply})

# print("Active models for this account: \n")  SCRIPT TO VERIFY THE AVAILABLE MODELS
# models = client.models.list()

# for model in models.data:
#     print(f"- {model.id}")
