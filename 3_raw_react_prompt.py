from dotenv import load_dotenv
from langsmith import traceable
import ollama
import re
import inspect

MAX_ITERATION=10
MODEL="qwen3:1.7b"

# --- Tools (LangChain @ tool decorator) ---
@traceable(run_type="tool")
def get_product_price(product:str) -> float:
  """
  Look up the price of a product in the catelog.
  """
  print(f"   >> Executing get_product_price(product='{product}')")
  prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
  return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
  """
  Apply a discount tier to a price and return the final price.
  Available tiers: bronze, silver, gold.
  """
  print(f"   >> Executing get_product_price(prize='{price}', discount_tier='{discount_tier}')")
  discount_percentage = {"bronze": 5, "silver": 12, "gold": 23}
  discount = discount_percentage.get(discount_tier, 0)
  return round(price * (1 - discount / 100), 2)

tools = {
  "get_product_price": get_product_price,
  "apply_discount": apply_discount,
}

def get_tool_descriptions(tools_dicts):
  descriptions = []
  for tool_name, tool_function in tools_dicts.items():
    original_function = getattr(tool_function, "__wrapped__", tool_function)
    signature = inspect.signature(original_function)
    docstring = inspect.getdoc(tool_function) or ""
    descriptions.append(f"{tool_name}{signature} - {docstring}")
  return "\n".join(descriptions)

get_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
  return ollama.chat(model=model, messages=messages, options=options)
# --- Agent Loop ---
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
  tools_dict = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
  }

  print(f"Question: {question}")
  print("="*60)

  # Implement the system message that we are going to sending it to the llm.
  messages = [
        {
          "role":"system",
          "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."


                """
                Answer the following questions as best you can. You have access to the following tools:
                {tool_descriptions}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Begin!

                Question: {{question}}
                """
            )
        },
        {"role": "user", "content": question},
    ]
  
  # thought step
  for iteration in range(1, MAX_ITERATION + 1):
    print(f"\n--- Iteration {iteration} ---")

    # Difference
    response = ollama_chat_traced(messages=messages)
    ai_message = response.message

    tool_calls = ai_message.tool_calls
    # If no tool calls, this is the final answer
    if not tool_calls:
      print(f"\nFinal Answer: {ai_message.content}")
      return ai_message.content
    # Process only the FIRST tool call — force one tool per iteration
    tool_call = tool_calls[0]
    # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
    tool_name = tool_call.function.name
    tool_args = tool_call.function.arguments

    print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

    tool_to_use = tools_dict.get(tool_name)
    if tool_to_use is None:
        raise ValueError(f"Tool '{tool_name}' not found")

    # Difference 7: Direct function call instead of tool.invoke()
    observation = tool_to_use(**tool_args)


    print(f"  [Tool Result] {observation}")

    messages.append(ai_message)
    messages.append(
        {
            "role": "tool",
            "content": str(observation),
        }
    )

  print("ERROR: Max iterations reached without a final answer")
  return None

def main():
  print("Hello LangChian Agent (.bind_tools)!")
  print()
  result=run_agent("What is the price of a laptop after applying a gold discount?")

if __name__ == "__main__":
  main()