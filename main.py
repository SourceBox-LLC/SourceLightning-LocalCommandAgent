import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_community.tools import ShellTool
from tempfile import TemporaryDirectory
from langchain_community.agent_toolkits import FileManagementToolkit

# Function to save API key to .env file
def save_api_key_to_env(key_name, api_key):
    with open(".env", "w") as env_file:
        env_file.write(f"{key_name}={api_key}\n")
    print(f"{key_name} saved to .env file.")

# Function to get and verify API key
def get_api_key(key_name):
    # Check if API key is present in environment variables
    api_key = os.getenv(key_name)
    if not api_key:
        # Prompt the user to enter the API key if not found
        api_key = input(f"Enter your {key_name}: ")
        save_api_key_to_env(key_name, api_key)
        # Reload the environment with the new API key
        load_dotenv()
        api_key = os.getenv(key_name)
    return api_key


# Load environment variables from the .env file
load_dotenv()

# Get the environment variables for the API key
anthropic_api_key = get_api_key("ANTHROPIC_API_KEY")

# We'll make a temporary directory to avoid clutter
working_directory = TemporaryDirectory()

# Create a File Management Toolkit
toolkit = FileManagementToolkit(
    root_dir=str(working_directory.name)
)  # If you don't provide a root_dir, operations will default to the current working directory

# Get the tools from the toolkit
file_management_tools = toolkit.get_tools()

# Initialize other tools
shell_tool = ShellTool()

# Create the agent
memory = MemorySaver()
model = ChatAnthropic(model_name="claude-3-sonnet-20240229", api_key=anthropic_api_key)

# Combine tools (shell tool + file management tools)
tools = [shell_tool] + file_management_tools  # Add all tools to the agent

agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Configuration for the agent
config = {"configurable": {"thread_id": "abc123"}}

# Use the agent in a loop
while True:
    prompt = input("Enter a prompt: ")

    # Use the agent to handle the user prompt
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=prompt)]}, config
    ):
        print(chunk)
        print("----")
