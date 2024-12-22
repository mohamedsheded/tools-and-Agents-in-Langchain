# README: Implementing Custom Tools, Built-in Tools, and Tool Calling with LLMs

## Table of Contents
1. [Custom Tool: Retriever Agent](#custom-tool-retriever-agent)
2. [Using Built-in Tools: DuckDuckGo](#using-built-in-tools-duckduckgo)
3. [Tool Calling](#tool-calling)
4. [Passing Tools with Arguments to LLM](#passing-tools-with-arguments-to-llm)
5. [Creating OpenAI Tools Agent and Agent Executor](#creating-openai-tools-agent-and-agent-executor)

---

## 1. Custom Tool: Retriever Agent

### Overview:
A retriever agent is designed to fetch relevant information from a knowledge base or document store based on user queries. This example uses FAISS for vector storage and HuggingFace for embeddings.

### Required Packages:
```bash
pip install langchain faiss-cpu transformers openai
```

### Example Implementation:
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Load embeddings and initialize retriever
def create_retriever_agent():
    # Load HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings()
    # Load FAISS vector database
    db = FAISS.load_local("./vectorstore", embeddings)
    # Set up retriever
    retriever = db.as_retriever()
    return retriever

# Create a QA chain with retriever
def retriever_agent(query):
    retriever = create_retriever_agent()
    llm = OpenAI()
    # Define QA chain
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain.run(query)

# Example Query
response = retriever_agent("What is generative AI?")
print(response)
```

---

## 2. Using Built-in Tools: DuckDuckGo

### Overview:
DuckDuckGo can be used as a search tool to retrieve real-time information from the web.

### Required Packages:
```bash
pip install langchain duckduckgo-search
```

### Example Implementation:
```python
from langchain.tools import DuckDuckGoSearchRun

# Initialize DuckDuckGo Search Tool
def duckduckgo_search_tool(query):
    # Create search instance
    search = DuckDuckGoSearchRun()
    # Run search query
    results = search.run(query)
    return results

# Example Query
response = duckduckgo_search_tool("latest AI trends 2024")
print(response)
```

---

## 3. Tool Calling

### Overview:
Tool calling allows the LLM to decide which tool to invoke based on the query provided. Tools are wrapped and described for easier integration with agents.

### Required Packages:
```bash
pip install langchain openai
```

### Example Implementation:
```python
from langchain.tools import Tool
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# Define Tools
retriever_tool = Tool(
    name="RetrieverAgent",
    func=retriever_agent,
    description="Fetches answers from a document store."
)

duckduckgo_tool = Tool(
    name="DuckDuckGoSearch",
    func=duckduckgo_search_tool,
    description="Searches the web for real-time information."
)

# Initialize LLM and Agent
llm = OpenAI(temperature=0)
tools = [retriever_tool, duckduckgo_tool]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Example Query
response = agent.run("Search the web for AI trends or retrieve stored information about generative AI.")
print(response)
```

---

## 4. Passing Tools with Arguments to LLM

### Overview:
Pass tools dynamically to the LLM by defining their arguments and function calls.

### Required Packages:
```bash
pip install langchain openai
```

### Example Implementation:
```python
import json

# Tool Definitions
def run_tool_with_args(tool_name, query):
    # Define available tools
    tools = {
        "retriever_agent": retriever_agent,
        "duckduckgo_search_tool": duckduckgo_search_tool
    }
    
    # Check if the tool exists and execute
    if tool_name in tools:
        result = tools[tool_name](query)
        return result
    else:
        return f"Tool {tool_name} not found."

# Example Usage
response = run_tool_with_args("duckduckgo_search_tool", "Latest AI news")
print(response)
```

---

## 5. Creating OpenAI Tools Agent and Agent Executor

### Overview:
This section demonstrates how to create an OpenAI tools agent with an agent executor and load prompts from `langchain.hub`.

### Required Packages:
```bash
pip install langchain openai
```

### Example Implementation:
```python
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import load_prompt
from langchain.chat_models import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Define tools
tools = [retriever_tool, duckduckgo_tool]

# Load a prompt template from langchain.hub
prompt = load_prompt("lc://prompts/agent-chat/prompt.json")

# Create OpenAI Tools Agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create Agent Executor
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example Query
response = executor.invoke("Find the latest AI trends and summarize generative AI.")
print(response)
```

---

## Conclusion
This README provides code examples and explanations for creating custom tools like a retriever agent, leveraging built-in tools such as DuckDuckGo, implementing tool calling, passing tools with arguments to an LLM, and building an OpenAI tools agent with an executor and custom prompts. Use these examples as a foundation to build and scale intelligent agents with advanced capabilities.

