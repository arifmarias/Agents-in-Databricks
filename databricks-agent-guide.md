# Building and Evaluating Agents in Databricks: A Comprehensive End-to-End Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Architecture Overview](#architecture-overview)
4. [Part 0: Data Preparation](#part-0-data-preparation)
5. [Part 1: Building Your First Agent](#part-1-building-your-first-agent)
6. [Part 2: Agent Evaluation and Deployment](#part-2-agent-evaluation-and-deployment)
7. [Best Practices and Tips](#best-practices-and-tips)

## Introduction

This comprehensive guide will walk you through building and evaluating AI agents in Databricks. We'll create a customer service agent that can handle product returns, answer questions, and make intelligent decisions using various tools and LLMs (Large Language Models).

### What You'll Learn
- How to prepare data and create Unity Catalog tables
- Building SQL and Python functions as agent tools
- Creating and testing agents in AI Playground
- Implementing agents with LangGraph and MLflow
- Evaluating agent performance with custom scorers
- Deploying agents to production endpoints

### Lab Structure
The hands-on lab consists of three main notebooks:
1. **Data Preparation** - Setting up tables and vector search indexes
2. **Tool Preparation** - Creating SQL and Python functions
3. **Agent Building and Evaluation** - Implementing and testing the agent

## Prerequisites

### Required Components
- Databricks workspace with Unity Catalog enabled
- Access to create catalogs and schemas
- Serverless or general-purpose compute cluster
- Access to AI Playground
- Claude 3.7 Sonnet or similar LLM endpoint

### Required Python Packages
```python
# Core packages
mlflow-skinny[databricks]
langgraph==0.3.4
databricks-langchain
databricks-agents
uv
```

## Architecture Overview

The agent system consists of:
1. **Unity Catalog** - Stores data tables, functions, and models
2. **Vector Search Index** - Enables semantic search for product documentation
3. **SQL Functions** - Query customer service data
4. **Python Functions** - Handle operations LLMs can't perform directly
5. **LLM Endpoint** - Processes natural language and orchestrates tools
6. **MLflow** - Tracks experiments and manages model lifecycle

## Part 0: Data Preparation

This section prepares all necessary data and infrastructure for the agent.

### Step 1: Set Up Configuration

Create a `config.py` file to manage catalog and schema names:

```python
# config.py
catalog_name = "your_catalog_name"  # Replace with your catalog
system_schema_name = "agents_lab"   # Shared schema for data
```

### Step 2: Create Catalog and Schema

```python
%run ./config

# Display configuration
print(f"Catalog for hands-on: {catalog_name}")
print(f"Shared schema for data storage: {system_schema_name}")

# Create catalog
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"GRANT USE CATALOG ON CATALOG {catalog_name} TO `account users`")

# Create schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{system_schema_name}")
spark.sql(f"GRANT USE SCHEMA, SELECT ON SCHEMA {catalog_name}.{system_schema_name} TO `account users`")
```

### Step 3: Load Sample Data

The sample data includes:
- **cust_service_data.csv** - Customer service interactions
- **policies.csv** - Company return policies
- **product_docs.csv** - Product documentation for RAG

```python
import os

# Load CSV files from workspace files
cust_service_data_df = spark.read.format("csv").load(
    f"file:{os.getcwd()}/data/cust_service_data.csv", 
    header=True, 
    inferSchema=True
)

policies_df = spark.read.format("csv").load(
    f"file:{os.getcwd()}/data/policies.csv", 
    header=True, 
    inferSchema=True
)

product_docs_df = spark.read.format("csv").load(
    f"file:{os.getcwd()}/data/product_docs.csv", 
    header=True, 
    inferSchema=True, 
    multiline='true'
)

# Display data to verify
display(cust_service_data_df)
display(policies_df)
display(product_docs_df)
```

### Step 4: Save Data to Unity Catalog Tables

```python
# Save as Unity Catalog tables
cust_service_data_df.write.mode("overwrite").saveAsTable(
    f"{catalog_name}.{system_schema_name}.cust_service_data"
)

policies_df.write.mode("overwrite").saveAsTable(
    f"{catalog_name}.{system_schema_name}.policies"
)

product_docs_df.write.mode("overwrite").saveAsTable(
    f"{catalog_name}.{system_schema_name}.product_docs"
)
```

### Step 5: Create Vector Search Index

This index enables semantic search for product documentation:

```python
# Enable Change Data Feed for the source table
source_table = f"{catalog_name}.{system_schema_name}.product_docs"

spark.sql(f"""
    ALTER TABLE {source_table}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

print(f"✅ Enabled Change Data Feed for {source_table}")

# Verify the setting
spark.sql(f"SHOW TBLPROPERTIES {source_table}").filter(
    "key = 'delta.enableChangeDataFeed'"
).show()
```

```python
import requests
import json
import time

# Get Databricks context
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = context.apiUrl().get()
token = context.apiToken().get()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Parameters
endpoint_name = "one-env-shared-endpoint-1"  # Adjust as needed
index_name = f"{catalog_name}.{system_schema_name}.product_docs_index"
source_table = f"{catalog_name}.{system_schema_name}.product_docs"

# Create index
url = f"{host}/api/2.0/vector-search/indexes"
payload = {
    "name": index_name,
    "endpoint_name": endpoint_name,
    "primary_key": "product_id",
    "index_type": "DELTA_SYNC",
    "delta_sync_index_spec": {
        "source_table": source_table,
        "pipeline_type": "TRIGGERED",
        "embedding_source_columns": [{
            "name": "product_doc",
            "embedding_model_endpoint_name": "databricks-gte-large-en"
        }]
    }
}

print("Creating Vector Search Index...")
response = requests.post(url, headers=headers, json=payload)

if response.status_code in [200, 201]:
    print("✅ Index creation request sent successfully!")
    print("📊 Initial sync will start automatically...\n")
else:
    print(f"❌ Index creation error: {response.status_code}")
    print(response.text)
```

### Step 6: Monitor Index Creation

```python
def monitor_index_status(index_name, timeout_minutes=60):
    """Monitor the status of index creation"""
    status_url = f"{host}/api/2.0/vector-search/indexes/{index_name}"
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    print(f"⏳ Monitoring index status...")
    print(f"Timeout: {timeout_minutes} minutes")
    print("-" * 70)
    
    while time.time() - start_time < timeout_seconds:
        try:
            response = requests.get(status_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", {})
                state = status.get("detailed_state", "Unknown")
                ready = status.get("ready", False)
                indexed_count = status.get("indexed_row_count", 0)
                
                elapsed = int(time.time() - start_time)
                print(f"\r⏱️ {elapsed//60}m {elapsed%60}s | "
                      f"State: {state} | "
                      f"Rows: {indexed_count:,} | "
                      f"Ready: {ready}", end="")
                
                if ready and ("ONLINE" in state or state == "READY"):
                    print(f"\n\n✅ Index is ready!")
                    print(f"📊 Total indexed rows: {indexed_count:,}")
                    return True
                    
        except Exception as e:
            print(f"\n⚠️ Exception: {e}")
        
        time.sleep(30)  # Check every 30 seconds
    
    return False

# Start monitoring
success = monitor_index_status(index_name, timeout_minutes=60)
if success:
    print("\n🎉 Vector Search Index is ready!")
```

## Part 1: Building Your First Agent

### Step 1: Create User-Specific Schema

```python
from databricks.sdk import WorkspaceClient
import re

# Get current user information
w = WorkspaceClient()
user_email = w.current_user.me().emails[0].value
username = user_email.split('@')[0]
username = re.sub(r'[^a-zA-Z0-9_]', '_', username)

# Create user-specific schema
user_schema_name = f"agents_lab_{username}"
print("Your catalog:", catalog_name)
print("Your schema:", user_schema_name)

# Create widgets for SQL/Python reference
dbutils.widgets.text("catalog_name", defaultValue=catalog_name, label="Catalog Name")
dbutils.widgets.text("system_schema_name", defaultValue=system_schema_name, label="System Schema Name")
dbutils.widgets.text("user_schema_name", defaultValue=user_schema_name, label="User Schema Name")

# Create user schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{user_schema_name}")
```

### Step 2: Create SQL Functions for Customer Service Workflow

The customer service workflow includes:
1. Get the latest return from the queue
2. Retrieve company policies
3. Get user ID for the latest return
4. Query order history using user ID
5. Provide current date to the LLM

#### Function 1: Get Latest Return

```sql
CREATE OR REPLACE FUNCTION
    IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_latest_return')()
RETURNS TABLE(
    purchase_date DATE, 
    issue_category STRING, 
    issue_description STRING, 
    name STRING
)
COMMENT 'Returns the latest customer service interaction (such as returns)'
RETURN (
    SELECT
        CAST(date_time AS DATE) AS purchase_date,
        issue_category,
        issue_description,
        name
    FROM cust_service_data
    ORDER BY date_time DESC
    LIMIT 1
);
```

#### Function 2: Get Return Policy

```sql
CREATE OR REPLACE FUNCTION
    IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_return_policy')()
RETURNS TABLE (
    policy STRING,
    policy_details STRING,
    last_updated DATE
)
COMMENT 'Returns return policy details'
LANGUAGE SQL
RETURN (
    SELECT
        policy,
        policy_details,
        last_updated
    FROM policies
    WHERE policy = 'Return Policy'
    LIMIT 1
);
```

#### Function 3: Get User ID by Name

```sql
CREATE OR REPLACE FUNCTION
    IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_user_id')(user_name STRING)
RETURNS STRING
COMMENT 'Takes customer name as input and returns corresponding user ID'
LANGUAGE SQL
RETURN
    SELECT customer_id
    FROM cust_service_data
    WHERE name = user_name
    LIMIT 1
;
```

#### Function 4: Get Order History

```sql
CREATE OR REPLACE FUNCTION
    IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_order_history')(user_id STRING)
RETURNS TABLE (
    returns_last_12_months INT, 
    issue_category STRING
)
COMMENT 'Takes customer user_id and returns number of returns in last 12 months and issue categories'
LANGUAGE SQL
RETURN
    SELECT 
        count(*) as returns_last_12_months, 
        issue_category
    FROM cust_service_data
    WHERE customer_id = user_id
    GROUP BY issue_category;
```

### Step 3: Create Python Function for Current Date

```python
def get_todays_date() -> str:
    """
    Returns today's date in 'YYYY-MM-DD' format.
    
    Returns:
        str: Today's date in 'YYYY-MM-DD' format.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

# Test the function
today = get_todays_date()
print(f"Today's date: {today}")
```

### Step 4: Register Python Function in Unity Catalog

```python
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

# Deploy the tool to UC with automatic metadata setup
python_tool_uc_info = client.create_python_function(
    func=get_todays_date, 
    catalog=catalog_name, 
    schema=user_schema_name, 
    replace=True
)

# Display deployed function name
print(f"Deployed Unity Catalog function: {python_tool_uc_info.full_name}")
```

### Step 5: Test in AI Playground

1. Navigate to AI Playground in the Databricks workspace
2. Select Claude 3.7 Sonnet as the model
3. Set the system prompt:
   ```
   Call tools until you are confident all company policies are met
   ```
4. Add the created functions as tools
5. Test with the question:
   ```
   Based on our policies, should we accept the latest return in the queue?
   ```

## Part 2: Agent Evaluation and Deployment

### Step 1: Create Agent Implementation (agent.py)

Create a comprehensive agent that combines all tools:

```python
# agent.py
from typing import Any, Generator, Optional, Sequence, Union
import mlflow
import os
import json
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

############################################
# Get configuration from environment variables
############################################
def get_env_config():
    """Get minimal required configuration from environment variables"""
    llm_endpoint = os.environ.get("LLM_ENDPOINT_NAME")
    uc_tool_names = os.environ.get("UC_TOOL_NAMES", "")
    vs_name = os.environ.get("VS_NAME", "")
    
    # Validate required items
    if not llm_endpoint:
        raise ValueError("LLM_ENDPOINT_NAME environment variable is required")
    if not uc_tool_names:
        raise ValueError("UC_TOOL_NAMES environment variable is required")
    if not vs_name:
        raise ValueError("VS_NAME environment variable is required")
    
    # Split tool names
    tool_names = [name.strip() for name in uc_tool_names.split(",") if name.strip()]
    
    config = {
        "llm_endpoint": llm_endpoint,
        "uc_tool_names": tool_names,
        "vs_name": vs_name
    }
    
    return config

# Get configuration
config = get_env_config()
LLM_ENDPOINT_NAME = config['llm_endpoint']
UC_TOOL_NAMES = config['uc_tool_names']
VS_NAME = config['vs_name']

# Print configuration for verification
print("Agent Configuration:")
print(f"LLM_ENDPOINT_NAME: {LLM_ENDPOINT_NAME}")
print(f"UC_TOOL_NAMES: {UC_TOOL_NAMES}")
print(f"VS_NAME: {VS_NAME}")

# Enable MLflow/LangChain auto-logging
mlflow.langchain.autolog()

# Initialize Databricks Function Client
client = DatabricksFunctionClient(disable_notice=True, suppress_warnings=True)
set_uc_function_client(client)

############################################
# Create LLM instance
############################################
llm = ChatDatabricks(endpoint=config["llm_endpoint"])

# System prompt (controls agent behavior)
system_prompt = """You are a Databricks Lab customer success specialist. 
When users ask questions about products, retrieve necessary information using tools 
and help users fully understand the products. Include relevant information that 
might interest customers and strive to provide value in every interaction."""

############################################
# Create tools
############################################
def create_tools() -> List[BaseTool]:
    """Create tools based on environment variable configuration"""
    tools = []
    
    # Add Vector Search tool
    if VS_NAME:
        try:
            vs_tool = VectorSearchRetrieverTool(
                index_name=VS_NAME,
                tool_name="search_product_docs",
                num_results=3,
                tool_description="Use this tool to search product documentation."
            )
            tools.append(vs_tool)
            print(f"Added Vector Search tool: {VS_NAME}")
        except Exception as e:
            print(f"Warning: Could not load Vector Search tool {VS_NAME}: {e}")
    
    # Add UC function tools
    if UC_TOOL_NAMES:
        try:
            uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
            tools.extend(uc_toolkit.tools)
            print(f"Added UC function tools: {UC_TOOL_NAMES}")
        except Exception as e:
            print(f"Warning: Could not add UC tools {UC_TOOL_NAMES}: {e}")
    
    return tools

# Create tools
tools = create_tools()

############################################
# Define agent logic
############################################
def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    # Bind tools to model
    model = model.bind_tools(tools)
    
    # Define function to determine next node
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # Continue if there are function calls, otherwise end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"
    
    # Preprocessing to add system prompt
    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    
    model_runnable = preprocessor | model
    
    # Function for model invocation
    def call_model(state: ChatAgentState, config: RunnableConfig):
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}
    
    # Custom tool execution function
    def execute_tools(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        # Get tool_calls
        tool_calls = last_message.get("tool_calls", [])
        if not tool_calls:
            return {"messages": []}
        
        # Execute tools
        tool_outputs = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            tool_args = tool_call.get("function", {}).get("arguments")
            tool_id = tool_call.get("id")
            
            # Find and execute tool
            tool_result = None
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        # Parse arguments
                        if isinstance(tool_args, str):
                            args = json.loads(tool_args)
                        else:
                            args = tool_args
                        # Execute tool
                        result = tool.invoke(args)
                        tool_result = str(result)
                    except Exception as e:
                        tool_result = f"Error executing tool: {str(e)}"
                    break
            
            if tool_result is None:
                tool_result = f"Tool {tool_name} not found"
            
            # Create tool execution result message
            tool_message = {
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_id,
                "name": tool_name
            }
            tool_outputs.append(tool_message)
        
        return {"messages": tool_outputs}
    
    # Build LangGraph workflow
    workflow = StateGraph(ChatAgentState)
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", execute_tools)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# LangGraphChatAgent class (wrapper for MLflow inference)
class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent
    
    def _convert_messages_to_dict(self, messages: list[ChatAgentMessage]) -> list[dict]:
        """Convert ChatAgentMessage to dictionary format"""
        converted = []
        if not messages:
            return converted
        
        for msg in messages:
            try:
                if msg is None:
                    continue
                
                # Convert ChatAgentMessage object to dictionary
                if hasattr(msg, 'dict'):
                    msg_dict = msg.dict()
                elif isinstance(msg, dict):
                    msg_dict = msg
                else:
                    continue
                
                # Handle tool role messages
                if msg_dict.get("role") == "tool":
                    if not msg_dict.get("content"):
                        msg_dict["content"] = "Tool execution completed"
                    if "tool_call_id" not in msg_dict and msg_dict.get("id"):
                        msg_dict["tool_call_id"] = msg_dict["id"]
                
                converted.append(msg_dict)
            except Exception as e:
                print(f"Error converting message: {e}")
                continue
        
        return converted
    
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        # Convert input messages to dictionary format
        request = {"messages": self._convert_messages_to_dict(messages)}
        messages = []
        
        # Collect messages from LangGraph stream
        try:
            for event in self.agent.stream(request, stream_mode="updates"):
                if event and isinstance(event, dict):
                    for node_data in event.values():
                        if node_data and isinstance(node_data, dict) and "messages" in node_data:
                            for msg in node_data.get("messages", []):
                                if msg is None:
                                    continue
                                
                                # Convert message object to dictionary
                                if hasattr(msg, 'dict'):
                                    msg_dict = msg.dict()
                                elif isinstance(msg, dict):
                                    msg_dict = msg
                                else:
                                    continue
                                
                                try:
                                    messages.append(ChatAgentMessage(**msg_dict))
                                except Exception as e:
                                    print(f"Warning: Failed to create ChatAgentMessage: {e}")
        except Exception as e:
            print(f"Error in predict method: {e}")
        
        return ChatAgentResponse(messages=messages)
    
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        # Similar implementation for streaming
        request = {"messages": self._convert_messages_to_dict(messages)}
        
        try:
            for event in self.agent.stream(request, stream_mode="updates"):
                if event and isinstance(event, dict):
                    for node_data in event.values():
                        if node_data and isinstance(node_data, dict) and "messages" in node_data:
                            for msg in node_data.get("messages", []):
                                if msg is None:
                                    continue
                                
                                # Process and yield chunks
                                # ... (similar processing as predict)
                                yield ChatAgentChunk(**{"delta": msg_dict})
        except Exception as e:
            print(f"Error in predict_stream method: {e}")
            return

# Create agent object and specify it for inference
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)
```

### Step 2: Set Environment Variables in Driver Notebook

```python
# driver.py
import os
from databricks.sdk import WorkspaceClient
import re

# Get current user information
w = WorkspaceClient()
user_email = w.current_user.me().emails[0].value
username = user_email.split('@')[0]
username = re.sub(r'[^a-zA-Z0-9_]', '_', username)

# Set user-specific schema
user_schema_name = f"agents_lab_{username}"

# Set environment variables
os.environ["LLM_ENDPOINT_NAME"] = "databricks-claude-3-7-sonnet"
os.environ["UC_TOOL_NAMES"] = f"{catalog_name}.{user_schema_name}.*"
os.environ["VS_NAME"] = f"{catalog_name}.{system_schema_name}.product_docs_index"

print("Environment variables set:")
print(f"LLM_ENDPOINT_NAME: {os.environ.get('LLM_ENDPOINT_NAME')}")
print(f"UC_TOOL_NAMES: {os.environ.get('UC_TOOL_NAMES')}")
print(f"VS_NAME: {os.environ.get('VS_NAME')}")
```

### Step 3: Test the Agent

```python
from agent import AGENT

# Test product question
response = AGENT.predict({
    "messages": [{
        "role": "user", 
        "content": "What troubleshooting tips do you have for Soundwave X5 Pro headphones?"
    }]
})
print(response)

# Test date function
response = AGENT.predict({
    "messages": [{
        "role": "user", 
        "content": "What is today's date?"
    }]
})
print(response)
```

### Step 4: Log Agent as MLflow Model

```python
import mlflow
from agent import tools, LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

# Determine Databricks resources for authentication pass-through
resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]

for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What color options are available for the Aria Modern Bookshelf?"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        extra_pip_requirements=["databricks-connect"]
    )

# Load model and create prediction wrapper
logged_model_uri = f"runs:/{logged_agent_info.run_id}/agent"
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

def predict_wrapper(query):
    model_input = {
        "messages": [{"role": "user", "content": query}]
    }
    response = loaded_model.predict(model_input)
    messages = response['messages']
    return messages[-1]['content']
```

### Step 5: Create Evaluation Dataset

```python
import pandas as pd

data = {
    "request": [
        "What color options are available for the Aria Modern Bookshelf?",
        "How should I clean the Aurora Oak Coffee Table without damaging it?",
        "How should the BlendMaster Elite 4000 be cleaned after use?",
        "What colors does the Flexi-Comfort Office Desk come in?",
        "What sizes are available for the StormShield Pro Men's Waterproof Jacket?"
    ],
    "expected_facts": [
        [
            "Aria Modern Bookshelf is available in Natural Oak finish.",
            "Aria Modern Bookshelf is available in Black finish.",
            "Aria Modern Bookshelf is available in White finish."
        ],
        [
            "Clean with a soft, slightly damp cloth.",
            "Do not use abrasive cleaners."
        ],
        [
            "Rinse the BlendMaster Elite 4000 jar.",
            "Rinse with warm water.",
            "Clean after every use."
        ],
        [
            "Flexi-Comfort Office Desk comes in 3 colors."
        ],
        [
            "StormShield Pro Men's Waterproof Jacket sizes are S, M, L, XL, XXL."
        ]
    ]
}

eval_dataset = pd.DataFrame(data)
display(eval_dataset)
```

### Step 6: Define Evaluation Scorers

Define LLM judges (scorers) to evaluate agent responses:

```python
from mlflow.genai.scorers import Guidelines, Safety
import mlflow.genai

# Create evaluation dataset
eval_data = []
for request, facts in zip(data["request"], data["expected_facts"]):
    eval_data.append({
        "inputs": {
            "query": request  # Match function argument
        },
        "expected_response": "\n".join(facts)
    })

# Define evaluation scorers
# LLM judges evaluate responses against guidelines
scorers = [
    Guidelines(
        guidelines="""The response must include all expected facts:
        - List all colors or sizes if applicable (partial lists are not acceptable)
        - Include exact specifications when applicable (e.g., "5 ATM", not vague expressions)
        - Include all steps when asked about cleaning procedures
        Fail if any fact is missing or incorrect.""",
        name="completeness_and_accuracy",
    ),
    Guidelines(
        guidelines="""The response must be clear and direct:
        - Answer the question precisely
        - List options in list format, procedures in step format
        - No marketing language or unnecessary background
        - Be concise yet complete.""",
        name="relevance_and_structure",
    ),
    Guidelines(
        guidelines="""The response must stay on topic:
        - Only answer about the specific product asked
        - Do not add fictional features or colors
        - Do not include general advice
        - Use the exact product name mentioned in the request.""",
        name="product_specificity",
    ),
]
```

### Step 7: Run Evaluation

```python
print("Running evaluation...")
with mlflow.start_run():
    results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=predict_wrapper,
        scorers=scorers,
    )

# Review results in MLflow UI
print("Evaluation complete! Check MLflow experiment for detailed results.")
```

### Step 8: Iterate and Improve

Based on evaluation results, you may need to:

1. **Adjust System Prompt**: If responses are too verbose or marketing-focused, modify the system prompt:
```python
# More focused system prompt
system_prompt = """You are a Databricks Lab customer success specialist. 
When users ask questions about products, retrieve necessary information using tools 
and answer only the specific question concisely. Do not add fictional features, 
colors, or general comments. Avoid marketing language or unnecessary background."""
```

2. **Adjust Retriever Settings**: Change the number of documents retrieved:
```python
vs_tool = VectorSearchRetrieverTool(
    index_name=VS_NAME,
    tool_name="search_product_docs",
    num_results=1,  # Reduce from 3 to 1 for more focused results
    tool_description="Use this tool to search product documentation."
)
```

3. **Re-run Evaluation**: After making changes, re-log the model and re-evaluate:
```python
# Re-log improved agent
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        extra_pip_requirements=["databricks-connect"]
    )

# Re-evaluate
print("Running improved evaluation...")
with mlflow.start_run():
    results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=predict_wrapper,
        scorers=scorers,
    )
```

### Step 9: Register Model in Unity Catalog

```python
mlflow.set_registry_uri("databricks-uc")

# Define UC model location
model_name = "product_agent"
UC_MODEL_NAME = f"{catalog_name}.{user_schema_name}.{model_name}"

# Register model in UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, 
    name=UC_MODEL_NAME
)

print(f"Model registered: {UC_MODEL_NAME}")
print(f"Version: {uc_registered_model_info.version}")
```

### Step 10: Deploy Agent to Model Serving Endpoint

```python
from databricks import agents

# Define environment variables as dictionary
environment_vars = {
    "LLM_ENDPOINT_NAME": os.environ["LLM_ENDPOINT_NAME"],
    "UC_TOOL_NAMES": os.environ["UC_TOOL_NAMES"],
    "VS_NAME": os.environ["VS_NAME"],
}

# Deploy model to review app and model serving endpoint
agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    tags={"endpointSource": "Agent Lab"},
    environment_vars=environment_vars,
    timeout=900,  # 15 minutes
)

print("Agent deployment initiated. Wait a few minutes for the endpoint to become ready.")
```

## Best Practices and Tips

### 1. Tool Design
- **Keep tools focused**: Each tool should have a single, clear purpose
- **Handle errors gracefully**: Tools should return meaningful error messages
- **Document thoroughly**: Use clear docstrings for automatic metadata generation

### 2. Prompt Engineering
- **Be specific**: Clear system prompts lead to better agent behavior
- **Test iterations**: Try different prompt variations and measure performance
- **Consider edge cases**: Include handling for unexpected inputs

### 3. Evaluation Strategy
- **Create comprehensive test sets**: Cover various use cases and edge cases
- **Use multiple scorers**: Evaluate different aspects of agent performance
- **Track metrics over time**: Use MLflow to monitor improvements

### 4. Performance Optimization
- **Vector search tuning**: Adjust `num_results` based on your use case
- **Cache frequently used data**: Consider materializing common queries
- **Monitor latency**: Track response times in production

### 5. Security Considerations
- **Use authentication**: Leverage Databricks authentication for endpoints
- **Audit tool usage**: Monitor which tools are called and by whom
- **Validate inputs**: Ensure tools validate and sanitize inputs

### 6. Troubleshooting Common Issues

#### Issue: Vector Search Index not ready
**Solution**: Ensure Change Data Feed is enabled and wait for initial sync to complete

#### Issue: Tools not found by agent
**Solution**: Verify Unity Catalog permissions and function names in environment variables

#### Issue: Poor evaluation scores
**Solution**: Review system prompt, adjust retriever settings, and ensure evaluation criteria align with agent capabilities

#### Issue: Deployment timeout
**Solution**: Increase timeout value or check for resource constraints

## Conclusion

You've now built a complete agent system in Databricks that can:
- Access and query structured data through SQL functions
- Perform computations using Python functions
- Search documentation using Vector Search
- Make intelligent decisions using LLMs
- Be evaluated and improved systematically
- Be deployed to production endpoints

This foundation can be extended with additional tools, more sophisticated evaluation metrics, and integration with other systems. The modular design allows for easy updates and improvements as your requirements evolve.

## Next Steps

1. **Extend the agent**: Add more tools for your specific use cases
2. **Improve evaluation**: Create synthetic data for comprehensive testing
3. **Build UI**: Create a Databricks App for user interaction
4. **Monitor production**: Set up alerts and monitoring for deployed endpoints
5. **Scale horizontally**: Deploy multiple specialized agents for different domains

Remember to iterate based on user feedback and continuously improve your agent's capabilities!