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

### What is an AI Agent?
An AI agent is an intelligent system that can understand natural language queries, reason about them, and take actions using various tools to accomplish tasks. Unlike simple chatbots, agents can:
- Access and query databases
- Perform calculations
- Search through documentation
- Make decisions based on business rules
- Learn from evaluation feedback

### Why Build Agents in Databricks?
Databricks provides a unified platform that combines:
- **Unity Catalog**: Centralized governance for data, functions, and models
- **Vector Search**: Semantic search capabilities for RAG (Retrieval Augmented Generation)
- **MLflow**: Experiment tracking and model lifecycle management
- **Model Serving**: Production-ready deployment infrastructure
- **AI Playground**: Interactive testing environment

### What You'll Learn
This guide walks you through building a customer service agent that can:
- Process product returns based on company policies
- Answer product-related questions using documentation
- Make intelligent decisions by combining multiple data sources
- Be systematically evaluated and improved

### Lab Structure
The hands-on lab consists of three main notebooks:
1. **Data Preparation** - Setting up the foundation
2. **Tool Preparation** - Creating agent capabilities
3. **Agent Building and Evaluation** - Implementing and deploying

## Prerequisites

### Required Components
- **Databricks workspace** with Unity Catalog enabled
- **Access permissions** to create catalogs and schemas
- **Compute resources**: Serverless or general-purpose cluster
- **AI Playground** access
- **LLM endpoint**: Claude 3.7 Sonnet or similar

### Required Python Packages
```python
# Core packages needed for agent development
mlflow-skinny[databricks]  # For model tracking and deployment
langgraph==0.3.4           # For building agent workflows
databricks-langchain        # Databricks-specific LangChain integrations
databricks-agents           # Agent deployment utilities
uv                         # Fast Python package installer
```

## Architecture Overview

### Understanding the Agent Architecture

The agent system consists of several interconnected components:

1. **Unity Catalog** - The governance layer
   - Stores all data tables, functions, and models in a governed manner
   - Provides access control and lineage tracking
   - Enables discovery and reuse of assets

2. **Vector Search Index** - The knowledge base
   - Converts text documents into mathematical representations (embeddings)
   - Enables semantic search to find relevant information
   - Powers the RAG (Retrieval Augmented Generation) capabilities

3. **SQL Functions** - Structured data access
   - Query customer service data
   - Retrieve business policies
   - Access order history

4. **Python Functions** - Custom logic
   - Handle operations LLMs cannot perform (like getting current date)
   - Implement business logic
   - Process data transformations

5. **LLM Endpoint** - The brain
   - Understands natural language queries
   - Decides which tools to use
   - Orchestrates the overall workflow
   - Generates human-readable responses

6. **MLflow** - The development platform
   - Tracks experiments and metrics
   - Manages model versions
   - Facilitates deployment

## Part 0: Data Preparation

### Why This Section Matters
Before building an agent, we need to establish the data foundation. This section sets up all the necessary infrastructure that the agent will use to answer questions and make decisions. Think of it as building the knowledge base and reference materials that a human customer service representative would need.

### Step 1: Set Up Configuration

**What we're doing**: Creating a centralized configuration file to manage names and settings across all notebooks.

**Why**: This ensures consistency and makes it easy to change settings in one place rather than hunting through multiple notebooks.

```python
# config.py
catalog_name = "your_catalog_name"  # Replace with your catalog
system_schema_name = "agents_lab"   # Shared schema for data

# This configuration will be imported by all notebooks
# ensuring everyone uses the same catalog and schema names
```

### Step 2: Create Catalog and Schema

**What we're doing**: Creating the Unity Catalog containers that will hold all our data and functions.

**Why**: Unity Catalog provides governance, security, and discovery. The catalog is like a database, and schemas are like folders within it. We're also granting permissions so team members can access these resources.

```python
%run ./config

# Display configuration for verification
print(f"Catalog for hands-on: {catalog_name}")
print(f"Shared schema for data storage: {system_schema_name}")

# Create catalog - this is the top-level container
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
# Grant permissions so all users in the account can use this catalog
spark.sql(f"GRANT USE CATALOG ON CATALOG {catalog_name} TO `account users`")

# Create schema - this is where we'll store our tables
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{system_schema_name}")
# Grant permissions for users to query tables in this schema
spark.sql(f"GRANT USE SCHEMA, SELECT ON SCHEMA {catalog_name}.{system_schema_name} TO `account users`")
```

### Step 3: Load Sample Data

**What we're doing**: Loading three CSV files that contain the business data our agent will need.

**Why**: These files represent different aspects of the business:
- **cust_service_data.csv**: Historical customer service interactions (for understanding patterns)
- **policies.csv**: Company policies (for making decisions)
- **product_docs.csv**: Product documentation (for answering questions)

```python
import os

# Load CSV files from workspace files
# We're using workspace files which are stored alongside the notebook
cust_service_data_df = spark.read.format("csv").load(
    f"file:{os.getcwd()}/data/cust_service_data.csv", 
    header=True,        # First row contains column names
    inferSchema=True    # Automatically detect data types
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
    multiline='true'    # Product descriptions may span multiple lines
)

# Display data to verify it loaded correctly
display(cust_service_data_df)
display(policies_df)
display(product_docs_df)
```

### Step 4: Save Data to Unity Catalog Tables

**What we're doing**: Converting the loaded DataFrames into permanent Unity Catalog tables.

**Why**: Unity Catalog tables provide:
- Persistence (data survives cluster restarts)
- Governance (access control, audit logging)
- Performance (optimized storage format)
- Discovery (other users can find and use these tables)

```python
# Save as Unity Catalog tables with Delta format (default)
# Delta provides ACID transactions, time travel, and optimization
cust_service_data_df.write.mode("overwrite").saveAsTable(
    f"{catalog_name}.{system_schema_name}.cust_service_data"
)

policies_df.write.mode("overwrite").saveAsTable(
    f"{catalog_name}.{system_schema_name}.policies"
)

product_docs_df.write.mode("overwrite").saveAsTable(
    f"{catalog_name}.{system_schema_name}.product_docs"
)

print("✅ Tables created successfully in Unity Catalog")
```

### Step 5: Create Vector Search Index

**What we're doing**: Creating a searchable index of product documentation that understands semantic meaning.

**Why**: Traditional database searches only find exact matches. Vector Search understands meaning - for example, it knows that "headphones" and "audio device" are related concepts. This enables the agent to find relevant information even when users use different words than what's in the documentation.

```python
# First, enable Change Data Feed for the source table
# This allows the Vector Search index to stay synchronized with table updates
source_table = f"{catalog_name}.{system_schema_name}.product_docs"

spark.sql(f"""
    ALTER TABLE {source_table}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

print(f"✅ Enabled Change Data Feed for {source_table}")

# Verify the setting was applied
spark.sql(f"SHOW TBLPROPERTIES {source_table}").filter(
    "key = 'delta.enableChangeDataFeed'"
).show()
```

**Creating the actual Vector Search index**:

```python
import requests
import json
import time

# Get Databricks workspace context for API calls
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = context.apiUrl().get()
token = context.apiToken().get()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Configure the Vector Search index
endpoint_name = "one-env-shared-endpoint-1"  # Compute endpoint for indexing
index_name = f"{catalog_name}.{system_schema_name}.product_docs_index"
source_table = f"{catalog_name}.{system_schema_name}.product_docs"

# Create index using Databricks API
url = f"{host}/api/2.0/vector-search/indexes"
payload = {
    "name": index_name,
    "endpoint_name": endpoint_name,
    "primary_key": "product_id",           # Unique identifier for each document
    "index_type": "DELTA_SYNC",            # Automatically syncs with source table
    "delta_sync_index_spec": {
        "source_table": source_table,
        "pipeline_type": "TRIGGERED",      # Index updates on demand
        "embedding_source_columns": [{
            "name": "product_doc",          # Column containing text to index
            "embedding_model_endpoint_name": "databricks-gte-large-en"  # Model for embeddings
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

**What we're doing**: Tracking the progress of index creation until it's ready for use.

**Why**: Creating a Vector Search index involves:
1. Reading all documents from the source table
2. Converting each document to an embedding (mathematical representation)
3. Building the search index structure

This process can take several minutes for large datasets, so we monitor progress to know when it's ready.

```python
def monitor_index_status(index_name, timeout_minutes=60):
    """
    Monitor the status of index creation
    
    Why: Vector Search indexing is asynchronous. We need to wait
    for it to complete before the agent can use it for searches.
    """
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
                
                # Check if indexing is complete
                if ready and ("ONLINE" in state or state == "READY"):
                    print(f"\n\n✅ Index is ready!")
                    print(f"📊 Total indexed rows: {indexed_count:,}")
                    return True
                    
        except Exception as e:
            print(f"\n⚠️ Exception: {e}")
        
        time.sleep(30)  # Check every 30 seconds to avoid overwhelming the API
    
    return False

# Start monitoring
success = monitor_index_status(index_name, timeout_minutes=60)
if success:
    print("\n🎉 Vector Search Index is ready for use!")
```

## Part 1: Building Your First Agent

### Why This Section Matters
Now that we have our data foundation, we need to create the tools that our agent will use to accomplish tasks. Think of these tools as the specific capabilities we're giving our agent - like giving a customer service representative access to different computer systems and teaching them specific procedures.

### Step 1: Create User-Specific Schema

**What we're doing**: Creating a personal workspace (schema) for each user to store their own functions.

**Why**: In a shared environment, multiple users might be working on agents simultaneously. Having separate schemas prevents conflicts and allows each person to experiment without affecting others.

```python
from databricks.sdk import WorkspaceClient
import re

# Get current user information from Databricks
w = WorkspaceClient()
user_email = w.current_user.me().emails[0].value
username = user_email.split('@')[0]
# Clean username to make it safe for database names
username = re.sub(r'[^a-zA-Z0-9_]', '_', username)

# Create user-specific schema name
user_schema_name = f"agents_lab_{username}"
print("Your catalog:", catalog_name)
print("Your schema:", user_schema_name)

# Create widgets for SQL/Python reference
# Widgets allow SQL cells to reference these Python variables
dbutils.widgets.text("catalog_name", defaultValue=catalog_name, label="Catalog Name")
dbutils.widgets.text("system_schema_name", defaultValue=system_schema_name, label="System Schema Name")
dbutils.widgets.text("user_schema_name", defaultValue=user_schema_name, label="User Schema Name")

# Create the user's personal schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{user_schema_name}")
```

### Step 2: Create SQL Functions for Customer Service Workflow

**What we're doing**: Creating reusable SQL functions that encapsulate common queries the agent will need.

**Why**: Instead of having the LLM write complex SQL queries (which it might get wrong), we create tested, optimized functions that the agent can simply call with parameters. This follows the principle of "give the LLM tools, not responsibilities."

#### Function 1: Get Latest Return

**Purpose**: Retrieves the most recent customer service request from the queue.

**Why needed**: Customer service typically works through a queue of requests. This function lets the agent get the next item to process.

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
    ORDER BY date_time DESC  -- Most recent first
    LIMIT 1                  -- Only get the latest one
);

-- Test the function to ensure it works
SELECT * FROM IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_latest_return')()
```

#### Function 2: Get Return Policy

**Purpose**: Retrieves the company's current return policy.

**Why needed**: The agent needs to check company policies before making decisions about returns. This ensures consistent policy application.

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

-- Test the function
SELECT * FROM IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_return_policy')()
```

#### Function 3: Get User ID by Name

**Purpose**: Converts a customer name to their unique identifier.

**Why needed**: Most database operations use IDs rather than names for performance and accuracy. This function provides the translation.

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

-- Test the function with a sample name
SELECT IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_user_id')('Nicolas Pelaez');
```

#### Function 4: Get Order History

**Purpose**: Retrieves a customer's return history for the past 12 months.

**Why needed**: Return policies often have limits (e.g., maximum returns per year). This function lets the agent check if a customer has exceeded those limits.

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
    GROUP BY issue_category;  -- Group to see patterns in return reasons

-- Test with a sample user ID
SELECT * FROM IDENTIFIER(:catalog_name || '.' || :user_schema_name || '.get_order_history')('453e50e0-232e-44ea-9fe3-28d550be6294')
```

### Step 3: Create Python Function for Current Date

**What we're doing**: Creating a Python function that returns the current date.

**Why needed**: LLMs are trained on historical data and don't inherently know the current date. Many business decisions (like checking if something is within a 30-day return window) require knowing today's date.

```python
def get_todays_date() -> str:
    """
    Returns today's date in 'YYYY-MM-DD' format.
    
    Why this function exists:
    - LLMs don't have real-time awareness
    - Date calculations are critical for policy enforcement
    - Standardized format ensures consistency
    
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

**What we're doing**: Making the Python function available as a Unity Catalog function that the agent can discover and use.

**Why**: Unity Catalog provides a centralized registry of functions. By registering here, the function becomes discoverable, governed, and can be used by any authorized user or agent.

```python
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

# Create a client to interact with Unity Catalog
client = DatabricksFunctionClient()

# Deploy the function to Unity Catalog
# This automatically:
# - Extracts metadata from the function's docstring
# - Sets up the execution environment
# - Makes it discoverable by agents
python_tool_uc_info = client.create_python_function(
    func=get_todays_date, 
    catalog=catalog_name, 
    schema=user_schema_name, 
    replace=True  # Overwrite if it already exists
)

# Display the fully qualified function name
print(f"Deployed Unity Catalog function: {python_tool_uc_info.full_name}")

# The function can now be called using SQL or by agents
```

### Step 5: Test in AI Playground

**What we're doing**: Using Databricks AI Playground to interactively test our agent with the created tools.

**Why**: AI Playground provides a user-friendly interface to:
- Test different prompts and see how the agent responds
- Observe which tools the agent chooses to use
- Debug issues before writing code
- Understand the agent's reasoning process

**Steps to test in AI Playground**:

1. **Navigate to AI Playground** in the Databricks workspace (found under "AI/ML" in the left navigation)

2. **Select the LLM**: Choose Claude 3.7 Sonnet or your preferred model

3. **Set the system prompt** to guide the agent's behavior:
   ```
   Call tools until you are confident all company policies are met
   ```

4. **Add your created functions** as tools:
   - Click "Add Tools"
   - Select your functions from Unity Catalog
   - The agent can now use these tools

5. **Test with a realistic question**:
   ```
   Based on our policies, should we accept the latest return in the queue?
   ```

6. **Observe the agent's process**:
   - The agent reads the latest return
   - Retrieves the return policy
   - Gets the customer's history
   - Makes a decision based on all information

7. **Review MLflow traces** to understand:
   - Which tools were called
   - In what order
   - What data was passed between them
   - How long each step took

## Part 2: Agent Evaluation and Deployment

### Why This Section Matters
Building an agent is just the beginning. To create a production-ready system, we need to:
- Implement the agent as deployable code (not just playground experiments)
- Systematically evaluate its performance
- Iterate based on evaluation results
- Deploy to a production endpoint

This section transforms our prototype into a production-ready system.

### Step 1: Create Agent Implementation (agent.py)

**What we're doing**: Creating a complete, production-ready agent implementation that combines all our tools with an LLM.

**Why**: The AI Playground is great for prototyping, but production requires:
- Code that can be version controlled
- Configuration through environment variables
- Proper error handling
- Standardized interfaces for deployment

```python
# agent.py - Complete agent implementation
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
# Configuration Management
############################################
def get_env_config():
    """
    Get configuration from environment variables
    
    Why: Environment variables allow us to:
    - Change settings without modifying code
    - Use different configurations for dev/staging/prod
    - Keep sensitive information out of code
    """
    llm_endpoint = os.environ.get("LLM_ENDPOINT_NAME")
    uc_tool_names = os.environ.get("UC_TOOL_NAMES", "")
    vs_name = os.environ.get("VS_NAME", "")
    
    # Validate that all required configuration is present
    if not llm_endpoint:
        raise ValueError("LLM_ENDPOINT_NAME environment variable is required")
    if not uc_tool_names:
        raise ValueError("UC_TOOL_NAMES environment variable is required")
    if not vs_name:
        raise ValueError("VS_NAME environment variable is required")
    
    # Parse tool names from comma-separated string
    tool_names = [name.strip() for name in uc_tool_names.split(",") if name.strip()]
    
    return {
        "llm_endpoint": llm_endpoint,
        "uc_tool_names": tool_names,
        "vs_name": vs_name
    }

# Load configuration
config = get_env_config()

# Print configuration for debugging
print("Agent Configuration:")
print(f"LLM_ENDPOINT_NAME: {config['llm_endpoint']}")
print(f"UC_TOOL_NAMES: {config['uc_tool_names']}")
print(f"VS_NAME: {config['vs_name']}")

# Enable automatic logging of agent interactions for debugging
mlflow.langchain.autolog()

# Initialize the function client for Unity Catalog
client = DatabricksFunctionClient(disable_notice=True, suppress_warnings=True)
set_uc_function_client(client)

############################################
# LLM Setup
############################################
# Create the LLM instance that will power our agent
llm = ChatDatabricks(endpoint=config["llm_endpoint"])

# System prompt defines the agent's personality and behavior
system_prompt = """You are a Databricks Lab customer success specialist. 
When users ask questions about products, retrieve necessary information using tools 
and help users fully understand the products. Include relevant information that 
might interest customers and strive to provide value in every interaction."""

############################################
# Tool Creation
############################################
def create_tools() -> List[BaseTool]:
    """
    Create all tools the agent can use
    
    Why: Tools give the agent capabilities beyond just generating text.
    Each tool is a specific action the agent can take.
    """
    tools = []
    
    # Add Vector Search tool for semantic document search
    if config['vs_name']:
        try:
            vs_tool = VectorSearchRetrieverTool(
                index_name=config['vs_name'],
                tool_name="search_product_docs",
                num_results=3,  # Return top 3 most relevant documents
                tool_description="Use this tool to search product documentation."
            )
            tools.append(vs_tool)
            print(f"Added Vector Search tool: {config['vs_name']}")
        except Exception as e:
            print(f"Warning: Could not load Vector Search tool: {e}")
    
    # Add Unity Catalog functions as tools
    if config['uc_tool_names']:
        try:
            uc_toolkit = UCFunctionToolkit(function_names=config['uc_tool_names'])
            tools.extend(uc_toolkit.tools)
            print(f"Added UC function tools: {config['uc_tool_names']}")
        except Exception as e:
            print(f"Warning: Could not add UC tools: {e}")
    
    return tools

# Create all tools
tools = create_tools()

############################################
# Agent Logic Implementation
############################################
def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    """
    Create the agent's decision-making logic using LangGraph
    
    Why LangGraph: It provides a structured way to:
    - Define the flow of agent reasoning
    - Handle tool calls and responses
    - Manage conversation state
    - Enable both batch and streaming responses
    """
    
    # Bind tools to the model so it knows what tools are available
    model = model.bind_tools(tools)
    
    def should_continue(state: ChatAgentState):
        """
        Decide whether to call more tools or provide final answer
        
        Why: The agent needs to know when it has enough information
        to answer the user's question
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message includes tool calls, execute them
        # Otherwise, we're done and can return the response
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"
    
    # Add system prompt to guide agent behavior
    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    
    model_runnable = preprocessor | model
    
    def call_model(state: ChatAgentState, config: RunnableConfig):
        """
        Call the LLM to generate a response or decide on tool usage
        """
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}
    
    def execute_tools(state: ChatAgentState):
        """
        Execute any tools the model decided to call
        
        Why: The LLM decides which tools to call, but we need
        to actually execute them and return results
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_calls = last_message.get("tool_calls", [])
        if not tool_calls:
            return {"messages": []}
        
        tool_outputs = []
        for tool_call in tool_calls:
            # Extract tool details
            tool_name = tool_call.get("function", {}).get("name")
            tool_args = tool_call.get("function", {}).get("arguments")
            tool_id = tool_call.get("id")
            
            # Find and execute the requested tool
            tool_result = None
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        # Parse arguments if they're JSON string
                        if isinstance(tool_args, str):
                            args = json.loads(tool_args)
                        else:
                            args = tool_args
                        
                        # Execute the tool and capture result
                        result = tool.invoke(args)
                        tool_result = str(result)
                    except Exception as e:
                        tool_result = f"Error executing tool: {str(e)}"
                    break
            
            if tool_result is None:
                tool_result = f"Tool {tool_name} not found"
            
            # Format tool results for the model
            tool_message = {
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_id,
                "name": tool_name
            }
            tool_outputs.append(tool_message)
        
        return {"messages": tool_outputs}
    
    # Build the agent workflow graph
    # This defines how the agent processes requests
    workflow = StateGraph(ChatAgentState)
    
    # Add nodes for agent reasoning and tool execution
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", execute_tools)
    
    # Define the flow:
    # 1. Start with agent node
    # 2. Agent decides whether to use tools
    # 3. If tools needed, execute them and return to agent
    # 4. Repeat until agent has final answer
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
    """
    Wrapper class that makes our LangGraph agent compatible with MLflow
    
    Why: MLflow expects a specific interface for chat agents.
    This class translates between LangGraph's format and MLflow's format.
    """
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent
    
    def _convert_messages_to_dict(self, messages: list[ChatAgentMessage]) -> list[dict]:
        """Convert ChatAgentMessage objects to dictionary format for processing"""
        converted = []
        if not messages:
            return converted
        
        for msg in messages:
            try:
                if msg is None:
                    continue
                
                # Handle different message formats
                if hasattr(msg, 'dict'):
                    msg_dict = msg.dict()
                elif isinstance(msg, dict):
                    msg_dict = msg
                else:
                    continue
                
                # Special handling for tool response messages
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
        """
        Process a conversation and return the agent's response
        
        Why: This is the main entry point for using the agent.
        It handles the complete cycle of understanding the query,
        calling tools, and generating a response.
        """
        request = {"messages": self._convert_messages_to_dict(messages)}
        messages = []
        
        # Stream through the agent's reasoning process
        try:
            for event in self.agent.stream(request, stream_mode="updates"):
                if event and isinstance(event, dict):
                    for node_data in event.values():
                        if node_data and isinstance(node_data, dict) and "messages" in node_data:
                            for msg in node_data.get("messages", []):
                                if msg is None:
                                    continue
                                
                                # Convert and collect messages
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

# Create and register the agent
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)