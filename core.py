# Importing libs and modules
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate # Keep for custom tool prompt if needed later
from langchain import hub # For pulling agent prompts

import os
from dotenv import load_dotenv

# Setting Google API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv('API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

if not GOOGLE_API_KEY:
    raise ValueError("API_KEY not found in environment variables.")
# GOOGLE_CSE_ID check will be done when the tool is initialized, or we can check here too.
# For now, let's assume it's set if search is intended.

genai.configure(api_key=GOOGLE_API_KEY) #This is for google.generativeai direct calls, ChatGoogleGenerativeAI uses it implicitly


# Path of vectore database
DB_FAISS_PATH = 'vectorstore/db_faiss'


# Agent prompt will be different, pulling from hub or defining a new one.
# The detailed persona from the original custom_prompt_template will be integrated into the agent's system message/prompt.
# Example of how to incorporate custom instructions into an agent prompt:
# You are a helpful AI assistant for DBATU. Your instructions are:
# (Detailed instructions from the old custom_prompt_template)
# You have access to the following tools: ...

#Loading the model
def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", 
        temperature=0.6,
        # convert_system_message_to_human=True # May be needed for some older agent types if system messages aren't directly supported by Gemini through LC yet
    )
    return llm

# This function will now create and return an AgentExecutor
def create_dbatu_agent_executor():
    llm = load_llm()
    
    # 1. Tool for searching DBATU's own documents (Vector DB)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    def search_dbatu_documents(query: str) -> str:
        """Searches Dr. Babasaheb Ambedkar Technological University's document database for relevant information."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    dbatu_retriever_tool = Tool(
        name="DBATUDocumentSearch",
        func=search_dbatu_documents,
        description="Use this tool to search for information specifically within Dr. Babasaheb Ambedkar Technological University's internal documents and knowledge base. Useful for specific university policies, detailed course information, internal announcements, etc."
    )

    # 2. Tool for Google Search
    if not GOOGLE_CSE_ID:
        print("Warning: GOOGLE_CSE_ID not found in environment. Google Search tool will not be available.")
        google_search_tool = None
    else:
        search_wrapper = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
        google_search_tool = Tool(
            name="GoogleSearch",
            func=search_wrapper.run,
            description="Use this tool for general web searches to find up-to-date information, news, or topics not covered in DBATU's internal documents. Useful for current events, external opinions, or broader topics."
        )

    tools = [dbatu_retriever_tool]
    if google_search_tool:
        tools.append(google_search_tool)

    # Get the ReAct prompt
    # The hwchase17/react-chat prompt is often better for conversational agents
    # We will need to customize this prompt to include all the specific DBATU instructions
    # For now, let's pull a standard ReAct prompt.
    # TODO: Customize this prompt with the detailed persona and rules from the original custom_prompt_template
    prompt_hub_template = "hwchase17/react" 
    try:
        prompt = hub.pull(prompt_hub_template)
        # Example of how one might try to inject custom instructions:
        # This is simplistic; proper customization requires modifying the prompt template structure.
        # For ReAct, this often means adding to the initial system message part of the prompt.
        # The original custom_prompt_template's content needs to be adapted to the agent's thinking process.
        
        # Placeholder for detailed instructions - this needs to be integrated carefully into the agent's prompt
        dbatu_specific_instructions = """
You are an AI assistant for Dr. Babasaheb Ambedkar Technological University (DBATU).
Your primary role is to provide detailed and accurate information about DBATU.
Be multilingual (Marathi, Indian languages). Maintain a professional, polite, helpful, patient, detailed, and natural conversational tone.
You are an official university representative.

Core Instructions:
1.  Answering: Prioritize DBATUDocumentSearch. If insufficient, use GoogleSearch. Do not speculate if no reliable answer.
2.  Greetings: Only greet if user greets (e.g., "Hi", "Hello"). Respond politely and briefly. No unsolicited greetings.
3.  Relevance: Strictly DBATU-related questions ONLY (academics, admissions, campus life, etc.). For unrelated questions, politely state: "I specialize in information about Dr. Babasaheb Ambedkar Technological University. Could you please ask a question related to DBATU?" and do not engage further.
4.  Links: If DBATUDocumentSearch provides a relevant link, include it. For GoogleSearch links, provide only if asked or highly relevant and verifiable. No invented links.
5.  Insufficient Info: If all tools fail, state: "I've searched for information on that topic but couldn't find specific details. For the most current information, you might want to check the official DBATU website: https://dbatu.ac.in/ or contact the relevant university department."
6.  Formatting: Use clear formatting, bullets (-), **bold** for emphasis.
"""
        # A common way to customize ReAct prompts is to modify the 'system_message' or 'instructions' part
        # This depends on the specific structure of the pulled prompt.
        # For "hwchase17/react", the instructions are part of the template string itself.
        # A more robust way is to construct the prompt manually or adapt the pulled one carefully.
        # For now, we are just logging it. The agent will use its default instructions + tool descriptions.
        # The actual detailed persona needs to be integrated into the prompt passed to create_react_agent.
        # This is a placeholder for now, will require more work to make the agent fully adopt the persona.
        
        # A simple way to prepend instructions to the prompt (might need refinement)
        # This is a basic example, proper prompt engineering for agents is key.
        # A better way is to create a new PromptTemplate object and modify its template string.
        original_template_str = prompt.template
        customized_template_str = dbatu_specific_instructions + "\n\n" + original_template_str
        prompt = PromptTemplate.from_template(customized_template_str)
        # This might break the input_variables expectation of the ReAct prompt.
        # Reverting to the original pulled prompt for now to ensure functionality.
        # Customizing agent prompts correctly is a more involved task.
        prompt = hub.pull(prompt_hub_template) 


    except Exception as e:
        print(f"Error pulling prompt from hub, using a default. Error: {e}")
        # Fallback prompt if hub fails - this is very basic
        template = """Answer the following questions as best you can. You have access to the following tools:
        {tools}
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

        Question: {input}
        Thought:{agent_scratchpad}"""
        prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor

# User input function
def user_input(user_question):
    # Check for GOOGLE_CSE_ID here or ensure it's handled in create_dbatu_agent_executor
    if not os.getenv('GOOGLE_CSE_ID'):
        print("WARNING: GOOGLE_CSE_ID is not set in the environment. Google Search functionality will be disabled.")
        # Optionally, you could decide to not even create the agent or use a version without search.
        # For now, create_dbatu_agent_executor handles the conditional tool addition.

    agent_executor = create_dbatu_agent_executor()
    
    try:
        # For ReAct agent, input is typically a string.
        # Chat history can be managed by passing `chat_history` to invoke.
        # The prompt needs to be set up to handle chat_history if used.
        response = agent_executor.invoke({"input": user_question})
        # The agent's response format might be different, usually in `response['output']`
        return {"output_text": response.get("output", "No output from agent.")}
    except Exception as e:
        print(f"Error during agent execution: {e}")
        return {"output_text": "Sorry, I encountered an error while processing your request."}

# The old custom_prompt_template, set_custom_prompt, get_conversational_chain, 
# and the old user_input logic have been removed as the agent structure replaces them.
# Ensure any necessary helper text or suggestions are integrated into the agent's prompt or system design.
