# Importing libs and modules
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SerpAPIWrapper # Changed from GoogleSearchAPIWrapper
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate 
from langchain import hub 

import os
from dotenv import load_dotenv

# Setting API Keys
load_dotenv()
GOOGLE_API_KEY = os.getenv('API_KEY') 
SERPER_API_KEY = os.getenv('SERPER_API_KEY') # New key for Serper

if not GOOGLE_API_KEY:
    raise ValueError("API_KEY (for Google Generative AI) not found in environment variables.")
if not SERPER_API_KEY:
    # This can be a warning if search is optional, or an error if it's crucial.
    # For an agent that's expected to search, it's better to make it clear if it's missing.
    print("Warning: SERPER_API_KEY not found in environment variables. Web search tool will not be available.")

genai.configure(api_key=GOOGLE_API_KEY)


# Path of vector database
DB_FAISS_PATH = 'vectorstore/db_faiss'

#Loading the model
def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", 
        temperature=0.6,
    )
    return llm

# This function will now create and return an AgentExecutor
def create_dbatu_agent_executor():
    llm = load_llm()
    
    # 1. Tool for searching DBATU's own documents (Vector DB)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # Ensure FAISS DB exists or handle creation if not found
    if not os.path.exists(DB_FAISS_PATH):
        # This is a critical error for this tool. The DB should be created by app_embeddings.py
        print(f"CRITICAL ERROR: FAISS database not found at {DB_FAISS_PATH}. DBATUDocumentSearch tool cannot function.")
        # Depending on desired behavior, could raise an error or return a non-functional tool/agent.
        # For now, let it proceed and the agent might fail or not use the tool.
        # A more robust solution would be to ensure app_embeddings.py has run successfully.
        # Or, provide a way for the agent to inform the user that internal search is unavailable.
        # Returning None or an empty list of tools if critical components are missing might be an option.
        # This function's purpose is to create the executor, so we'll let it try.
        pass # Allow it to try loading, FAISS.load_local will raise the error.

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    def search_dbatu_documents(query: str) -> str:
        """
        Searches Dr. Babasaheb Ambedkar Technological University's (DBATU) document database 
        for specific information related to the university, its courses, policies, etc.
        Use this tool first for any DBATU-specific questions.
        """
        try:
            docs = retriever.invoke(query)
            if not docs:
                return "No specific information found in DBATU documents for your query."
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error during DBATU document search: {e}")
            return "There was an error searching the DBATU documents."

    dbatu_retriever_tool = Tool(
        name="DBATUInternalKnowledgeSearch", # Renamed for clarity
        func=search_dbatu_documents,
        description="Searches Dr. Babasaheb Ambedkar Technological University's (DBATU) internal knowledge base for specific information. Use this for questions about university policies, course details, admission criteria, internal circulars, etc. This should be your primary tool for DBATU-specific queries."
    )

    # 2. Tool for Serper Web Search
    serper_search_tool = None
    if SERPER_API_KEY:
        try:
            search_wrapper = SerpAPIWrapper(serpapi_api_key=SERPER_API_KEY)
            serper_search_tool = Tool(
                name="WebSearch", # Generic name for web search
                func=search_wrapper.run,
                description="Performs a web search for general information, current events, or topics not covered by DBATU's internal knowledge base. Use this if DBATUInternalKnowledgeSearch doesn't yield results or for broader context."
            )
        except Exception as e:
            print(f"Error initializing Serper search tool: {e}. Web search will be unavailable.")
    else:
        print("SERPER_API_KEY not set. WebSearch tool is disabled.")


    tools = [dbatu_retriever_tool]
    if serper_search_tool:
        tools.append(serper_search_tool)
    
    if not tools:
         # This case should ideally not happen if dbatu_retriever_tool is always added.
         # But as a safeguard if dbatu_retriever_tool also had an init failure.
        print("CRITICAL: No tools available for the agent.")
        # Return a dummy executor or raise an error, as an agent with no tools is not useful.
        # For now, let it try to create an agent, it will likely not be very effective.
        # A better approach is to handle this more gracefully in user_input.

    # TODO: Customize this prompt with the detailed persona and rules from the original custom_prompt_template.
    # The current approach of simple concatenation for dbatu_specific_instructions is very basic
    # and likely does not integrate well with the ReAct prompt structure.
    # Effective agent behavior requires careful prompt engineering.
    prompt_hub_template = "hwchase17/react" 
    try:
        prompt = hub.pull(prompt_hub_template)
        
        # Placeholder for detailed instructions integration
        # This needs to be properly structured within the chosen agent prompt (e.g., ReAct).
        # For instance, modifying the system message or the thought process examples.
        dbatu_persona_and_rules = """
        **Your Role and Persona:**
        You are a helpful, polite, and knowledgeable AI assistant representing Dr. Babasaheb Ambedkar Technological University (DBATU).
        Your primary role is to provide detailed and accurate information to users about DBATU.
        You are multilingual and can understand and respond in various Indian languages, including Marathi, in a natural and conversational manner.
        Always maintain a respectful, helpful, and patient tone. Provide detailed and comprehensive answers.
        Your responses should sound natural and conversational. Remember you are an official assistant for DBATU.

        **Core Operational Rules:**
        1.  **Tool Usage Priority**: For any question about DBATU, always use the `DBATUInternalKnowledgeSearch` tool first. If it doesn't provide a satisfactory answer, then consider using `WebSearch` for broader, up-to-date, or external information.
        2.  **Greetings**: Only greet the user if they explicitly greet you (e.g., "Hi", "Hello"). Respond with a polite and brief greeting (e.g., "Hello! How can I assist you with DBATU today?"). Do not initiate greetings otherwise.
        3.  **Question Relevance**: You MUST ONLY answer questions related to Dr. Babasaheb Ambedkar Technological University (academics, admissions, campus life, facilities, official announcements, etc.). If a question is unrelated, politely state: "I specialize in information about Dr. Babasaheb Ambedkar Technological University. How can I help you with a DBATU-related question?" Do not engage further on unrelated topics.
        4.  **Citing Links**: If the `DBATUInternalKnowledgeSearch` tool returns information that includes a web link, you may provide it. For links from `WebSearch`, only provide them if specifically asked or if they are highly relevant and verifiable official sources. Do not invent links.
        5.  **Handling Insufficient Information**: If, after using all available tools, you cannot find a reliable answer, state: "I've looked through the available resources but couldn't find specific information on that topic. For the most current details, please check the official DBATU website: https://dbatu.ac.in/ or consider contacting the relevant university department." Do not speculate or make up answers.
        6.  **Formatting**: Use clear formatting. Utilize bullet points (-) for lists. Use **bold text** for emphasis.
        """
        # This is a naive way of prepending. A proper ReAct prompt customization is more involved.
        # It often requires modifying specific parts of the prompt template (e.g., system message, examples).
        # For now, this is a placeholder to show where the instructions *should* go.
        # The agent might not pick these up effectively without proper prompt engineering.
        # It's better to use the default hub prompt if unsure how to customize it properly.
        # This customization is left as a TODO.
        # print(f"Original prompt template:\n{prompt.template}")
        # prompt.template = dbatu_persona_and_rules + "\n\n" + prompt.template
        # print(f"Attempted customized prompt template:\n{prompt.template}")
        # For safety and functionality, using the default react prompt without direct modification here.
        # The persona should be primarily enforced by the system message part of the React prompt.
        # The 'hub.pull("hwchase17/react-chat")' might be better as it's designed for chat.

    except Exception as e:
        print(f"Error pulling react prompt from hub, using a basic fallback. Error: {e}")
        # Basic fallback ReAct prompt template
        template = """Answer the following questions as best you can. You have access to the following tools: {tools}. Use the following format: Question: the input question you must answer. Thought: you should always think about what to do. Action: the action to take, should be one of [{tool_names}]. Action Input: the input to the action. Observation: the result of the action. ... (this Thought/Action/Action Input/Observation can repeat N times). Thought: I now know the final answer. Final Answer: the final answer to the original input question. Begin! Question: {input} Thought:{agent_scratchpad}"""
        prompt = PromptTemplate.from_template(template)
        # Manually ensure input_variables are what create_react_agent expects if not using hub.pull
        prompt.input_variables = ["input", "tools", "tool_names", "agent_scratchpad"]


    agent = create_react_agent(llm, tools, prompt)
    # handle_parsing_errors=True can help with robustness if the LLM outputs poorly formatted action requests.
    # max_iterations can prevent excessively long loops.
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax", # More descriptive error
        max_iterations=10, # Set a reasonable max iterations
        early_stopping_method="generate" # if LLM generates a Final Answer
    )
    
    return agent_executor

# User input function
def user_input(user_question):
    # Environment variable for Serper API Key
    if not SERPER_API_KEY: # Check again, as it's crucial for the search tool
        print("CRITICAL WARNING: SERPER_API_KEY is not set. Web search functionality will be severely limited or disabled.")
        # Optionally, inform the user directly in the response if search is unavailable.

    # Check if FAISS DB exists, otherwise internal search won't work.
    if not os.path.exists(DB_FAISS_PATH) or not os.listdir(DB_FAISS_PATH): # Check if directory is empty too
        print(f"CRITICAL WARNING: FAISS database at {DB_FAISS_PATH} is missing or empty. DBATUInternalKnowledgeSearch tool will not function correctly. Please run app_embeddings.py.")
        # Consider returning a message indicating that internal search is unavailable.

    agent_executor = create_dbatu_agent_executor()
    
    try:
        response = agent_executor.invoke({"input": user_question})
        # The actual response is usually in 'output' key
        final_answer = response.get("output", "I am sorry, I could not find an answer to your question.")
        return {"output_text": final_answer}
    except Exception as e:
        print(f"Error during agent execution: {e}")
        # Provide a more user-friendly error message
        return {"output_text": "I apologize, but I encountered an unexpected issue while trying to process your request. Please try again later."}

# The old custom_prompt_template, set_custom_prompt, get_conversational_chain, 
# and the old user_input logic have been removed as the agent structure replaces them.
# Ensure any necessary helper text or suggestions are integrated into the agent's prompt or system design.
