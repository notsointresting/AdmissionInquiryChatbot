# Importing libs and modules
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory # Added for chat history
from langchain_core.messages import SystemMessage # Added for system message in prompt
from langchain import hub
import os
from dotenv import load_dotenv

# Setting API Keys
load_dotenv()
GOOGLE_API_KEY = os.getenv('API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("API_KEY (for Google Generative AI) not found in environment variables.")
if not SERPER_API_KEY:
    print("Warning: SERPER_API_KEY not found in environment variables. Web search tool will not be available.")

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
os.environ['SERPER_API_KEY'] = SERPER_API_KEY

# Path of vector database
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Global variable for chat history (simple approach for script-based execution)
# For a web app, this should be session-based.
CHAT_HISTORY_MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#Loading the model
def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Updated model
        temperature=0.5, # Adjusted temperature
    )
    return llm

# This function will now create and return an AgentExecutor
def create_dbatu_agent_executor(): # Removed memory argument, will use global
    llm = load_llm()

    # 1. Tool for searching DBATU's own documents (Vector DB)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    if not os.path.exists(DB_FAISS_PATH):
        print(f"CRITICAL ERROR: FAISS database not found at {DB_FAISS_PATH}. DBATUDocumentSearch tool cannot function.")
        # Return a non-functional agent or raise error, as this tool is critical
        # For now, we'll let it proceed, but this should be handled more gracefully
        db = None # Indicate DB is not available
    else:
        try:
            db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading FAISS database: {e}. DBATUInternalKnowledgeSearch tool will not function.")
            db = None


    def search_dbatu_documents(query: str) -> str:
        """
        Searches Dr. Babasaheb Ambedkar Technological University's (DBATU) document database
        for specific information related to the university, its courses, policies, etc.
        Use this tool first for any DBATU-specific questions.
        """
        if db is None:
            return "DBATU document database is currently unavailable."
        try:
            retriever = db.as_retriever()
            docs = retriever.invoke(query)
            if not docs:
                return "No specific information found in DBATU documents for your query."
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error during DBATU document search: {e}")
            return "There was an error searching the DBATU documents."

    dbatu_retriever_tool = Tool(
        name="DBATUInternalKnowledgeSearch",
        func=search_dbatu_documents,
        description="Searches Dr. Babasaheb Ambedkar Technological University's (DBATU) internal knowledge base for specific information. Use this for questions about university policies, course details, admission criteria, internal circulars, etc. This should be your primary tool for DBATU-specific queries."
    )

    # 2. Tool for Serper Web Search
    serper_search_tool = None
    if SERPER_API_KEY:
        try:
            search_wrapper = GoogleSerperAPIWrapper(serpapi_api_key=SERPER_API_KEY)
            serper_search_tool = Tool(
                name="WebSearch",
                func=search_wrapper.run,
                description="Performs a web search for general information, current events, or topics not covered by DBATU's internal knowledge base. Use this ONLY if DBATUInternalKnowledgeSearch doesn't yield results or for broader context AND the question is related to DBATU. Do not use for non-DBATU questions."
            )
        except Exception as e:
            print(f"Error initializing Serper search tool: {e}. Web search will be unavailable.")
    else:
        print("SERPER_API_KEY not set. WebSearch tool is disabled.")

    tools = [dbatu_retriever_tool]
    if serper_search_tool:
        tools.append(serper_search_tool)

    if not tools:
        print("CRITICAL: No tools available for the agent.")
        # This scenario should be handled by returning a message to the user
        # or raising an exception that `user_input` can catch.

    # Enhanced Persona and Rules
    dbatu_persona_and_rules = """
**Your Role and Persona:**
You are a helpful, polite, and knowledgeable AI assistant representing Dr. Babasaheb Ambedkar Technological University (DBATU).
Your primary role is to provide detailed and accurate information to users about DBATU.
You are multilingual and can understand and respond in various Indian languages, including Marathi, in a natural and conversational manner.
Always maintain a respectful, helpful, and patient tone. Provide detailed and comprehensive answers.
Your responses should sound natural and conversational. Remember you are an official assistant for DBATU.

**Core Operational Rules:**
1.  **Chat History**: You have access to the previous conversation turns. Use this history to understand context, avoid repetition, and answer follow-up questions effectively.
2.  **Tool Usage Priority**:
    *   For any question about DBATU, ALWAYS use the `DBATUInternalKnowledgeSearch` tool first.
    *   If `DBATUInternalKnowledgeSearch` does not provide a satisfactory answer, AND the question is still related to DBATU, you MAY then use `WebSearch` for broader, up-to-date, or external information.
3.  **Strict Question Relevance**: You MUST ONLY answer questions related to Dr. Babasaheb Ambedkar Technological University (academics, admissions, campus life, facilities, official announcements, contact information, specific department details, courses, curriculum, etc.).
    *   If a question is unrelated to DBATU, politely state: "I specialize in information about Dr. Babasaheb Ambedkar Technological University. How can I help you with a DBATU-related question?" Do NOT use any tools or attempt to answer unrelated questions.
4.  **Greetings**: Only greet the user if they explicitly greet you (e.g., "Hi", "Hello"). Respond with a polite and brief greeting (e.g., "Hello! How can I assist you with DBATU today?"). Do not initiate greetings otherwise.
5.  **Providing Links**:
    *   If the `DBATUInternalKnowledgeSearch` tool returns information that includes a web link (especially to a PDF or official page), you SHOULD provide it.
    *   If specifically asked for the university's location or directions, provide the main DBATU Google Maps link: https://www.google.com/maps/search/?api=1&query=Dr.+Babasaheb+Ambedkar+Technological+University%2C+Lonere
    *   For links from `WebSearch`, only provide them if they are highly relevant, verifiable official DBATU sources, or direct answers to a document request (like a PDF timetable). Do not invent links.
6.  **Handling Insufficient Information**: If, after using all available tools appropriately, you cannot find a reliable answer, state: "I've looked through the available resources but couldn't find specific information on that topic. For the most current details, please check the official DBATU website: https://dbatu.ac.in/ or consider contacting the relevant university department." Do not speculate or make up answers.
7.  **Formatting**: Use clear formatting. Utilize bullet points (-) for lists. Use **bold text** for emphasis.
8.  **Detailed Answers**: Strive to provide comprehensive and detailed answers based on the information retrieved. If a document (like a timetable PDF) is found, state that you found it and provide the link.
"""

    # Using a chat-specific ReAct prompt from the hub
    # Ensure this prompt supports 'chat_history'
    # The hwchase17/react-chat prompt includes `chat_history` in its input_variables
    try:
        # Using a prompt that is designed for chat and ReAct agents
        prompt = hub.pull("hwchase17/react-chat") # This prompt expects 'chat_history'
        # We can customize the system message part of this pulled prompt
        # The react-chat prompt usually has a structure where you can inject a system message
        # For example, by modifying prompt.messages[0] if it's a SystemMessagePromptTemplate
        # Or by adding a new SystemMessage if the structure allows.
        # For simplicity here, we'll prepend the persona to the main template,
        # but a more robust way is to modify the prompt's message structure.
        # Let's inspect the pulled prompt's messages to see how to best inject the system message.
        # For now, we will create a new ChatPromptTemplate that includes our system message
        # and the existing messages from the hub's prompt.

        # Create a new ChatPromptTemplate with our system message and placeholders for history and input
        # The 'react-chat' prompt typically has system, human, ai, and tool messages.
        # We will define our own structure incorporating the persona and rules.
        # A common structure for ReAct agents with chat history:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=dbatu_persona_and_rules),
                MessagesPlaceholder(variable_name="chat_history"), # For Langchain's memory
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"), # For ReAct agent's thoughts and tool usage
            ]
        )
    except Exception as e:
        print(f"Error pulling or customizing chat prompt from hub, using a basic fallback. Error: {e}")
        # Fallback if hub.pull fails or if the prompt structure is not as expected
        template = dbatu_persona_and_rules + """
        Previous conversation:
        {chat_history}

        New human question: {input}
        Begin!
        {agent_scratchpad}"""
        prompt = PromptTemplate.from_template(template)
        # Manually ensure input_variables are what create_react_agent expects
        prompt.input_variables = ["input", "tools", "tool_names", "agent_scratchpad", "chat_history"]


    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=CHAT_HISTORY_MEMORY, # Pass the global memory object here
        verbose=True,
        handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax. If you think you have the final answer, make sure to output 'Final Answer:'.", # More descriptive error
        max_iterations=10,
        early_stopping_method="generate"
    )

    return agent_executor

# Global agent executor instance (to maintain memory across calls in a simple script)
# This will be initialized once.
DBATU_AGENT_EXECUTOR = None

def get_agent_executor():
    global DBATU_AGENT_EXECUTOR
    if DBATU_AGENT_EXECUTOR is None:
        DBATU_AGENT_EXECUTOR = create_dbatu_agent_executor()
    return DBATU_AGENT_EXECUTOR

# User input function
def user_input(user_question):
    if not SERPER_API_KEY:
        print("CRITICAL WARNING: SERPER_API_KEY is not set. Web search functionality will be severely limited or disabled.")
    if not os.path.exists(DB_FAISS_PATH) or not os.listdir(DB_FAISS_PATH):
        print(f"CRITICAL WARNING: FAISS database at {DB_FAISS_PATH} is missing or empty. DBATUInternalKnowledgeSearch tool will not function correctly. Please run app_embeddings.py.")
        # Potentially return a message to the user if critical tools are down.

    agent_executor = get_agent_executor() # Get the globally managed agent executor

    try:
        # The agent_executor now uses the global CHAT_HISTORY_MEMORY internally
        response = agent_executor.invoke({"input": user_question})
        final_answer = response.get("output", "I am sorry, I could not find an answer to your question.")

        # The memory is automatically updated by the AgentExecutor when configured with one.
        # No need to manually save user_question and final_answer to CHAT_HISTORY_MEMORY here
        # if it's correctly passed to and used by the AgentExecutor.

        return {"output_text": final_answer, "chat_history": CHAT_HISTORY_MEMORY.chat_memory.messages}
    except Exception as e:
        print(f"Error during agent execution: {e}")
        # Log the full error for debugging
        import traceback
        traceback.print_exc()
        return {"output_text": "I apologize, but I encountered an unexpected issue while trying to process your request. Please try again later.", "chat_history": CHAT_HISTORY_MEMORY.chat_memory.messages}

# Example usage (for testing purposes, can be removed or commented out)
if __name__ == '__main__':
    print("DBATU Chatbot Initialized. Type 'quit' to exit.")
    # print("Initial chat history:", CHAT_HISTORY_MEMORY.chat_memory.messages) # For debugging
    # # Test 1: Greeting
    # response = user_input("Hello")
    # print("Bot:", response["output_text"])
    # # print("Chat history after greeting:", response["chat_history"]) # For debugging

    # Test 2: DBATU specific question
    # response = user_input("Tell me about the engineering programs at DBATU.")
    # print("Bot:", response["output_text"])
    # # print("Chat history after DBATU question:", response["chat_history"]) # For debugging

    # Test 3: Follow-up question
    # response = user_input("What are the admission criteria for those programs?")
    # print("Bot:", response["output_text"])
    # # print("Chat history after follow-up:", response["chat_history"]) # For debugging

    # Test 4: Unrelated question
    # response = user_input("What's the weather like today?")
    # print("Bot:", response["output_text"])
    # # print("Chat history after unrelated q:", response["chat_history"]) # For debugging

    # Test 5: Request for location
    # response = user_input("Where is DBATU located? Can you give me a map link?")
    # print("Bot:", response["output_text"])

    # Test 6: Simulated PDF link request (assuming a tool could find one)
    # This requires mocking the tool's behavior or having a test document.
    # For now, we'll just ask a question that *might* lead to a PDF.
    # response = user_input("Can I get the timetable for the first year computer engineering PDF?")
    # print("Bot:", response["output_text"])

    while True:
        query = input("You: ")
        if query.lower() == 'quit':
            break
        response_data = user_input(query)
        print("Bot:", response_data["output_text"])
        # print("DEBUG Current Chat History:", CHAT_HISTORY_MEMORY.chat_memory.messages)
        # print("-" * 20)
