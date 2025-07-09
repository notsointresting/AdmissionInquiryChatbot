from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain import hub
import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv('API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("API_KEY (for Google Generative AI) not found in environment variables.")
if not SERPER_API_KEY:
    print("Warning: SERPER_API_KEY not found in environment variables. Web search tool will not be available.")

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
os.environ['SERPER_API_KEY'] = SERPER_API_KEY

DB_FAISS_PATH = 'vectorstore/db_faiss'

CHAT_HISTORY_MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.5,
    )
    return llm

def create_dbatu_agent_executor():
    llm = load_llm()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    if not os.path.exists(DB_FAISS_PATH):
        print(f"CRITICAL ERROR: FAISS database not found at {DB_FAISS_PATH}. DBATUDocumentSearch tool cannot function.")
        db = None
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
9.  **CRITICAL ReAct Output Format**: ALL your responses MUST strictly follow the ReAct output format. This means you MUST output either a `Thought:` followed by an `Action:` and `Action Input:`, OR a `Thought:` followed by `Final Answer:`. This applies even for direct greetings, simple statements, or when you are declining to answer a non-DBATU related question. For example:
    *   If the user says "Hi", you might respond:
        Thought: The user greeted me. I should respond with a polite greeting and offer assistance according to Rule #4.
        Final Answer: Hello! How can I assist you with DBATU today?
    *   If the user asks an off-topic question, you might respond:
        Thought: The user's question is not related to DBATU. According to Rule #3, I must decline and restate my specialization.
        Final Answer: I specialize in information about Dr. Babasaheb Ambedkar Technological University. How can I help you with a DBATU-related question?
"""

    prompt_from_hub = None
    customization_successful = False        try:
            pulled_prompt = hub.pull("hwchase17/react-chat")

        if isinstance(pulled_prompt, ChatPromptTemplate) and pulled_prompt.messages:
            if isinstance(pulled_prompt.messages[0], SystemMessage) or \
               (hasattr(pulled_prompt.messages[0], 'prompt') and isinstance(pulled_prompt.messages[0].prompt, SystemMessage)):
                pulled_prompt.messages[0] = SystemMessage(content=dbatu_persona_and_rules)
            else:
                pulled_prompt.messages.insert(0, SystemMessage(content=dbatu_persona_and_rules))
            
            prompt_from_hub = pulled_prompt
            customization_successful = True
        else:
            print("Pulled prompt from hub is not a ChatPromptTemplate with messages, or messages list is empty.")

    except Exception as e:
        print(f"Error during hub.pull or initial prompt customization: {e}. Will use basic fallback.")

    if customization_successful and prompt_from_hub:
        prompt = prompt_from_hub
    else:
        print("Using basic fallback prompt template.")
        fallback_template = dbatu_persona_and_rules + """

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.

Previous conversation:
{chat_history}

New human question: {input}
Begin!
{agent_scratchpad}"""
        prompt = PromptTemplate.from_template(fallback_template)

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=CHAT_HISTORY_MEMORY,
        verbose=True,
        handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax. If you think you have the final answer, make sure to output 'Final Answer:'.",
        max_iterations=10
    )

    return agent_executor

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

    agent_executor = get_agent_executor()

    try:
        response = agent_executor.invoke({"input": user_question})
        final_answer = response.get("output", "I am sorry, I could not find an answer to your question.")

        return {"output_text": final_answer, "chat_history": CHAT_HISTORY_MEMORY.chat_memory.messages}
    except Exception as e:
        print(f"Error during agent execution: {e}")
        import traceback
        traceback.print_exc()
        return {"output_text": "I apologize, but I encountered an unexpected issue while trying to process your request. Please try again later.", "chat_history": CHAT_HISTORY_MEMORY.chat_memory.messages}

# Example usage (for testing purposes, can be removed or commented out)
if __name__ == '__main__':
    print("DBATU Chatbot Initialized. Type 'quit' to exit.")

    while True:
        query = input("You: ")
        if query.lower() == 'quit':
            break
        response_data = user_input(query)
        print("Bot:", response_data["output_text"])
