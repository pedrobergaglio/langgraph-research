import operator #operator.add(x, y) is equivalent to the expression x+y
from pydantic import BaseModel, Field, create_model, model_validator
from typing import Annotated, List, TypedDict, Literal, Optional, Type, Any
from typing_extensions import TypedDict # just a dict with types
from langchain_community.document_loaders import WikipediaLoader # for queries on wikipedia
from langchain_community.tools.tavily_search import TavilySearchResults # for web queries
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string # message types
from langchain_openai import ChatOpenAI # to call openai
from langgraph.constants import Send # to call any number of parallel nodes
from langgraph.graph import END, MessagesState, START, StateGraph # esentials for graphs
from langgraph.errors import NodeInterrupt
import json
import utils.functions as fn
from utils.procedures import Action, Procedure, ToyStoredProceduresDB

### LLM

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 

### General prompts

general_instructions = f"""Sos un asistente virtual que ayuda a los empleados a usar el software de la empresa.
Mantené las respuestas claras y profesionales, pero con alta onda. Concentrate en completar tareas específicas de manera eficiente.
IMPORTANTE: Todos los mensajes tienen que ser en español (Argentino) - usá un lenguaje informal pero profesional común en Argentina, 
tuteando y usando modismos argentinos apropiados para un ambiente laboral.

"""

### Schema

class messageClass(BaseModel):
    content:str

class ActionsDB(BaseModel):
    actions:List[Action]

class ProcedureCreation(BaseModel):
    actions:List[int]
    message:str
    description:str

class BestProcedure(BaseModel):
    id: int = Field("the index of the selected procedure in the list of procedures")
    message:str

""" class OrchestratorState(TypedDict):
    procedure:Procedure # TODO let the orchestrator edit the retrieved procedure based on human request
    request:str #initial request
    human_feedback_select_message:str #later feedback TODO load the entire conversation, not just the last update
    stored_procedures:ToyStoredProceduresDB
 """

""" class ExecutorState(MessagesState): 
    #we use messages to store the questions from the agent and answers from the user
    procedure:Procedure
    done_actions:List[Action] = [] #to store the actions done by the executor
 """

class GeneralState(MessagesState): #this is the general state with the general data needed.
    request:str
    stored_procedures:ToyStoredProceduresDB
    human_feedback_select_message:str
    human_feedback_action_message: Optional[Literal['yes', 'no']] = None
    procedure:Procedure
    pending_actions:List[Action]
    done_actions: List[Action]
    actions_database:ActionsDB
    data: Optional[dict[str, Any]] = None
    vector_index: Any
    options: Optional[List[str]] = None

""" class CreatorState(MessagesState):
    actions:List[Action] #to store the actions to create the procedure
    stored_procedures:ToyStoredProceduresDB
    actions_database:ActionsDB
    procedure_description:str
    human_feedback_select_message:str
 """

class CreateSelect(BaseModel):
    selection: Literal["create_procedure", "select_procedure"]

### Nodes and edges

orchestrator_instructions=general_instructions+"""You are tasked to select one of the stored procedures based on the user request. 
Follow these instructions carefully:

1. First, review the user request and feedback in the conversation.
        
2. Examine each of the stored procedures that have been provided: 
        
{stored_procedures}
    
3. Determine the most accurate procedure based upon information above.
                    
4. Return the ID of the selected one, starting 0 as the ID for the first procedure, 
and a one sentence message to the user to continue the conversation and informing the selected procedure and why."""

def select_procedure(state: GeneralState):
    
    """ Select procedure to do """
    
    #get data from the orchestrator state
    request = state["request"]
    human_feedback_select_message = state.get('human_feedback_select_message', '') #this can be none
    stored_procedures = state["stored_procedures"]
    messages = state.get('messages', [])

    # the message is the feedback if there is not the first user message
    if request == "" or request == None:
        message = human_feedback_select_message
    # else the message is the user first request
    else:
        message = request
    
    
    # Enforce structured output to get the index of the procedure
    structured_llm = llm.with_structured_output(BestProcedure) 
    
    # parsing the System message
    system_message = orchestrator_instructions.format(stored_procedures=stored_procedures)

    # select procedure
    procedure_id = structured_llm.invoke([SystemMessage(content=system_message)]+messages+[HumanMessage(content=message)])

    #raise NodeInterrupt({type(stored_procedures)})

    procedure = stored_procedures.procedures[procedure_id.id]

    messages.append(HumanMessage(content=message))
    messages.append(AIMessage(content=procedure_id.message))
    
    # Write the procedure to state
    return {"stored_procedures": stored_procedures,
            "procedure": procedure, 
            "pending_actions": procedure.actions, 
            "human_feedback_select_message": "approve", #we return the selected procedure and reset the feedback
            "request": '',
            "messages": messages
            } 

# un punto concreto donde se para, para poder recibir feedback después mediante otro metodo
def human_feedback_select(state: GeneralState): 
    """ No-op node that should be interrupted on """
    pass

def human_feedback_action(state: GeneralState): 
    """ No-op node that should be interrupted on """
    pass

def human_feedback_create(state: GeneralState):
    pass

# Executor workflow

# for each analyst: ask questions to drill down and refine your understanding of the topic.
question_instructions = general_instructions+"""You are an executor tasked with execute an specific procedure of actions. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

def filter_required_fields(schema: dict) -> dict:
    def is_optional(field_schema):
        return (isinstance(field_schema.get("type"), list) and 
                "null" in field_schema["type"])
    
    def filter_field(field_schema, depth=0):
        if not isinstance(field_schema, dict) or depth > 3:
            return None
            
        filtered_schema = field_schema.copy()
        
        # Check if field is optional
        if is_optional(field_schema):
            return None
            
        # Handle array items
        if "items" in field_schema:
            filtered_items = filter_field(field_schema["items"], depth + 1)
            if filtered_items is None:
                return None
            filtered_schema["items"] = filtered_items
            filtered_schema["type"] = "array"
            
        # Handle object properties
        if "properties" in field_schema:
            filtered_props = {}
            for prop_name, prop_schema in field_schema["properties"].items():
                filtered_prop = filter_field(prop_schema, depth + 1)
                if filtered_prop is not None:
                    filtered_props[prop_name] = filtered_prop
            
            if not filtered_props:
                return None
                
            filtered_schema["properties"] = filtered_props
            
            if "required" in field_schema:
                required = [r for r in field_schema["required"] 
                          if r in filtered_props]
                if required:
                    filtered_schema["required"] = required
                
        return filtered_schema

    filtered = filter_field(schema)
    return filtered if filtered is not None else {"type": "object", "properties": {}}
# here we are already using the state for the executor, completely uncoupled from the other states.
# here the executor executes the selected procedure
def executor(state: GeneralState):

    """ Node to execute the procedure """

    # Get state
    done_actions = state.get("done_actions", [])
    human_feedback_action_message = state.get('human_feedback_action_message', '')
    actions = state.get("pending_actions", [])
    messages = state.get('messages', [])
    human_feedback_select_message = state.get('human_feedback_select_message', '')
    data = state.get('data', {})
    # just to add the confirmation message to the messages
    if human_feedback_select_message.lower() == "approve":
        messages.append(HumanMessage(content="I confirm the procedure"))
        human_feedback_select_message = ""

    if human_feedback_select_message.lower() != "":
        #raise NodeInterrupt(f"IN human_feedback_select_message {human_feedback_select_message}")
        messages.append(HumanMessage(content=human_feedback_select_message))
    
    for action in actions:

        # If the action has no params but has a params schema, we need to fill them
        if (not action.params and action.params_schema) or (action.params and human_feedback_select_message.lower() != ""):

            for i in range(5):
            
                # Filter schema to only required fields
                required_schema = filter_required_fields(action.params_schema)

                can_fill_schema = {
                    "title": "CanFillResponse", 
                    "type": "object",
                    "properties": {
                        "can_fill": {
                            "type": "string",
                            "enum": ["false", "true", "CLIENTES", "PRODUCTS"]
                        },
                        "text": {
                            "type": "string",
                            "description": "Message to display to user, or search query"
                        }
                    },
                    "required": ["can_fill", "text"]
                }

                # First request: Can we fill the params?
                decision_llm = llm.with_structured_output(
                    can_fill_schema,
                    method="json_schema", 
                    strict=True
                )
                
                can_fill = decision_llm.invoke([
                    SystemMessage(content=general_instructions+f"""
                Analyze if we can fill these parameters based on information provided in the conversation, 
                or if the user has provided new data in the last message.

                RULES FOR REQUIRED FIELDS:
                1. Only use data explicitly provided in the conversation 
                2. Required fields must have actual values from conversation
                3. IDs from CLIENTES or PRODUCTS tables must be actual IDs, not names/descriptions
                4. List any missing fields when asking user
                5. BEFORE CALLING THE ID AND PRODUCT SEARCH, MAKE SURE THE USER PROVIDED ALL THE OTHER REQUIRED FIELDS

                ---               

                Response options:
                - If all parameters can be filled with data from conversation:
                RULES FOR REQUIRED FIELDS:
                can_fill: true 
                message: List the exact values you'll use
                                  
                - If any other parameter is missing:
                can_fill: false
                message: List the missing required fields in a conversational way in one sentence
                
                - If  client ID is not in chat history and you still need to lookup a client ID:
                can_fill: CLIENTES
                message: Write a short query focused on name/document/business to search the vector index, it's not a message for the user.
                Example: "Jorge Botta" or "CUIT 20123456789"
                Do NOT include any extra text - just the search terms.
                
                - If product ID is not in chat history and you still need to lookup a product ID:
                can_fill: PRODUCTS  
                message: Write a short product query to search the vector index, it's not a message for the user.
                Example: "Rueda 14" or "Tornillo M8"
                Do NOT include any extra text - just the search terms.

                Schema: {required_schema}

                ---

                BEFORE REPLY, ANALYZE THE FOLLOWING QUESTIONS:

                - DID THE USER PROVIDE ALL THE REQUIRED FIELDS? 
                IF NOT, RETURN FALSE, WHAT ARE ALL THE VALUES THAT ARE MISSING?

                - DO YOU HAVE THE CLIENT ID IN THE CHAT HISTORY?
                IF NOT, DO YOU HAVE THE CLIENT NAME? IF SO, RETURN CLIENTES AND THE CLIENT NAME

                - DO YOU HAVE THE PRODUCT ID IN THE CHAT HISTORY?
                IF NOT, DO YOU HAVE THE PRODUCT NAME? IF SO, RETURN PRODUCTS AND THE PRODUCT NAME

                - DID THE USER PROVIDE UPDATES TO THE DATA IN THE LAST MESSAGE ABOUT THE PREVIOUSLY FILLED PARAMS?
                IF SO, USE THAT DATA TO FILL AGAIN THE PARAMS, SELECTING TRUE. 


                """)
                ] + messages)

                # If we can fill the params, we need to fill them
                if can_fill["can_fill"] == "true":

                    # Fill the params
                    params_llm = llm.with_structured_output(
                        action.params_schema,
                        method="json_schema",
                        strict=True
                    )
                    
                    filled_params = params_llm.invoke([
                        SystemMessage(content=general_instructions+"Fill the parameters based on the conversation context")
                    ] + messages)

                    # log the filled params for the llms to have context:
                    message = AIMessage(content=f"Filled params: {filled_params}")
                    message.name = 'log'
                    messages.append(message)
                    
                    action.params = filled_params
                    data.update(filled_params)
                    # Also pass the action metadata to the data
                    if action.metadata:
                        data.update(action.metadata)
                    
                    # If we have all the params, we can break the loop
                    break

                elif can_fill["can_fill"] == "CLIENTES":
                    vector_index_dict = state.get("vector_index", {})
                    if not vector_index_dict:
                        raise ValueError("Vector index not found in state")

                    test_retriever = vector_index_dict["CLIENTES"].as_retriever(similarity_top_k=5)
                    nodes = test_retriever.retrieve(can_fill["text"])
                    
                    # Define schema for client matching results
                    client_match_schema = {
                        "title": "ClientMatch",
                        "type": "object",
                        "properties": {
                            "matches": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "CLIENTE": {"type": "string"},
                                        "DNI_CUIT": {"type": "string"},
                                        "ID": {"type": "string"}
                                    },
                                    "required": ["CLIENTE", "DNI_CUIT", "ID"]
                                }
                            }
                        },
                        "required": ["matches"]
                    }

                    # Get matches using structured LLM with schema
                    client_match_llm = llm.with_structured_output(client_match_schema, method="json_schema")

                    matches_dict = client_match_llm.invoke([
                        SystemMessage(content=f"""You are a client matching assistant.
                        Found these client records:
                        {[n.metadata for n in nodes]}
                        
                        Query: {can_fill["text"]}

                        RETURN EXACT JSON FORMAT:
                        {{
                            "matches": [
                                {{
                                    "CLIENTE": "Client name",
                                    "DNI_CUIT": "299103940",
                                    "ID": "Client ID"
                                }}
                            ]
                        }}

                        RULES:
                        - Match exact field names (CLIENTE, DNI_CUIT)
                        - Include all relevant matches
                        - Empty matches should be []
                        """)
                    ])
                    

                    # No matches found 
                    if len(matches_dict["matches"]) == 0:
                        response = llm.invoke([SystemMessage(content=general_instructions+f"Send a one sentence message for the user, telling that couldnt found {can_fill['text']} in the database")]+messages)
                        messages.append(AIMessage(content=response.content))
                        return {
                                "messages": messages, 
                                "human_feedback_action_message": "",
                                "human_feedback_select_message": "",
                                "done_actions": done_actions,
                                "pending_actions": actions,
                                "data": data,
                        "options": None,
                        }

                    # Single exact match found
                    elif len(matches_dict["matches"]) == 1:
                        match = matches_dict["matches"][0]
                        message = AIMessage(content=f"CLIENT ID READY TO SEND, THERE'S NO NEED FOR ANOTHER CLIENTES SEARCH FOUND: CLIENT ID:{match['ID']}, name: {match['CLIENTE']}")
                        message.name = 'log'
                        messages.append(message)

                    # Multiple potential matches found
                    else:
                        llm_str = llm.with_structured_output(messageClass)
                        response = llm_str.invoke([SystemMessage(content=general_instructions+"Send a one sentence message to the user explaining that multiple clients were found and asking them to select the correct one using the buttons below")]+messages)
                        messages.append(AIMessage(content=response.content))
                        return {
                            "messages": messages,
                            "human_feedback_action_message": "",
                            "human_feedback_select_message": "", 
                            "done_actions": done_actions,
                            "pending_actions": actions,
                            "data": data,
                            "options": [
                                f"{m['CLIENTE']} CUIT: {m['DNI_CUIT']}" 
                                for m in matches_dict["matches"]
                            ]
                        }

                elif can_fill["can_fill"] == "PRODUCTS":
                    vector_index_dict = state.get("vector_index", {})
                    if not vector_index_dict:
                        raise ValueError("Vector index not found in state")

                    test_retriever = vector_index_dict["PRODUCTS"].as_retriever(similarity_top_k=5)
                    nodes = test_retriever.retrieve(can_fill["text"])
                    
                    # Define schema for product matching results
                    product_match_schema = {
                        "title": "ProductMatch",
                        "type": "object",
                        "properties": {
                            "matches": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "product": {"type": "string"},
                                        "id": {"type": "string"},
                                        "supplier": {"type": "string"},
                                        "brand": {"type": "string"}
                                    },
                                    "required": ["product", "id", "supplier", "brand"]
                                }
                            }
                        },
                        "required": ["matches"]
                    }

                    # Get matches using structured LLM with schema
                    product_match_llm = llm.with_structured_output(product_match_schema, method="json_schema")

                    matches_dict = product_match_llm.invoke([
                        SystemMessage(content=f"""You are a product matching assistant.
                        Found these product records:
                        {[n.metadata for n in nodes]}
                        
                        Query: {can_fill["text"]}

                        RETURN EXACT JSON FORMAT:
                        {{
                            "matches": [
                                {{
                                    "product": "Product name",
                                    "id": "Product ID",
                                    "supplier": "Supplier name",
                                    "brand": "Brand name"
                                }}
                            ]
                        }}

                        RULES:
                        - Match exact field names
                        - Include ALL relevant matches. If you discard any relevant, the system will break. 
                        - If you are in doubt about a record, just add it so the user can decide.
                        - Empty matches should be []
                        """)
                    ])

                    # No matches found 
                    if len(matches_dict["matches"]) == 0:
                        response = llm.invoke([SystemMessage(content=general_instructions+f"Send a one sentence message for the user, telling that couldn't find {can_fill['text']} in the database")]+messages)
                        messages.append(AIMessage(content=response.content))
                        return {
                                "messages": messages, 
                                "human_feedback_action_message": "",
                                "human_feedback_select_message": "",
                                "done_actions": done_actions,
                                "pending_actions": actions,
                                "data": data,
                                "options": None,
                        }

                    # Single exact match found
                    elif len(matches_dict["matches"]) == 1:
                        match = matches_dict["matches"][0]
                        message = AIMessage(content=f"PRODUCT ID, THERE'S NO NEED TO MAKE ANOTHER PRODUCTS SEARCH found: PRODUCT ID: {match['id']}, name: {match['product']}")
                        message.name = 'log'
                        messages.append(message)

                    # Multiple potential matches found
                    else:
                        llm_str = llm.with_structured_output(messageClass)
                        response = llm_str.invoke([SystemMessage(content=general_instructions+"Send a one sentence message to the user explaining that multiple products were found and asking them to select the correct one using the buttons below, and ask the user to give more feedback if none of those are correct")]+messages)
                        messages.append(AIMessage(content=response.content))

                        options = [
                            f"PRODUCT NAME: {m['product']} (PRODUCT ID: {m['id']})" 
                            for m in matches_dict["matches"]
                        ]

                        # log the options for the llms to have context:
                        message = AIMessage(content=f"Options PRODUCT NAME (PRODUCT ID): {options}")
                        message.name = 'log'
                        messages.append(message)

                        return {
                            "messages": messages,
                            "human_feedback_select_message": "", 
                            "done_actions": done_actions,
                            "pending_actions": actions,
                            "data": data,
                            "options": [
                                f"{m['product']} ({m['brand']}) - ID: {m['id']}" 
                                for m in matches_dict["matches"]
                            ]
                        }

                # If we can't fill the params, we need to ask the user  
                else:
                    messages.append(AIMessage(content=can_fill["text"]))
                    return {
                            "messages": messages,
                            "human_feedback_action_message": "",
                            "human_feedback_select_message": "",
                            "done_actions": done_actions, 
                            "pending_actions": actions,
                            "data": data,
                        "options": None,
                            }

        # Once we have the params, check action confirmation and execute
        confirmed = action.confirmed
        if human_feedback_action_message.lower() == 'yes':
            messages.append(HumanMessage(content="I confirm the action"))
            confirmed = True
        if human_feedback_action_message.lower() == 'no':
            messages.append(HumanMessage(content="I don't confirm the action, end this conversation."))
            return {
                    "messages": messages,
                    "human_feedback_action_message": "",
                    "human_feedback_select_message": "",
                    "done_actions": done_actions, 
                    "pending_actions": [],
                    "data": data
                    }

        if confirmed:
            # There is a function to execute
            if action.function and action.function != "":
                try:
                    # Execute the function
                    result = getattr(fn, action.function)(data)
                    # Ensure result is valid JSON
                    if not isinstance(result, dict):
                        raise ValueError("Function must return a dictionary")
                    data.update(result)
                    
                    if result.get("status") != "error":
                        message = AIMessage(content=f"The action {action.name} was executed successfully")
                        message.name = 'log'
                        messages.append(message)
                        #llm_str = llm.with_structured_output(messageClass)
                        #response = llm_str.invoke([SystemMessage(content=general_instructions+f"Send a one sentence message to the user to inform that the {action.name} was executed successfully")])
                        #messages.append(AIMessage(content=response.content))
                        done_actions.append(action)
                        actions = actions[1:]
                    else:
                        message = AIMessage(content=f"Error executing action {action.name}: {result.get('response')}, data sent: {data}")
                        message.name = 'log'
                        messages.append(message)
                        llm_str = llm.with_structured_output(messageClass)
                        response = llm_str.invoke([SystemMessage(content=general_instructions+"Send a one sentence message to the user to inform about the error and suggest to try again")])
                        messages.append(AIMessage(content=response.content))
                        return {
                        "messages": messages,
                        "human_feedback_action_message": "",
                        "human_feedback_select_message": "",
                        "done_actions": done_actions, 
                        "pending_actions": actions,
                        "data": data,
                    "options": None,
                    }
                        
                except Exception as e:
                    messages.append(AIMessage(content=f"Error during execution: {str(e)}", name="log"))
                    llm_str = llm.with_structured_output(messageClass)
                    response = llm_str.invoke([SystemMessage(content=general_instructions+"Send a one sentence message to the user to inform about the error and suggest to try again")])
                    messages.append(AIMessage(content=response.content))
                    return {
                        "messages": messages,
                        "human_feedback_action_message": "",
                        "human_feedback_select_message": "",
                        "done_actions": done_actions, 
                        "pending_actions": actions,
                        "data": data,
                    "options": None,
                    }
            
            # There is no function to execute
            else:
                message = AIMessage(content=f"The action {action.name} was executed successfully")
                message.name = 'log'
                messages.append(message)
                done_actions.append(action)
                actions = actions[1:]
    
        # If the action is not confirmed, we need to ask the user
        else:
            message = AIMessage(content=f"The action '{action.name}' needs your confirmation")
            message.name = 'log'
            messages.append(message)
            llm_str = llm.with_structured_output(messageClass)
            response = llm_str.invoke([SystemMessage(content=general_instructions+f"""Send a message to the user asking for confirmation to execute '{action.name}'.
            Generate a brief overview of the action
            1. Action name: {action.name}
            2. Parameters that will be used: {json.dumps(action.params, indent=2) if action.params else 'No parameters'}
            Format this nicely and ask if they want to proceed.""")]+messages)
            messages.append(AIMessage(content=response.content))
            return {
                    "messages": messages,
                    "human_feedback_action_message": "",
                    "human_feedback_select_message": "",
                    "done_actions": done_actions, 
                    "pending_actions": actions,
                    "data": data,
                    "options": None,
                    }


    # If all actions are done, we can finish
    return {
                        "messages": messages,
                        "human_feedback_action_message": "",
                        "human_feedback_select_message": "",
                        "options": None,
                        "done_actions": done_actions, 
                        "pending_actions": actions,
                        }

# prompt for the procedure creator
creator_instructions=general_instructions+"""You are tasked to create an ordered list of actions (called stored procedure) based on the user request, 
and with the actions database you are provided. 

Follow these instructions carefully:

1. First, review the user request in the conversation.
        
2. Examine each of the actions that have been provided: 
        
{actions_database}

3. Consider the last user feedback in the conversation.
    
4. Determine the most accurate procedure to create based upon context provided, always using actions present in the database.
                    
5. Return
    a. An array of actions id's to create the procedure
    b. A 1 sentence message for the user to continnue the conversation. Explain why you selected the actions you did.
    c. The procedure description to be stored in the database for latter use.
"""

def create_procedure(state: GeneralState):

    actions_database = state["actions_database"]
    messages = state["messages"]
    human_feedback_select_message = state.get('human_feedback_select_message', '')

    if human_feedback_select_message != "":
        messages.append(HumanMessage(content=human_feedback_select_message))

    # Enforce structured output 
    structured_llm = llm.with_structured_output(ProcedureCreation) 
    
    # parsing the System message
    system_message = creator_instructions.format(actions_database=actions_database.actions)

    # select procedure
    procedure_creation = structured_llm.invoke([SystemMessage(content=system_message)]+messages)
    actions_id = procedure_creation.actions

    procedure_actions = []

    for action in actions_id:
        procedure_actions.append(actions_database.actions[action])
    
    description = procedure_creation.description
    
    messages.append(AIMessage(content=procedure_creation.message))

    # Write the procedure to state
    return {
        "actions": procedure_actions, 
        "messages": messages,
        "procedure_description": description,
        "human_feedback_select_message": ""
    }
    
# node to initialize or loop the conversation
def start(state: GeneralState):

    actions_database = state.get('actions_database', {})
    stored_procedures = state.get('stored_procedures', {})
    messages = state.get('messages', [])

    if actions_database != {} or stored_procedures != {}:
        llm_str = llm.with_structured_output(messageClass)
        
        response = llm_str.invoke([SystemMessage(content=general_instructions+"Send a one sentence message to the user to conclude the last conversation and tell that you are available to do another task")]+messages)
        messages.append(AIMessage(content=response.content))
        return {"messages": messages}

    else: 
        
        vector_index_dict = fn.index_from_storage()
        
        return {"vector_index": vector_index_dict,
            "stored_procedures": ToyStoredProceduresDB(procedures=[
        
        Procedure(
        description="PRODUCTION READY. Create new estimante/order in system",
        actions=[
            Action(
            id=13, 
            name="save the estimate/order info, without sending it into the system",
            type="erp",
            confirmed=False,
            params_schema={
            "title": "Order",
            "type": "object",
            "properties": {
            "ID_CLIENTE": {
                "type": "string",
                "description": "Client ID"
            },
            "TIPO_DE_ENTREGA": {
                "type": "string",
                "enum": ["CLIENTE", "RETIRA EN FÁBRICA", "OTRO"],
                "description": "Delivery type"
            },
            "VENDEDOR": {
                "type": "string",
                "enum": ["ESTEBAN", "FLORENCIA", "NICOLAS", "PATRICIA", "FEDERICO", "Prueba"],
                "description": "The user that is talking to you, ask his name, there's no default"
            },
            "OPCIONES": {
                "type": "array",
                "items": {
                "type": "object",
                "properties": {
                "MONEDA": {
                "type": "string",
                "enum": ["PESO", "DOLAR"],
                "description": "Currency. Salesperson must define which is the currency to use, There is no default"
                },
                "PRODUCTOS_PEDIDOS": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "ID_PRODUCTO": {
                    "type": "integer",
                    "description": "Product ID"
                    },
                    "CANTIDAD": {
                    "type": "integer",
                    "description": "Quantity"
                    },
                    "PRECIO_LISTA_S_IVA": {
                    "type": ["number", "null"],
                    "description": "Optional list price without VAT, if not provided, it must be left as null because the system itself will calculate it"
                    }
                    },
                    "required": ["ID_PRODUCTO", "CANTIDAD", "PRECIO_LISTA_S_IVA"]
                }
                },
                "DESCUENTO_GENERAL": {
                    "type": ["number", "null"],
                    "description": "General discount"
                }
                },
                "required": ["MONEDA", "PRODUCTOS_PEDIDOS", "DESCUENTO_GENERAL"]
                }
            },
            "CARGAR_COMO_PEDIDO": {
                "type": ["boolean", "null"],
                "description": "Defaults to false, because this is actually an estimate, not a confirmed order"
            }
            },
            "required": ["ID_CLIENTE", "TIPO_DE_ENTREGA", "VENDEDOR", "OPCIONES", "CARGAR_COMO_PEDIDO"]
            },
            metadata={"table_name": "PEDIDOS"},
            ),
            Action(
            id=14,
            name="add products to the order",
            type="erp",
            confirmed=True,
            params_schema={
            "title": "OrderProducts",
            "type": "object",
            "properties": {
            "products": {
                "type": "array",
                "items": {
                "type": "object",
                "properties": {
                "ID_PRODUCTO": {
                "type": "integer",
                "title": "Product ID found in the database",
                },
                "TIPO": {
                "type": "string",
                "enum": ["ESTANDAR", "INTENSO", "PIEDRA GRANITO"],
                "title": "Type of the color, each color has a corresponding type in the database"
                },
                "COLOR": {
                "type": "string",
                "title": "Color of the product (TYPE GOES IN THE OTHER FIELD), in the database it is a foreign key",
                },
                "CANTIDAD": {
                "type": "integer",
                "title": "Quantity"
                }
                },
                "required": ["ID_PRODUCTO", "TIPO", "COLOR", "CANTIDAD"]
                }
            },
            },
            "required": ["products"]
            },
            #metadata={"table_name": "PRODUCTOS_PEDIDOS"},
            function="add_products_to_order",),
            Action(
            id=15,
            name="save order",
            type="erp",
            confirmed=False,
            function="save_order"
            )
            ])],
        
         ),
        "human_feedback_select_message": "approve"
            }

""" Procedure(
                description="PRODUCTION READY. Create new order in system",
                actions=[
            Action(
                id=13, 
                name="send the main order info to the system",
                type="erp",
                confirmed=True,
                params_schema={
                    "title": "Order",
                    "type": "object",
                    "properties": {
                        "ID_CLIENTE": {
                            "type": "string",
                            "description": "Client ID"
                        },
                        "TIPO_DE_ENTREGA": {
                            "type": "string",
                            "enum": ["CLIENTE", "RETIRA EN FÁBRICA", "OTRO"],
                            "description": "Delivery type"
                        },
                        "METODO_DE_PAGO": {
                            "type": "string",
                            "enum": ["EFECTIVO", "DÓLARES", "MERCADO PAGO", "CHEQUE", "TRANSFERENCIA BANCARIA"],
                            "description": "Payment method"
                        },
                        "DIRECCION": {
                            "type": ["string", "null"],
                            "description": "Optional delivery address. Leave empty if not needed.",
                        },
                        "NOTA": {
                            "type": ["string", "null"],
                            "description": "Optional note. Leave empty if not needed.",
                        }
                    },
                    "required": ["ID_CLIENTE", "TIPO_DE_ENTREGA", "METODO_DE_PAGO", "DIRECCION", "NOTA"]
                },
                metadata={"table_name": "PEDIDOS"},
                function="create_order",
                ),
            Action(
                id=14,
                name="add products to the order",
                type="erp",
                confirmed=True,
                params_schema={
                    "title": "OrderProducts",
                    "type": "object",
                    "properties": {
                        "products": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "ID_PRODUCTO": {
                                        "type": "integer",
                                        "title": "Product ID found in the database",
                                    },
                                    "TIPO": {
                                        "type": "string",
                                        "enum": ["ESTANDAR", "INTENSO", "PIEDRA GRANITO"],
                                        "title": "Type of the color, each color has a corresponding type in the database"
                                    },
                                    "COLOR": {
                                        "type": "string",
                                        "title": "Color of the product (TYPE GOES IN THE OTHER FIELD), in the database it is a foreign key",
                                    },
                                    "CANTIDAD": {
                                        "type": "integer",
                                        "title": "Quantity"
                                    }
                                },
                                "required": ["ID_PRODUCTO", "TIPO", "COLOR", "CANTIDAD"]
                            }
                        },
                    },
                    "required": ["products"]
                    },
                    #metadata={"table_name": "PRODUCTOS_PEDIDOS"},
                    function="add_products_to_order",),
            Action(
                    id=15,
                    name="save order",
                    type="erp",
                    confirmed=False,
                    function="save_order"
                )
                ])
         """

def save_procedure(state: GeneralState):

    actions = state.get('actions', [])
    stored_procedures = state.get('stored_procedures', [])
    procedure_description = state.get('procedure_description', "Created procedure")

    if actions != []:
        stored_procedures.procedures.append(Procedure(description=procedure_description, actions=actions))
        return {"stored_procedures": stored_procedures}

# this start all the interviews in parallel using Send()
# only if the human approves the analysts creation
# else, it sends another interation of the creation of the analysts creation
def decide_execute_orchestrator(state: GeneralState):

    """ Conditional edge to initiate executor or return to orchestrator """    

    # Check if human feedback
    human_feedback_select_message = state.get('human_feedback_select_message','approve')
    if human_feedback_select_message.lower() != 'approve':
        # Return to create_analysts
        return "select_procedure"

    # Otherwise 
    else:
        return "executor"

def decide_execute_end(state: GeneralState):

    """ Conditional edge to initiate executor or return to orchestrator """ 

    pending_actions = state.get('pending_actions', [])
    human_feedback_action_message =state.get('human_feedback_action_message','')
    human_feedback_select_message = state.get('human_feedback_select_message', '')

    if pending_actions == []:
        return "start" # if user cancels OR all tasks are done just finish the execution
    elif human_feedback_action_message != '' or human_feedback_select_message != '':
        return "executor"
    
    # Otherwise 
    else:
        raise NodeInterrupt(f"human_feedback_select_message {human_feedback_select_message}")
        raise NodeInterrupt("There are still actions pending")
        #return "executor"

def decide_create_done(state: GeneralState):

    human_feedback_select_message = state.get('human_feedback_select_message', "")

    if human_feedback_select_message == "":
        return "save_procedure"
    
    else: 
        return "create_procedure"
    
    """ messages = state.get('messages', [])

    if messages != [] and messages[-1].content == "":
        return "save_procedure"
    
    else: return "create_procedure" """

# prompt to select wether to create a new procedure or select one
# TODO give more context of the situation for the llm in the prompt
# actually look at the procedures to figure out if we need to create a new one, or we can use a stored one
decide_create_select_instructions="""You are tasked to select one between two options: 
create a new procedure or select one of the stored procedures based on the user request. 
Based on the users request, return the corresponding procedure to use.

"""
"""
Follow these instructions carefully:

1. First, review the user request:
{request}


        
2. Examine each of the stored procedures that have been provided: 
        
{stored_procedures}

3. Consider the last user feedback if provided here:

{human_feedback_select_message}
    
4. Determine the most accurate procedure based upon information above.
                    
5. Return the ID of the selected one, starting 0 as the ID for the first procedure"""

def decide_create_select(state: GeneralState):

    return "select_procedure"
    
    #get data from the orchestrator state
    request = state["request"]


    # Enforce structured output to get the index of the procedure
    structured_llm = llm.with_structured_output(CreateSelect) 
    
    # parsing the System message
    system_message = decide_create_select_instructions#.format(request=request)

    # select create or select
    selection = structured_llm.invoke([SystemMessage(content=general_instructions+system_message)]+[HumanMessage(content=request)])

    if selection.selection == "create_procedure":
        return Send("create_procedure", {
            "actions_database": state["actions_database"],
            "stored_procedures": state["stored_procedures"],
            "messages": [HumanMessage(content=request)],
            "procedure_description": "New procedure",
            "human_feedback_select_message": ""
        })
    else:
        return selection.selection

# Add nodes and edges 
builder = StateGraph(GeneralState) #initialization of the orchestrator graph
builder.add_node("select_procedure", select_procedure)
builder.add_node("human_feedback_select", human_feedback_select) # nothing but just to stop on it
builder.add_node("human_feedback_action", human_feedback_action) # nothing but just to stop on it
builder.add_node("executor", executor)      
builder.add_node("start", start)
builder.add_node("create_procedure", create_procedure)
builder.add_node("human_feedback_create", human_feedback_create)
builder.add_node("save_procedure", save_procedure)

# Logic
builder.add_edge(START, "start")
builder.add_conditional_edges("start", decide_create_select, ["create_procedure", "select_procedure"])

builder.add_edge("create_procedure", "human_feedback_create")
builder.add_conditional_edges("human_feedback_create", decide_create_done, ["create_procedure", "save_procedure"])
builder.add_edge("save_procedure", "start")

builder.add_edge("select_procedure", "human_feedback_select")
builder.add_conditional_edges("human_feedback_select", decide_execute_orchestrator, ["select_procedure", "executor"])
builder.add_edge("executor", "human_feedback_action")
builder.add_conditional_edges("human_feedback_action", decide_execute_end, ["executor", "start"])

# Compile
graph = builder.compile(interrupt_before=['human_feedback_create'], interrupt_after=['start']) #'human_feedback_select',