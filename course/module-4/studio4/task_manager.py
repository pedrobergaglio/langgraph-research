import operator #operator.add(x, y) is equivalent to the expression x+y
from pydantic import BaseModel, Field, create_model, model_validator
from typing import Annotated, List, TypedDict, Literal, Optional, Type, Any
"""Add context specific metadata to a type.
Example: Annotated[int, runtime_check.Unsigned] indicates to the
hypothetical runtime_check module that this type is an unsigned int."""

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

### LLM

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 

### Schema

class messageClass(BaseModel):
    content:str

class Action(BaseModel):
    id: int
    type: str
    name: str
    confirmed: bool = True
    params_schema: Optional[dict] = None
    params: Optional[dict] = None
    function: str = ""
    metadata: Optional[dict] = None

class CanFillResponse(BaseModel):
    can_fill: bool
    message: str

class ActionsDB(BaseModel):
    actions:List[Action]

class ProcedureCreation(BaseModel):
    actions:List[int]
    message:str
    description:str

class Procedure(BaseModel):
    description: str = Field(description="What the procedure does")
    actions: List[Action] = Field(description="Ordered list of actions that the procedure does")
    #types:List[type] to search procedures by the actions it does (gmail, bank, erp...)
    @property # we can get all the data in text for a model to use by defining this and using it like: analyst.persona
    def info(self) -> str:
        return f"Description: {self.description}\nActions: {self.actions}"

class BestProcedure(BaseModel):
    id: int = Field("the index of the selected procedure in the list of procedures")
    message:str

class ToyStoredProceduresDB(BaseModel):
    procedures:List[Procedure]

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

orchestrator_instructions="""You are tasked to select one of the stored procedures based on the user request. 
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
question_instructions = """You are an executor tasked with execute an specific procedure of actions. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

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

        # If the action has no params, we need to fill them
        if not action.params:

            # First request: Can we fill the params?
            decision_llm = llm.with_structured_output(
                CanFillResponse,
                method="json_schema",
                strict=True,
            )
            
            can_fill = decision_llm.invoke([
                SystemMessage(content=f"""
                Based on the conversation context, can you fill these parameters?
                Schema: {action.params_schema}

                Remember that the user has to provide the values for the parameters and you don't have to guess them.
                
                If yes, explain what values you'll use in one sentence.
                If no, send a message to the user explaining what information is missing.
                """)
            ] + messages)

            # If we can fill the params, we need to fill them
            if can_fill.can_fill:
                # Second request: Fill the params
                params_llm = llm.with_structured_output(
                    action.params_schema,
                    method="json_schema",
                    strict=True
                )
                
                filled_params = params_llm.invoke([
                    SystemMessage(content="Fill the parameters based on the conversation context")
                ] + messages)
                
                action.params = filled_params
                data.update(filled_params)
                # Also pass the action metadata to the data
                data.update(action.metadata)

            # If we can't fill the params, we need to ask the user  
            else:
                messages.append(AIMessage(content=can_fill.message))
                return {
                        "messages": messages,
                        "human_feedback_action_message": "",
                        "human_feedback_select_message": "",
                        "done_actions": done_actions, 
                        "pending_actions": actions,
                        "data": data,
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
                    # Print debug info
                    print(f"Data being passed to function: {data}")
                    
                    # Execute the function
                    result = getattr(fn, action.function)(data)
                    
                    # Ensure result is valid JSON
                    if not isinstance(result, dict):
                        raise ValueError("Function must return a dictionary")
                        
                    data.update(result)
                    
                    if result.get("status") != "error":
                        messages.append(AIMessage(content=f"The action {action.name} was executed successfully"))
                        done_actions.append(action)
                        actions = actions[1:]
                    else:
                        messages.append(AIMessage(content=f"Error executing action {action.name}: {result.get('response')}"))
                        
                except json.JSONDecodeError as e:
                    messages.append(AIMessage(content=f"Error parsing JSON response: {str(e)}, data sent: {data}"))
                    return {
                        "messages": messages,
                        "human_feedback_action_message": "",
                        "human_feedback_select_message": "",
                        "done_actions": done_actions, 
                        "pending_actions": actions,
                        "data": data
                    }

                except AttributeError as e:
                    messages.append(AIMessage(content=f"Error executing function {action.function}: {str(e)}"))
                    return {
                        "messages": messages,
                        "human_feedback_action_message": "",
                        "human_feedback_select_message": "",
                        "done_actions": done_actions, 
                        "pending_actions": actions,
                        "data": data
                    }
                except Exception as e:
                    messages.append(AIMessage(content=f"Error during execution: {str(e)}"))
                    return {
                        "messages": messages,
                        "human_feedback_action_message": "",
                        "human_feedback_select_message": "",
                        "done_actions": done_actions, 
                        "pending_actions": actions,
                        "data": data
                    }
            
                

            
        # If the action is not confirmed, we need to ask the user
        else:
            messages.append(AIMessage(content=f"The action {action.name} needs your confirmation"))
            return {
                    "messages": messages,
                    "human_feedback_action_message": "",
                    "human_feedback_select_message": "",
                    "done_actions": done_actions, 
                    "pending_actions": actions,
                    "data": data
                    }


    # If all actions are done, we can finish
    return {
                        "messages": messages,
                        "human_feedback_action_message": "",
                        "human_feedback_select_message": "",
                        "done_actions": done_actions, 
                        "pending_actions": actions,
                        }

# prompt for the procedure creator
creator_instructions="""You are tasked to create an ordered list of actions (called stored procedure) based on the user request, 
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
        
        response = llm_str.invoke([SystemMessage(content="Send a one sentence message to the user to conclude the last conversation and tell that you are available to do another task")]+messages)
        messages.append(AIMessage(content=response.content))
        return {"messages": messages}

    else: return {"stored_procedures": ToyStoredProceduresDB(procedures=[
        Procedure(
            description="random test procedure [DO NOT USE]",
            actions=[
                Action(
                    id=9, 
                    name="select the system url", 
                    type="erp", 
                    confirmed=True,
                    params_schema={
                        "title": "URLParams",
                        "description": "Parameters for sending an order to a URL",
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The url to send the order. Make sure it's a valid url with https://"
                            }
                        },
                        "required": ["url"]
                    },
                    metadata={"table_name": "PEDIDOS"}, 
                    function="fetch_page"
                ),
                Action(
                    id=10, 
                    name="send main order information to system", 
                    type="erp", 
                    confirmed=True,
                    params_schema={
                        "title": "URLParams",
                        "description": "Parameters for sending an order to a URL, the user has to provide it, there's no default value",
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The url to send the order, the user has to provide it, there's no default value"
                            }
                        },
                        "required": ["url"]
                    },
                    function="fetch_page"
                ),
                Action(
                    id=11, 
                    name="create option for the order", 
                    type="erp", 
                    confirmed=True,
                    params_schema={
                        "title": "URLParams",
                        "description": "Parameters for sending an order to a URL",
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The url to send the order, the user has to provide it, there's no default value"
                            }
                        },
                        "required": ["url"]
                    },
                    function="fetch_page"
                ),
                Action(
                    id=12, 
                    name="add products to the last order", 
                    type="erp", 
                    confirmed=True,
                    params_schema={
                        "title": "URLParams",
                        "description": "Parameters for sending an order to a URL",
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The url to send the order"
                            }
                        },
                        "required": ["url"]
                    },
                    function="fetch_page"
                ),
            ]
        ),
        Procedure(
        description="create new order in system",
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
                "type": "string",
                "description": "Delivery address (leave empty string ('') if not needed if the delivery type is 'RETIRA EN FÁBRICA' or 'CLIENTE')"
            },
            "NOTA": {
                "type": "string",
                "description": "Note (leave empty string ('') if not needed)"
            }
        },
        "required": ["ID_CLIENTE", "TIPO_DE_ENTREGA", "METODO_DE_PAGO", "DIRECCION", "NOTA"]
    },
    metadata={"table_name": "PEDIDOS"},
    function="appsheet_add"
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
                                    "strict": True
                                },
                                "TIPO": {
                                    "type": "string",
                                    "enum": ["ESTANDAR", "INTENSO", "PIEDRA GRANITO"],
                                    "title": "Type of the color, each color has a corresponding type in the database"
                                },
                                "COLOR": {
                                    "type": "string",
                                    "title": "Color of the product (TYPE GOES IN THE OTHER FIELD), in the database it is a foreign key",
                                    "strict": True
                                },
                                "CANTIDAD": {
                                    "type": "integer",
                                    "title": "Quantity",
                                    "strict": True
                                }
                            },
                            "required": ["ID_PRODUCTO", "TIPO", "COLOR", "CANTIDAD"]
                        }
                    },
                },
                "required": ["products"]
                },
                metadata={"table_name": "PEDIDOS"},
                function="appsheet_add",)
                ])
        ]),
        "human_feedback_select_message": "approve"
            }

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
# TODO actually look at the procedures to figure out if we need to create a new one, or we can use a stored one
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
    selection = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=request)])

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