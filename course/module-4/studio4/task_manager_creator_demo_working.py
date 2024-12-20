import operator #operator.add(x, y) is equivalent to the expression x+y
from pydantic import BaseModel, Field # base for classes 
from typing import Annotated, List, TypedDict, Literal, Optional
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

### LLM

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 

### Schema

class Action(BaseModel):
    id:int
    type:str # (gmail, bank, erp...)
    name:str
    confirmed:bool = True #if the action can be done

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

class ToyStoredProceduresDB(BaseModel):
    procedures:List[Procedure]

""" class OrchestratorState(TypedDict):
    procedure:Procedure # TODO let the orchestrator edit the retrieved procedure based on human request
    request:str #initial request
    human_feedback_select_message:str #later feedback TODO load the entire conversation, not just the last update
    stored_procedures:ToyStoredProceduresDB
 """

class ExecutorState(MessagesState): 
    #we use messages to store the questions from the agent and answers from the user
    procedure:Procedure
    actions_done:List[Action] = [] #to store the actions done by the executor

class GeneralState(TypedDict): #this is the general state with the general data needed.
    request:str
    stored_procedures:ToyStoredProceduresDB
    human_feedback_select_message:str
    human_feedback_confirm_message: Optional[Literal['yes', 'no']] = None
    procedure:Procedure
    actions_pending:List[Action]
    actions_done: List[Action]
    actions_database:ActionsDB

class CreatorState(MessagesState):
    actions:List[Action] #to store the actions to create the procedure
    stored_procedures:ToyStoredProceduresDB
    actions_database:ActionsDB
    procedure_description:str
    human_feedback_select_message:str

class CreateSelect(BaseModel):
    selection: Literal["create_procedure", "select_procedure"]

### Nodes and edges

orchestrator_instructions="""You are tasked to select one of the stored procedures based on the user request. 
Follow these instructions carefully:

1. First, review the user request:
{request}
        
2. Examine each of the stored procedures that have been provided: 
        
{stored_procedures}

3. Consider the last user feedback if provided here:

{human_feedback_select_message}
    
4. Determine the most accurate procedure based upon information above.
                    
5. Return the ID of the selected one, starting 0 as the ID for the first procedure"""

def select_procedure(state: GeneralState):
    
    """ Select procedure to do """
    
    #get data from the orchestrator state
    request = state["request"]
    human_feedback_select_message = state.get('human_feedback_select_message', '') #this can be none
    stored_procedures = state["stored_procedures"]
        
    # Enforce structured output to get the index of the procedure
    structured_llm = llm.with_structured_output(BestProcedure) 
    
    # parsing the System message
    system_message = orchestrator_instructions.format(request=request, stored_procedures=stored_procedures,
                                                            human_feedback_select_message=human_feedback_select_message)

    # select procedure
    procedure_id = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Select the correct procedure")])

    #raise NodeInterrupt({type(stored_procedures)})

    procedure = stored_procedures.procedures[procedure_id.id]
    
    # Write the procedure to state
    return {"stored_procedures": stored_procedures,
            "procedure": procedure, 
            "actions_pending": procedure.actions, 
            "human_feedback_select_message": "approve", #we return the selected procedure and reset the feedback
            } 

# un punto concreto donde se para, para poder recibir feedback después mediante otro metodo
def human_feedback_select(state: GeneralState): 
    """ No-op node that should be interrupted on """
    pass

def human_feedback_confirm(state: GeneralState): 
    """ No-op node that should be interrupted on """
    pass

def human_feedback_create(state: CreatorState):
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
    actions_done = state.get("actions_done", [])
    human_feedback_confirm_message = state.get('human_feedback_confirm_message', 'no') #this can be none
    #next_action = None
    actions = state.get("actions_pending", [])

    for action in actions:

        #next_action = action
        confirmed = action.confirmed

        if human_feedback_confirm_message == 'yes' and not confirmed:
            #human_feedback_confirm_message = 'no'
            confirmed = True
        if confirmed:
            actions_done.append(action)
            actions = actions[1:]
            #raise NodeInterrupt(f"The task {action['name']} has been completed")
            return {"actions_done": actions_done, 
            "actions_pending": actions, 
            "human_feedback_select_message": "approve", 
            "human_feedback_confirm_message": "no"} #we return the actions done and the executor state

        else:
            raise NodeInterrupt(
                f"The task {action.name} needs confirmation from the user")

        

        #next_action = None

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

def create_procedure(state: CreatorState):

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

    if actions_database != {} and stored_procedures != {}:
        pass

    else: return {"stored_procedures": ToyStoredProceduresDB(procedures=[
        Procedure(
            description="review last bill received in gmail and pay it",
            actions=[
                Action(id=0, name="read last 10 emails and find the bill", type="gmail", confirmed=True),
                Action(id=1, name="load bill into erp", type="erp", confirmed=True),
                Action(id=2, name="send payment to provider", type="bank", confirmed=False)
            ]
        ),
        Procedure(
            description="prepare salaries of employees and pay them",
            actions=[
                Action(id=3, name="calculate monthly hours per employee", type="google sheets", confirmed=True),
                Action(id=4, name="load results into erp", type="erp", confirmed=True),
                Action(id=5, name="send payment to employees", type="bank", confirmed=False)
            ]
        ),
        Procedure(
            description="report on financial state of the company",
            actions=[
                Action(id=6, name="load last month data from erp", type="erp", confirmed=True),
                Action(id=7, name="calculate profits and costs", type="calculator", confirmed=True),
                Action(id=8, name="generate sections", type="agents", confirmed=True),
                Action(id=9, name="generate report", type="report generator", confirmed=True)
            ]
        )
    ]),
    "actions_database": ActionsDB(actions=[
        Action(id=0, name="read last 10 emails and find the bill", type="gmail", confirmed=True),
        Action(id=1, name="load bill into erp", type="erp", confirmed=True),
        Action(id=2, name="send payment to provider", type="bank", confirmed=False),
        Action(id=3, name="calculate monthly hours per employee", type="google sheets", confirmed=True),
        Action(id=4, name="load results into erp", type="erp", confirmed=True),
        Action(id=5, name="send payment to employees", type="bank", confirmed=False),
        Action(id=6, name="load last month data from erp", type="erp", confirmed=True),
        Action(id=7, name="calculate profits and costs", type="calculator", confirmed=True),
        Action(id=8, name="generate sections", type="agents", confirmed=True),
        Action(id=9, name="generate report", type="report generator", confirmed=True),
        Action(id=10, name="schedule a meeting with the team", type="calendar", confirmed=True),
        Action(id=11, name="order office supplies", type="inventory", confirmed=False),
        Action(id=12, name="backup company data", type="IT", confirmed=True),
        Action(id=13, name="update employee contact information", type="HR", confirmed=True),
        Action(id=14, name="approve leave requests", type="HR", confirmed=False),
        Action(id=15, name="prepare quarterly financial report", type="finance", confirmed=False),
        Action(id=16, name="conduct employee performance reviews", type="HR", confirmed=True),
        Action(id=17, name="organize team-building activities", type="HR", confirmed=True),
        Action(id=18, name="review and approve expense reports", type="finance", confirmed=False),
        Action(id=19, name="update company website", type="IT", confirmed=True)
    ]), 
    "human_feedback_select_message": "approve"
    }

def save_procedure(state: CreatorState):

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

    actions_penging = state.get('actions_pending', [])

    # Check if human feedback
    #human_feedback_confirm_message=state.get('human_feedback_confirm_message','no')
    if actions_penging == []:

        return "start" # if user cancels OR all tasks are done just finish the execution

    # Otherwise 
    else:
        return "executor"

def decide_create_done(state: CreatorState):

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
    
    #get data from the orchestrator state
    #raise NodeInterrupt("sldñfkjasñdflkajs")
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
builder.add_node("human_feedback_confirm", human_feedback_confirm) # nothing but just to stop on it
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
builder.add_edge("executor", "human_feedback_confirm")
builder.add_conditional_edges("human_feedback_confirm", decide_execute_end, ["executor", "start"])

# Compile
graph = builder.compile(interrupt_before=['human_feedback_select', 'start', 'human_feedback_create'])