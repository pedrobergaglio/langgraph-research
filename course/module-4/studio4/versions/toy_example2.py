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
    type:str # (gmail, bank, erp...)
    name:str
    confirmed:bool = True #if the action can be done

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
    human_feedback_message:str #later feedback TODO load the entire conversation, not just the last update
    stored_procedures:ToyStoredProceduresDB
 """

class ExecutorState(MessagesState): 
    #we use messages to store the questions from the agent and answers from the user
    procedure:Procedure
    actions_done:List[Action] = [] #to store the actions done by the executor

class GeneralState(TypedDict): #this is the general state with the general data needed.
    request:str
    stored_procedures:ToyStoredProceduresDB
    human_feedback_message:str
    human_confirmation_message: Optional[Literal['yes', 'no']] = None
    procedure:Procedure
    actions_pending:List[Action]
    actions_done: List[Action]

### Nodes and edges

# prompt for the analyst generator, the main topic, the last feedback, and the number of analysts
orchestrator_instructions="""You are tasked to select one of the stored procedures based on the user request. 
Follow these instructions carefully:

1. First, review the user request:
{request}
        
2. Examine each of the stored procedures that have been provided: 
        
{stored_procedures}

3. Consider the last user feedback if provided here:

{human_feedback_message}
    
4. Determine the most accurate procedure based upon information above.
                    
5. Return the ID of the selected one, starting 0 as the ID for the first procedure"""

# in the creation, the state for analyst generation is used. 
def select_procedure(state: GeneralState):
    
    """ Select procedure to do """
    
    #get data from the orchestrator state
    request = state["request"]
    human_feedback_message = state.get('human_feedback_message', '') #this can be none
    stored_procedures = {
        "procedures": [
            {
                "description": "review last bill received in gmail and pay it",
                "actions": [
                    {"name": "read last 10 emails and find the bill", "type": "gmail", "confirmed": True},
                    {"name": "load bill into erp", "type": "erp", "confirmed": True},
                    {"name": "send payment to provider", "type": "bank", "confirmed": False}
                ]
            },
            {
                "description": "prepare salaries of employees and pay them",
                "actions": [
                    {"name": "calculate monthly hours per employee", "type": "google sheets", "confirmed": True},
                    {"name": "load results into erp", "type": "erp", "confirmed": True},
                    {"name": "send payment to employees", "type": "bank", "confirmed": False}
                ]
            },
            {
                "description": "report on financial state of the company",
                "actions": [
                    {"name": "load last month data from erp", "type": "erp", "confirmed": True},
                    {"name": "calculate profits and costs", "type": "calculator", "confirmed": True},
                    {"name": "generate sections", "type": "agents", "confirmed": True},
                    {"name": "generate report", "type": "report generator", "confirmed": True}
                ]
            }
        ]
    }
        
    # Enforce structured output to get the index of the procedure
    structured_llm = llm.with_structured_output(BestProcedure) 
    
    # parsing the System message
    system_message = orchestrator_instructions.format(request=request, stored_procedures=stored_procedures,
                                                            human_feedback_message=human_feedback_message)

    # select procedure
    procedure_id = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Select the correct procedure")])

    procedure = stored_procedures["procedures"][procedure_id.id]
    
    # Write the procedure to state
    return {"stored_procedures": stored_procedures,
            "procedure": procedure, 
            "actions_pending": procedure["actions"], 
            "human_feedback_message": None, #we return the selected procedure and reset the feedback
            } 

# un punto concreto donde se para, para poder recibir feedback después mediante otro metodo
def human_feedback(state: GeneralState): 
    """ No-op node that should be interrupted on """
    pass

def human_confirmation(state: GeneralState): 
    """ No-op node that should be interrupted on """
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
    human_confirmation_message = state.get('human_confirmation_message', 'no') #this can be none
    #next_action = None
    actions = state["actions_pending"]

    for action in actions:

        

        #next_action = action
        confirmed = action['confirmed']

        if human_confirmation_message.lower() == 'yes' and not confirmed:
            #human_confirmation_message = 'no'
            confirmed = True
        if confirmed:
            actions_done.append(action)
            actions = actions[1:]
            #raise NodeInterrupt(f"The task {action['name']} has been completed")
            return {"actions_done": actions_done, 
            "actions_pending": actions, 
            "human_feedback_message": None, 
            "human_confirmation_message": "no"} #we return the actions done and the executor state

        else:
            raise NodeInterrupt(f"The task {action['name']} needs confirmation from the user")

        

        #next_action = None

    

# this initialize all the interviews in parallel using Send()
# only if the human approves the analysts creation
# else, it sends another interation of the creation of the analysts creation
def decide_execute_orchestrator(state: GeneralState):

    """ Conditional edge to initiate executor or return to orchestrator """    

    # Check if human feedback
    human_feedback_message=state.get('human_feedback_message','approve')
    if human_feedback_message.lower() != 'approve':
        # Return to create_analysts
        return "select_procedure"

    ***REMOVED***wise 
    else:
        return "executor"

def decide_execute_end(state: GeneralState):

    """ Conditional edge to initiate executor or return to orchestrator """ 

    #actions_done = state.get('actions_done', [])
    actions_penging = state.get('actions_pending', {})

    # Check if human feedback
    #human_confirmation_message=state.get('human_confirmation_message','no')
    if actions_penging == []:

        return END # if user cancels OR all tasks are done just finish the execution

    ***REMOVED***wise 
    else:
        return "executor"


# Add nodes and edges for the ORCHESTRATOR graph
builder = StateGraph(GeneralState) #initialization of the orchestrator graph
builder.add_node("select_procedure", select_procedure)
builder.add_node("human_feedback", human_feedback) # nothing but just to stop on it
builder.add_node("human_confirmation", human_confirmation) # nothing but just to stop on it
builder.add_node("executor", executor)

# Logic
builder.add_edge(START, "select_procedure")
builder.add_edge("select_procedure", "human_feedback")
builder.add_conditional_edges("human_feedback", decide_execute_orchestrator, ["select_procedure", "executor"])
builder.add_edge("executor", "human_confirmation")
builder.add_conditional_edges("human_confirmation", decide_execute_end, ["executor", END])


# Compile
graph = builder.compile(interrupt_before=['human_feedback'])