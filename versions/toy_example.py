import operator #operator.add(x, y) is equivalent to the expression x+y
from pydantic import BaseModel, Field # base for classes 
from typing import Annotated, List, TypedDict 
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

### LLM

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 

### Schema

class Action(BaseModel):
    type:str # (gmail, bank, erp...)
    name:str

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

class OrchestratorState(TypedDict):
    procedure:Procedure # TODO let the orchestrator edit the retrieved procedure based on human request
    request:str #initial request
    human_feedback_message:str #later feedback TODO load the entire conversation, not just the last update
    stored_procedures:ToyStoredProceduresDB

class ExecutorState(MessagesState): 
    #we use messages to store the questions from the agent and answers from the user
    procedure:Procedure
    tasks_done:List[Action] = [] #to store the actions done by the executor

class GeneralState(TypedDict): #this is the general state with the general data needed.
    request:str
    stored_procedures:ToyStoredProceduresDB
    human_feedback_message:str
    procedure:Procedure
    tasks_done: Annotated[List[Action], operator.add]


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
def select_procedure(state: OrchestratorState):
    
    """ Select procedure to do """
    
    #get data from the orchestrator state
    request = state["request"]
    human_feedback_message = state.get('human_feedback_message', '') #this can be none
    stored_procedures = {
        "procedures": [
            {
                "description": "review last bill received in gmail and pay it",
                "actions": [
                    {"name": "read last 10 emails and find the bill", "type": "gmail"},
                    {"name": "load bill into erp", "type": "erp"},
                    {"name": "send payment to provider", "type": "bank"}
                ]
            },
            {
                "description": "prepare salaries of employees and pay them",
                "actions": [
                    {"name": "calculate monthly hours per employee", "type": "google sheets"},
                    {"name": "load results into erp", "type": "erp"},
                    {"name": "send payment to employees", "type": "bank"}
                ]
            },
            {
                "description": "report on financial state of the company",
                "actions": [
                    {"name": "load last month data from erp", "type": "erp"},
                    {"name": "calculate profits and costs", "type": "calculator"},
                    {"name": "generate sections", "type": "agents"},
                    {"name": "generate report", "type": "report generator"}
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
    
    # Write the procedure to state
    return {"procedure": stored_procedures["procedures"][procedure_id.id]}

# un punto concreto donde se para, para poder recibir feedback después mediante otro metodo
def human_feedback(state: OrchestratorState): 
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
def execute_procedure(state: ExecutorState):

    """ Node to execute the procedure """

    # Get state
    procedure = state["procedure"]
    tasks_done = state["tasks_done"]

    for action in procedure["actions"]:
        # Execute action
        tasks_done.append(action)

    return {"tasks_done": [tasks_done]}

# this initialize all the interviews in parallel using Send()
# only if the human approves the analysts creation
# else, it sends another interation of the creation of the analysts creation
def decide_execute_orchestrator(state: OrchestratorState):

    """ Conditional edge to initiate executor or return to orchestrator """    

    # Check if human feedback
    human_analyst_feedback=state.get('human_feedback_message','approve')
    if human_analyst_feedback.lower() != 'approve':
        # Return to create_analysts
        return "select_procedure"

    # Otherwise 
    """ else:
        return END """


# Add nodes and edges for the ORCHESTRATOR graph
orchestrator_builder = StateGraph(OrchestratorState) #initialization of the orchestrator graph
orchestrator_builder.add_node("select_procedure", select_procedure)
orchestrator_builder.add_node("human_feedback", human_feedback) # nothing but just to stop on it

# Flow
orchestrator_builder.add_edge(START, "select_procedure")
orchestrator_builder.add_edge("select_procedure", "human_feedback") # feedback
# if the user does not approve, iterate.
orchestrator_builder.add_conditional_edges("human_feedback", decide_execute_orchestrator, ["select_procedure", END])

# send the request to the orchestrator via Send() API
def send_orchestrator(state:GeneralState):
    Send("orchestrator", 
         {"request": state["request"],
          "stored_procedures": state.get('stored_procedures', None),
          "human_feedback_message": state.get('human_feedback_message', '')})

# Add nodes and edges 
builder = StateGraph(GeneralState) # initialize with the general state
builder.add_node("orchestrator", orchestrator_builder.compile(interrupt_before=['human_feedback'])) # orchestrator graph with its state
builder.add_node("executor", execute_procedure)
builder.add_node("send_orchestrator", send_orchestrator)

# Logic
builder.add_edge(START, "send_orchestrator")
builder.add_edge("send_orchestrator", "orchestrator") # send the request to the orchestrator
builder.add_edge("orchestrator", "executor")
builder.add_edge("executor", END)

# Compile
graph = builder.compile()