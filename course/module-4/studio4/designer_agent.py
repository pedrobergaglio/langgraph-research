from typing import List, TypedDict, Annotated
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START, MessagesState

# Use 4o-mini LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Updated Schema ---
class Module(BaseModel):
    name: str = Field(description="Module name")
    utility_description: str = Field(description="What is its utility?")
    usage_description: str = Field(description="How is it used?")
    tables_and_columns: List[str] = Field(description="List of tables and their columns")
    key_features_and_actions: List[str] = Field(description="Essential features and actions")

class Design(BaseModel):
    modules: List[Module] = Field(description="List of ERP system modules")

class InterviewResponse(BaseModel):
    message: str = Field(description="The interviewer's next message or question")
    is_finished: bool = Field(description="Whether all required information has been gathered")

# Following MessagesState pattern with messages and reducer
class ERPDesignState(MessagesState):
    company_name: str                        # Company name
    is_finished: bool                        # Whether interview is complete
    examples: str                            # Optional examples to assist design context
    erp_design: Design                       # Current ERP design
    feedback: str                            # Human feedback text (empty if accepted)
    final_design: Design                     # Final accepted ERP design

SYSTEM_PROMPT = """
Eres un asesor de tecnología especializado en el diseño de sistemas ERP. 

Tu tarea es entrevistar al cliente para recopilar toda la información necesaria, utilizando un enfoque conversacional amigable. 
Debes obtener información sobre los siguientes puntos:

1. Número de empleados
2. Número de usuarios únicos que tendrá el sistema
3. Número de roles diferentes
4. Industria de la empresa
5. Descripción completa de productos y servicios ofrecidos, modelo de negocio
6. Segmentos de clientes y sus características
7. Diferentes proveedores, cómo trabajan y cómo se les paga
8. Módulos generales o información que estén seguros que el sistema debería cubrir
9. Nivel de simplicidad vs. completitud del sistema (si quieren algo muy simple o algo muy completo)
10. ¿Cuál es el dolor principal que quieren resolver con la implementación del sistema?
11. Qué otro software utilizan y qué rol juega cada uno
12. ¿Con qué software debemos integrarnos para lograr una experiencia increíble?
13. ¿Tienen documentos para compartir sobre cómo trabajan hoy que les gustaría reemplazar?
14. Datos de marca de la empresa para personalización

INSTRUCCIONES IMPORTANTES:
- Mantén la conversación natural y amigable, como un consultor de gestión empresarial
- Pregunta de manera conversacional, no como una lista de verificación
- Si el usuario no proporciona suficiente detalle, haz preguntas de seguimiento
- Cuando tengas toda la información requerida, establece is_finished = True
- De lo contrario, mantén is_finished = False
- Solo termina cuando tengas información sobre TODOS los puntos mencionados

Comienza con una presentación cordial y tu primera pregunta.
"""

# --- Stage Prompts and Functions ---
# Stage 1: Interview (UI in Spanish Argentina)
def interview_user(state: ERPDesignState):
    """Conduct an open-ended but guided interview to collect company information.
       Uses a conversational approach while ensuring all required data is collected.
    """
    # Initialize state if needed
    if "messages" not in state or state["messages"] == []:  
        state["messages"] = []
        state["is_finished"] = False
        
        # Initial system message to set up the conversation
        return {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="Hola, estoy interesado en un sistema ERP para mi empresa.")]}
    
    feedback = state.get("feedback", "")

    """ if feedback == "":
        return {"messages": [SystemMessage(content="Por favor, responda los mensajes para continuar la entrevista.")]} """


    # If there was a human response, process it and continue the conversation
    if len(state["messages"]) > 0:
        # Use structured output to get next message and check if finished
        structured_llm = llm.with_structured_output(InterviewResponse)
        
        # Add more explicit instructions for the structured output format
        prompt = f"""
        Basado en la conversación hasta ahora, genera el siguiente mensaje para el cliente.

        INSTRUCCIONES PARA EL FORMATO DE SALIDA:
        
        Debes responder con un objeto JSON que contiene exactamente estos dos campos:
        1. "message": El mensaje que le darás al cliente (como string)
        2. "is_finished": Un booleano (true/false) que indica si ya tienes toda la información necesaria
        
        Ejemplo de formato correcto:
        {{
          "message": "Gracias por compartir esa información. ¿Podría decirme más sobre...?",
          "is_finished": false
        }}
        
        O cuando hayas completado la entrevista:
        {{
          "message": "Excelente, he recopilado toda la información necesaria para diseñar su sistema ERP. Gracias por su tiempo.",
          "is_finished": true
        }}
        
        Asegúrate de establecer 'is_finished' como true SOLO cuando hayas recopilado información sobre TODOS los 14 puntos mencionados.
        """
        
        response = structured_llm.invoke(
            state["messages"] + 
            [SystemMessage(content=prompt)] +
            [HumanMessage(content=feedback)]
        )
        
        # Update state with completion status
        state["is_finished"] = response.is_finished
        
        # Return next message from interviewer and update feedback
        return {
            "messages": [HumanMessage(content=feedback), AIMessage(content=response.message)],
            "is_finished": response.is_finished,
            "feedback": ""  # Reset feedback after using it
        }
    
    # If waiting for human input, return current state
    return {}

# Stage 2: Generate ERP Design - using structured output and considering message history
def generate_erp_design(state: ERPDesignState):
    """Generate ERP design based on collected company info in message history."""
    # Extract information from conversation history
    messages = state["messages"]

    there_is_system = False
    for message in messages:
        #check if the message is a system message
        if isinstance(message, SystemMessage):
            flag = True
            break
    # If there is no system message, add it at first message
    if not there_is_system:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    conversation_text = get_buffer_string(messages)
    
    # Include optional examples if available
    context = conversation_text
    if state.get("examples"):
        context += "\nEjemplos:\n" + state["examples"]
        
    prompt = f"""Based on the following conversation with a company representative:
{context}

Extract all relevant information and generate an ERP design proposal. The design should be structured as a list of modules,
where each module has:
- Module name
- Utility description
- Usage description
- Tables and Columns (presented as a list in JSON-like format)
- Key features and actions

Please use only the information provided and format your response clearly."""

    structured_llm = llm.with_structured_output(Design)
    design = structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate ERP design proposal.")
    ])
    
    # Return the structured design
    return {"messages": messages,
            "erp_design": design}

# Stage 3: Obtain Human Feedback (UI in Spanish)
def get_feedback(state: ERPDesignState):
    """Acquire human feedback on the generated ERP design."""
    # If waiting for human feedback
    if len(state["messages"]) > 0 and not isinstance(state["messages"][-1], HumanMessage):
        # Add a request for feedback to the conversation
        return {"messages": [SystemMessage(content="Por favor, ingrese retroalimentación sobre el diseño propuesto (deje en blanco para aceptar):")]}
    else:
        # Get feedback from the most recent human message
        feedback = state["messages"][-1].content if state["messages"] else ""
        return {"feedback": feedback}

def iterate_design(state: ERPDesignState):
    """Iterate the ERP design based on provided feedback."""
    # Get conversation context
    messages = state["messages"]
    conversation_text = get_buffer_string(messages)
    
    # Include existing design and feedback
    try:
        design_json = state['erp_design'].json()
    except Exception as e:
        print(f"Error serializing design: {e}")
        # Provide fallback approach
        design_json = str(state['erp_design'])
    
    prompt = f"""Based on the following conversation and feedback:
{conversation_text}

The current ERP design is:
{design_json}

Human feedback:
{state['feedback']}

Revise the ERP design proposal based on this feedback.
The design should be structured as a list of modules with the same format as before."""

    try:
        structured_llm = llm.with_structured_output(Design)
        design = structured_llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Revise the ERP design proposal accordingly.")
        ])
        return {"erp_design": design}
    except Exception as e:
        print(f"Error in LLM invocation: {e}")
        # Return existing design if there's an error
        return {"erp_design": state['erp_design']}

def finalize_design(state: ERPDesignState):
    """Finalize and accept the ERP design."""
    return {"final_design": state["erp_design"]}

# --- Graph Setup ---
workflow = StateGraph(ERPDesignState)

# Add nodes
workflow.add_node("interview_user", interview_user)
workflow.add_node("generate_design", generate_erp_design)
workflow.add_node("get_feedback", get_feedback)
workflow.add_node("iterate_design", iterate_design)
workflow.add_node("finalize_design", finalize_design)

# Graph edges
workflow.add_edge(START, "interview_user")

# Route based on interview completion
def interview_router(state: ERPDesignState) -> str:
    """Route based on whether interview is complete."""
    return "generate_design" if state.get("is_finished", False) else "interview_user"

workflow.add_conditional_edges("interview_user", interview_router, ["interview_user", "generate_design"])

# Design feedback flow
workflow.add_edge("generate_design", "get_feedback")

def feedback_router(state: ERPDesignState) -> str:
    """Route based on feedback presence."""
    feedback = state.get("feedback", "").strip()
    return "finalize_design" if feedback == "" else "iterate_design"

workflow.add_conditional_edges("get_feedback", feedback_router, ["iterate_design", "finalize_design"])
workflow.add_edge("iterate_design", "get_feedback")
workflow.add_edge("finalize_design", END)

# Compile graph with interruption points for human input
graph = workflow.compile(interrupt_before=["interview_user", "get_feedback"])
