from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

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
    #general_description: str = Field(description="General description of the ERP design")
    modules: List[Module] = Field(description="List of ERP system modules")

class ERPDesignState(TypedDict):
    last_answer: str            # Last human answer
    company_name: str           # Company name
    prompt: str                 # Current prompt for human input
    user_info: str              # Collected company info during interview
    question_index: int         # Current question index
    examples: str               # Optional examples to assist design context
    erp_design: List[Module]             # Current ERP design (text)
    feedback: str               # Human feedback text (empty if accepted)
    final_design: Design           # Final accepted ERP design

# --- Stage Prompts and Functions ---
# Stage 1: Interview (UI in Spanish Argentina)
def interview_user(state: ERPDesignState):
    """Stage1: Interview to collect company info.
       The node waits for human input on each question.
       If 'last_answer' is not provided, it returns the question prompt.
       When an answer is provided, it appends it and moves to the next question.
    """
    questions = [
        "¿Cuántos empleados tiene la empresa?",
        "¿Cuántos usuarios únicos tendrá el sistema?",
        "¿Cuántos roles diferentes hay en la empresa?",
        "¿En qué industria se desempeña la empresa?",
        "Describa en detalle los productos y servicios que ofrece la empresa.",
        "¿Quiénes son sus segmentos de clientes y cuáles son sus características?",
        "Describa a sus proveedores y cómo se gestionan los pagos.",
        "Mencione los módulos generales que el sistema debe cubrir.",
        "¿Qué nivel de simplicidad vs. completitud espera en el sistema?",
        "¿Cuál es el dolor principal que desea resolver con el sistema?",
        "¿Qué otros softwares utiliza y qué papel cumple cada uno?",
        "¿Con qué sistemas se debe integrar para ofrecer una experiencia increíble?",
        "¿Dispone de documentos sobre el funcionamiento actual que se quieran reemplazar?",
        "Proporcione datos de la imagen corporativa para personalización."
    ]
    # Initialize state if needed
    if "question_index" not in state:
        state["question_index"] = 0
        state["user_info"] = ""
    
    idx = state["question_index"]
    # If waiting for human input, check for key "last_answer"
    if "last_answer" not in state or state["last_answer"] == "":
        # Ask next question and signal UI to wait for human response
        prompt = f"Por favor, responda: {questions[idx]}"
        # Return prompt so that external UI knows to collect input.
        return {"prompt": prompt}
    else:
        # Append the provided answer and clear it for the next iteration.
        answer = state.pop("last_answer")
        state["user_info"] += f"\n{questions[idx]}: {answer}"
        state["question_index"] = idx + 1
        # If still more questions, return next question prompt.
        if state["question_index"] < len(questions):
            next_prompt = f"Por favor, responda: {questions[state['question_index']]}"
            return {"prompt": next_prompt, "user_info": state["user_info"], "question_index": state["question_index"]}
        ***REMOVED***wise, all questions answered; continue to next stage.
        return {}

# Stage 2: Generate ERP Design - now using structured output for proper design format.
def generate_erp_design(state: ERPDesignState):
    """Generate ERP design based on collected company info and examples using structured output."""
    context = state["user_info"]
    if state.get("examples"):
        context += "\nEjemplos:\n" + state["examples"]
    prompt = f"""Based on the following company information:
{context}

Generate an ERP design proposal. The design should be structured as a list of modules,
where each module has:
- Module name
- Utility description
- Usage description
- Tables and Columns (presented as a list in JSON-like format)
- Key features and actions

Please use only the provided information and format your response clearly."""
    structured_llm = llm.with_structured_output(Design)
    design = structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate ERP design proposal.")
    ])
    # Save the structured design as JSON (string) for further processing.
    return {"erp_design": design.modules}

# Stage 3: Obtain Human Feedback (UI in Spanish)
def get_feedback(state: ERPDesignState):
    """Acquire human feedback on the generated ERP design.
       This node waits for human input provided via the 'last_answer' field.
    """
    # If waiting for human feedback, check "last_answer"
    if "last_answer" not in state or state["last_answer"] == "":
        prompt = "Por favor, ingrese retroalimentación sobre el diseño propuesto (deje en blanco para aceptar):"
        return {"prompt": prompt}
    else:
        feedback = state.pop("last_answer")
        return {"last_answer": feedback}

def iterate_design(state: ERPDesignState):
    """Iterate the ERP design based on provided feedback.
       This reuses the generation approach by including the previous design and human feedback.
    """
    context = state["user_info"]
    if state.get("examples"):
        context += "\nEjemplos:\n" + state["examples"]
    # Append previous design and the feedback received
    context += f"\n\nDesign Iteration:\n{state.get('erp_design')}\n"
    context += f"\nFeedback:\n{state.get('last_answer')}\n"
    prompt = f"""Based on the following company information and previous design iteration:
{context}

Revise the ERP design proposal based on the human feedback.
The design should be structured as a list of modules,
where each module has:
- Module name
- Utility description
- Usage description
- Tables and Columns (presented as a list in JSON-like format)
- Key features and actions

Please use only the provided information and format your response clearly."""
    structured_llm = llm.with_structured_output(Design)
    design = structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Revise the ERP design proposal accordingly.\nFeedback:\n{state.get('last_answer')}")
    ])
    return {"erp_design": design.modules}

def finalize_design(state: ERPDesignState):
    """Finalize and accept the ERP design."""
    return {"final_design": state["erp_design"]}

# --- Graph Setup ---
workflow = StateGraph(ERPDesignState)

# Stage 1: Interview nodes - loop until all questions are answered
workflow.add_node("interview_user", interview_user)

# Stage 2: ERP design generation
workflow.add_node("generate_design", generate_erp_design)

# Stage 3: Feedback and iteration
workflow.add_node("get_feedback", get_feedback)
workflow.add_node("iterate_design", iterate_design)
workflow.add_node("finalize_design", finalize_design)

# Graph edges:
# Start with interview stage. Loop interview_user until complete.
workflow.add_edge(START, "interview_user")
def interview_router(state: ERPDesignState) -> str:
    """Route based on whether all interview questions are completed."""
    return "generate_design" if state.get("question_index", 0) >= 13 else "interview_user"

workflow.add_conditional_edges("interview_user",
    interview_router,
    ["interview_user", "generate_design"])

# Then generate design and get feedback
workflow.add_edge("generate_design", "get_feedback")
# If feedback is non-empty, iterate design and ask for feedback again (loop back to get_feedback)
def feedback_router(state: ERPDesignState) -> str:
    """Route based on feedback presence to either iterate or finalize design."""
    return "iterate_design" if state.get("last_answer", "").strip() != "" else "finalize_design"

workflow.add_conditional_edges("get_feedback",
    feedback_router,
    ["iterate_design", "finalize_design"])
# After iterate, update design then ask for feedback again
workflow.add_edge("iterate_design", "get_feedback")
workflow.add_edge("finalize_design", END)

# Compile graph ensuring interruption at interview stage for human interaction if needed
graph = workflow.compile(interrupt_before=["interview_user", "get_feedback"])
