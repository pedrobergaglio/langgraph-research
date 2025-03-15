from typing import List, TypedDict, Annotated, Literal, Optional
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START, MessagesState

# Use 4o-mini LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Updated Schema with new ERP Design Structure ---
class Table(BaseModel):
    """Database table representation for the ERP system"""
    name: str = Field(description="Table name")
    columns: List[str] = Field(description="List of columns with format: column_name (data_type)")
    description: str = Field(description="Description of the table functionality")

class Action(BaseModel):
    """Automation or action that can be performed in the ERP system"""
    name: str = Field(description="Action name")
    table: str = Field(description="Reference to the table this action operates on")
    steps: List[str] = Field(description="Steps that explain the action workflow")
    access: List[Literal["button", "insert", "update", "delete"]] = Field(
        description="How this action is accessed in the UI"
    )
    description: str = Field(description="Description of the action functionality")

class View(BaseModel):
    """UI view/screen in the ERP system"""
    name: str = Field(description="View name")
    table: str = Field(description="Reference to the table this view displays")
    style: Literal["table", "gallery", "board"] = Field(description="View display style")
    position: Literal["main_menu", "side_menu"] = Field(description="Position of the view in the UI")
    columns_displayed: List[str] = Field(description="Columns from the table to display in this view")

# New container classes for structured output
class TableList(BaseModel):
    """Container for a list of tables"""
    items: List[Table]

class ActionList(BaseModel):
    """Container for a list of actions"""
    items: List[Action]

class ViewList(BaseModel):
    """Container for a list of views and main color selection"""
    items: List[View]
    main_color: Literal[
        "red #A93226",
        "green #229954",
        "blue #2471A3",
        "light blue #2E86C1",
        "gray #6C7A89",
        "violet #7D3C98",
        "pink #D01C5B",
        "orange #B94A00",
        "yellow #D68910", 
        "brown #6E4A3F",
        "teal #17A589",
        "lime #73C6B6",
        "dark blue #212F3D",
        "magenta #884EA0",
        "olive #5E821F"
    ] = Field(description="Main color theme for the ERP system UI")

class ERPDesign(BaseModel):
    """Complete ERP system design with tables, actions, and views"""
    tables: List[Table] = Field(description="List of database tables")
    actions: List[Action] = Field(description="List of actions and automations")
    views: List[View] = Field(description="List of main menu views")
    main_color: str = Field(description="Main color theme for the ERP system UI")

class InterviewResponse(BaseModel):
    """Response format for the interview process"""
    message: str = Field(description="The interviewer's next message or question")
    is_finished: bool = Field(description="Whether all required information has been gathered")

# Following MessagesState pattern with messages and reducer
class ERPDesignState(MessagesState):
    company_name: str               # Company name
    is_finished: bool               # Whether interview is complete
    examples: str                   # Optional examples to assist design context
    erp_design: Optional[ERPDesign] # Current ERP design (can be None initially)
    feedback: str                   # Human feedback text (empty if accepted)
    final_design: Optional[ERPDesign] # Final accepted ERP design

# --- Stage Prompts and Functions ---
# Stage 1: Interview (UI in Spanish Argentina)
def interview_user(state: ERPDesignState):
    """Conduct an open-ended but guided interview to collect company information.
       Uses a conversational approach while ensuring all required data is collected.
    """
    # Initialize state if needed
    if "messages" not in state or state["messages"] == []:  
        # Initialize state with empty values
        state["messages"] = []
        state["is_finished"] = False
        
        # Initial system message to set up the conversation
        return {"messages": [SystemMessage(content="""
        You are a technology consultant specializing in ERP system design.

        Your task is to have a friendly conversation with the client, with the overall goal of gathering all necessary information.
        You must obtain information about the following points, but it's very important that the conversation feels natural, 
        and that you answer any question the customer has about how we work (you can make up the details) and recommendations about the system:

        1. Company industry
        2. Complete description of products and services offered, business model
        3. Customer segments and their characteristics
        4. Number of employees
        5. Number of unique users the system will have
        6. Number of different roles
        7. Different suppliers, how they work and how they are paid
        8. General modules or information they are sure the system should cover
        9. Level of simplicity vs. completeness of the system (whether they want something very simple or very complete)
        10. What is the main pain point they want to solve with the system implementation?
        11. What other software they use and what role each plays
        12. What software should we integrate with to achieve an incredible experience?
        13. Do they have documents to share about how they work today that they would like to replace?
        14. Company branding data for customization, and main color to use in the system.

        IMPORTANT INSTRUCTIONS:
        - Keep the conversation natural and friendly, like a business management consultant
        - Make sure to listen and reply with detail to the user's answers
        - Maintain a logical flow and ask for more details when needed
        - Maintain a clearly not too excited but interested tone, like always gently smiling while listening
        - Ask questions conversationally, not like a checklist
        - If the user doesn't provide enough detail, ask follow-up questions
        - When you have all required information, set is_finished = True
        - Otherwise, keep is_finished = False
        - Only finish when you have information about ALL mentioned points
        - ALL MESSAGES MUST BE IN ARGENTINIAN SPANISH

        Example of a robotic conversation and a natural one:
        - ¿Podrías darme una descripción completa de lo que fabrican y cómo funciona su modelo de negocio?
        - claro, como te decía importamos de italia los alternadores y somos oficiales de honda, el resto lo fabricamos todo nosotros, 
        tenemos una metalurgica donde se fabrican varios detalles, después hay un proceso de ensamblado que realizan dos personas. 
        contamos con 3 locales a la calle, pero sobre todo vendemos a clientes grandes como barrios privados u otrso distribuidores. 
        eso es suficiente? o hay algo más especifico que te interesaría?
        Robotic response:
        - Gracias. Ahora, ¿Quiénes son sus principales clientes y cuáles son sus características?
        (Se esta preguntando sobre los clientes cuando el usuario acaba de explicar bastante bien como funciona esa parte, la respuesta parece no haber escuchado en absoluto lo que el cliente dijo)
        Natural response:
        - Claro entiendo perfecto, entonces básicamente venden a la calle, pero sobre todo a clientes grandes como barrios privados y otros distribuidores. 
          Cuáles son sus estrategias para llegar a ellos y seguir creciendo ese segmento?
        (Se está profundizando en la información que ya se tiene mostrando interés especificamente en lo que el cliente ha comentado, no repitiendo la pregunta sin escuchar)

        Start with a cordial introduction and your first question.
        """), HumanMessage(content="Hola, estoy interesado en un sistema ERP para mi empresa.")]}

    # If there was a human response, process it and continue the conversation
    if len(state["messages"]) > 0 and isinstance(state["messages"][-1], HumanMessage):
        # Use structured output to get next message and check if finished
        structured_llm = llm.with_structured_output(InterviewResponse)
        
        # Add more explicit instructions for the structured output format
        prompt = f"""
        Based on the conversation so far, generate the next message for the client.

        OUTPUT FORMAT INSTRUCTIONS:
        
        You must respond with a JSON object containing exactly these two fields:
        1. "message": The message you will give to the client (as string)
        2. "is_finished": A boolean (true/false) indicating if you have all needed information
        
        Example of correct format:
        {{
          "message": "Gracias por compartir esa información. Me podrías contar un poco más sobre...?",
          "is_finished": false
        }}
        
        Or when you have completed the interview:
        {{
          "message": "¡Excelente! Ya tengo toda la información necesaria para diseñar tu sistema ERP. Muchas gracias por tu tiempo.",
          "is_finished": true
        }}
        
        Make sure to set 'is_finished' as true ONLY when you have collected information about ALL 14 points mentioned.
        """
        
        response = structured_llm.invoke(
            state["messages"] + 
            [SystemMessage(content=prompt)]
        )
        
        # Update state with completion status
        state["is_finished"] = response.is_finished
        
        # Return next message from interviewer
        return {
            "messages": [AIMessage(content=response.message)],
            "is_finished": response.is_finished,
        }
    
    # If waiting for human input, return current state
    return {}

# Unified design function that handles both initial design and iterations
def design_erp(state: ERPDesignState):
    """
    Unified function to design the ERP system.
    This function handles both initial design and iterations based on feedback.
    
    The design process follows a stepwise approach:
    1. First, generate database tables as the foundation of the system
    2. Then, create actions/automations that operate on those tables
    3. Finally, design UI views and select a color theme that matches company branding
    
    Each step builds on the previous one, ensuring a cohesive design.
    """
    # Extract conversation history for context
    messages = state["messages"]
    conversation_text = get_buffer_string(messages)
    
    # Include optional examples if available
    context = conversation_text
    if state.get("examples"):
        context += "\nEjemplos:\n" + state["examples"]
    
    # Check if this is an iteration based on feedback
    has_feedback = state.get("feedback", "").strip() != ""
    feedback_text = state.get("feedback", "")
    existing_design = state.get("erp_design")
    
    # Create base system prompt
    system_prompt = f"""You are an expert ERP system designer.
Based on the following conversation with a company representative:

{context}

"""
    
    # Add feedback context if this is an iteration
    if has_feedback:
        system_prompt += f"""
The current ERP design has been reviewed and received this feedback:
{feedback_text}

You need to redesign the ERP system taking this feedback into account.
"""
    
    # Include existing design for context in feedback iterations
    if existing_design and has_feedback:
        try:
            design_json = existing_design.json()
            system_prompt += f"""
The previous design was:
{design_json}

"""
        except Exception as e:
            print(f"Error serializing existing design: {e}")
    
    # STEP 1: Design Tables with refined prompt for clarity
    tables_prompt = system_prompt + """
STEP 1: Design Database Tables

First, create the necessary database tables for this ERP system. Each table should have:

- name: A clear, descriptive name in Spanish (e.g., "Clientes", "Productos", "Pedidos")
- columns: A list of columns with format "column_name (data_type)" 
  Examples: 
    - "id_cliente (INT)" for numeric IDs
    - "nombre (VARCHAR)" for text fields
    - "fecha_creacion (DATE)" for dates
    - "precio (DECIMAL)" for monetary values
- description: A detailed explanation of what the table stores and its purpose in the system

Tables should follow these guidelines:
- Create primary tables for main entities (clients, products, orders, etc.)
- Create relationship tables when needed to connect entities
- Include appropriate ID fields for relationships (e.g., id_cliente in an Ordenes table)
- Design with normalization principles (avoid data duplication)
- Keep column names consistent across tables
- All names and descriptions must be in Spanish Argentina

Focus on creating a comprehensive but minimal set of tables that covers all the company's needs.
"""

    # Get tables design using container class 
    structured_llm = llm.with_structured_output(TableList)
    tables_container = structured_llm.invoke([
        SystemMessage(content=tables_prompt),
        HumanMessage(content="Diseñe las tablas de base de datos para este sistema ERP.")
    ])
    tables = tables_container.items
    
    # Format table details for better context in subsequent steps
    table_details = []
    for table in tables:
        columns_text = ", ".join(table.columns)
        table_details.append(f"""
TABLE: {table.name}
DESCRIPTION: {table.description}
COLUMNS: {columns_text}
""")

    # STEP 2: Design Actions/Automations with refined prompt for clarity
    actions_prompt = system_prompt + f"""
STEP 2: Design Actions and Automations

Now, create the necessary actions and automations for this ERP system based on the tables you've designed:
{"".join(table_details)}

Each action should have:

- name: A clear, descriptive name in Spanish (e.g., "Actualizar Inventario Al Recibir Orden")
- table: The primary table this action is associated with (must be one of the tables you designed)
- steps: A numbered list of specific steps that explain exactly what happens during this action
- access: How this action is triggered, using one or more of these specific values:
  • "button" - When the action is manually initiated by a user clicking a button
  • "insert" - When the action is triggered automatically after a new record is created
  • "update" - When the action is triggered automatically after a record is modified
  • "delete" - When the action is triggered automatically after a record is deleted
- description: A detailed explanation of what this action accomplishes

Important guidelines:
- Focus on business process automations, not basic CRUD operations
- Standard create/read/update/delete operations are built-in and don't need to be defined
- Include automations that connect multiple tables or perform calculations
- Consider workflows like approvals, notifications, or status updates
- Define automations for key business processes mentioned in the interview
- All names and descriptions must be in Spanish Argentina

Examples of good actions:
- Automatic inventory updates when orders are processed
- Email notifications when stock levels are low
- Calculations for reports or dashboards
- Status changes based on specific conditions
"""

    # Get actions design using container class
    structured_llm = llm.with_structured_output(ActionList)
    actions_container = structured_llm.invoke([
        SystemMessage(content=actions_prompt),
        HumanMessage(content="Diseñe las acciones y automatizaciones para este sistema ERP.")
    ])
    actions = actions_container.items
    
    # Format action details for better context in the views step
    action_details = []
    for action in actions:
        steps_text = "\n   - " + "\n   - ".join(action.steps)
        access_methods = ", ".join(action.access)
        action_details.append(f"""
ACTION: {action.name}
TABLE: {action.table}
DESCRIPTION: {action.description}
ACCESS METHODS: {access_methods}
STEPS:{steps_text}
""")
    
    # STEP 3: Design UI Views and select main color with refined prompt for clarity
    views_prompt = system_prompt + f"""
STEP 3: Design User Interface Views and Select Main Color Theme

Finally, define the displayed views for this ERP system and select a main color theme.

Here is the complete system you've designed so far:

TABLES:
{"".join(table_details)}

ACTIONS:
{"".join(action_details)}

For each view, specify:

- name: A clear, descriptive name in Spanish that users will see in the menu
- table: The table this view displays (must be one of the tables you designed)
- style: How the data should be displayed, choose one:
  • "gallery" - RECOMMENDED - Card-based layout with images or previews 
  • "board" - Kanban-style board with cards organized by status - use when tasks or statuses are relevant
  • "table" - use only when detailed data is needed at a glance - Traditional tabular format with rows and columns 
- position: Where this view should appear in the UI, choose one:
  • "main_menu" - Primary views that are immediately visible (MAXIMUM 3)
  • "side_menu" - Secondary views accessible through navigation
- columns_displayed: Specific columns from the table to show in this view
  (use the exact column names from the table definition)

Important guidelines:
- Create views  for all tables that users need to interact with
- System tables (logs, settings, etc.) should not have views
- Main menu should only contain 3 or fewer most important views
- Secondary views go in side_menu
- Select columns that provide the most relevant information at a glance
- Consider which views support which actions
- All names must be in Spanish Argentina

Also, select ONE main color theme for the ERP system based on the company's branding and preferences mentioned during the interview. Choose from:

- red #A93226 - For energetic, bold, retail or food businesses
- green #229954 - For environmental, financial, or healthcare businesses
- blue #2471A3 - For professional services, technology, or corporate businesses
- light blue #2E86C1 - For healthcare, education, or modern tech businesses
- gray #6C7A89 - For neutral, conservative, or professional businesses
- violet #7D3C98 - For creative, luxury, or premium businesses
- pink #D01C5B - For fashion, beauty, or youth-oriented businesses
- orange #B94A00 - For enthusiastic, creative, or retail businesses
- yellow #D68910 - For optimistic, affordable, or food businesses
- brown #6E4A3F - For reliable, natural, or traditional businesses
- teal #17A589 - For modern, trustworthy, or healthcare businesses
- lime #73C6B6 - For fresh, organic, or environmental businesses
- dark blue #212F3D - For corporate, secure, or financial businesses
- magenta #884EA0 - For innovative, bold, or creative businesses
- olive #5E821F - For natural, traditional, or quality-focused businesses

Match the color to the company's industry, brand positioning, and any color preferences mentioned in the interview.
"""

    # Get views design and color using container class
    structured_llm = llm.with_structured_output(ViewList)
    views_container = structured_llm.invoke([
        SystemMessage(content=views_prompt),
        HumanMessage(content="Diseñe las vistas de interfaz y seleccione un tema de color para este sistema ERP.")
    ])
    views = views_container.items
    main_color = views_container.main_color
    
    # Combine everything into final comprehensive design
    erp_design = ERPDesign(
        tables=tables,
        actions=actions,
        views=views,
        main_color=main_color
    )
    
    # Return the completed design
    return {"erp_design": erp_design}

# Get human feedback on the design
def get_feedback(state: ERPDesignState):
    """
    Acquire human feedback on the generated ERP design.
    
    This function handles:
    1. Requesting feedback from the human user in Spanish
    2. Extracting the feedback response
    3. Returning the feedback to be processed by the workflow
    
    If feedback is empty, the design is accepted as-is.
    """
    # If waiting for human feedback
    if len(state["messages"]) > 0 and not isinstance(state["messages"][-1], HumanMessage):
        # Add a request for feedback to the conversation
        return {"messages": [SystemMessage(content="Por favor, ingrese retroalimentación sobre el diseño propuesto (deje en blanco para aceptar):")]}
    else:
        # Get feedback from the most recent human message
        feedback = state["messages"][-1].content if state["messages"] else ""
        return {"feedback": feedback}

# Finalize the accepted design
def finalize_design(state: ERPDesignState):
    """
    Finalize and accept the ERP design.
    
    This function marks the completion of the design process by:
    1. Moving the current design to the final_design field
    2. Signaling that the process is complete
    """
    return {"final_design": state["erp_design"]}

# --- Graph Setup ---
workflow = StateGraph(ERPDesignState)

# Add nodes
workflow.add_node("interview_user", interview_user)
workflow.add_node("design_erp", design_erp)
workflow.add_node("get_feedback", get_feedback)
workflow.add_node("finalize_design", finalize_design)

# Graph edges
workflow.add_edge(START, "interview_user")

# Route based on interview completion
def interview_router(state: ERPDesignState) -> str:
    """Route based on whether interview is complete."""
    return "design_erp" if state.get("is_finished", False) else "interview_user"

workflow.add_conditional_edges("interview_user", interview_router, ["interview_user", "design_erp"])

# Design feedback flow
workflow.add_edge("design_erp", "get_feedback")

def feedback_router(state: ERPDesignState) -> str:
    """Route based on feedback presence."""
    feedback = state.get("feedback", "").strip()
    return "finalize_design" if feedback == "" else "design_erp"  # Route back to design_erp if feedback exists

workflow.add_conditional_edges("get_feedback", feedback_router, ["design_erp", "finalize_design"])
workflow.add_edge("finalize_design", END)

# Compile graph with interruption points for human input
graph = workflow.compile(interrupt_before=["interview_user", "get_feedback"])
