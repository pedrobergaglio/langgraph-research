from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Action(BaseModel):
    id: int
    type: str
    name: str
    confirmed: bool = True
    params_schema: Optional[dict] = None
    params: Optional[dict] = None
    function: str = ""
    metadata: Optional[dict] = None


class Procedure(BaseModel):
    description: str = Field(description="What the procedure does")
    actions: List[Action] = Field(description="Ordered list of actions that the procedure does")
    #types:List[type] to search procedures by the actions it does (gmail, bank, erp...)
    @property # we can get all the data in text for a model to use by defining this and using it like: analyst.persona
    def info(self) -> str:
        return f"Description: {self.description}\nActions: {self.actions}"

class ToyStoredProceduresDB(BaseModel):
    procedures:List[Procedure]

