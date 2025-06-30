from pydantic import BaseModel

class Answer(BaseModel):
    Option: str
    Reason: str
