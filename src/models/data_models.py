from pydantic import BaseModel
from typing import List, Optional, Union
from datetime import date


class PurchaseHistory(BaseModel):
    ticket_type: str
    date: date
    zones_traveled: Optional[int] = None
    price: int


class Ticket(BaseModel):
    name: str
    category: str
    number_of_zones: Optional[Union[int, None]]


class Persona(BaseModel):
    persona_id: str
    expected_ticket: Ticket
    date: date
    long_prompt: str
    long_prompt_no: str


class LLMResponse(BaseModel):
    raw: str
    status: str
    recommended_ticket: Optional[Ticket] = None
    explanation: Optional[str] = None
    question: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cot_output: Optional[str] = None


class TestResult(BaseModel):
    persona_id: str
    expected_ticket: Ticket
    status: str
    cot_output: Optional[str] = None
    recommended_ticket: Optional[Ticket] = None
    question: Optional[str]
    is_correct: bool
    timestamp: str
    model: str
    time_used: Optional[float]
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    explanation: Optional[str]
    raw: Optional[str] = None


class TestResultPerModel(BaseModel):
    persona_id: str
    expected_ticket: dict
    recommended_ticket: Optional[dict]
    question: Optional[str]
    is_correct: bool
    time_used: float
    cot_output: Optional[str] = None
    status: str
    timestamp: str
    model: str
    temperature: Optional[float]
    reasoning_effort: Optional[str]
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    include_persona: bool
    include_examples: bool
    include_cot_examples: bool
    include_cot_output: bool
    explanation: Optional[str]
    raw: Optional[str] = None