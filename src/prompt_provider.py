import json

from utils.logger import setup_logger
from utils.load_json import load_json 

logger = setup_logger()

class PromptProvider:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.available_tickets = load_json(self.base_path, "bus_tickets.json")
        self.general_info = load_json(self.base_path, "general_info.json")
        self.categories = load_json(self.base_path, "categories.json")
        self.prompts = load_json(self.base_path, "prompts.json")

    def get_prompt(
        self,
        include_persona: bool = False,
        include_examples: bool = False,
        include_cot_examples: bool = False,
        include_cot_output: bool = False,
    ) -> str:
        prompt_intro = self.prompts.get("base_instruction")
        if not prompt_intro:
            raise ValueError(f"❌ Missing base instruction in prompts.json.")

        prompt_parts = []

        if include_persona:
            persona_prompt = self.prompts.get("persona")
            if persona_prompt:
                prompt_parts.append(persona_prompt.strip())

        prompt_parts.append(prompt_intro.strip())

        if include_examples:
            examples_prompt = self.prompts.get("examples")
            if examples_prompt:
                prompt_parts.append(json.dumps(examples_prompt, ensure_ascii=False))
                
        if include_cot_examples:
            examples_with_cot_prompt = self.prompts.get("examples_with_cot")
            if examples_with_cot_prompt:
                prompt_parts.append(json.dumps(examples_with_cot_prompt, ensure_ascii=False))


        combined_prompt = "\n\n".join(prompt_parts)

        data_block = f"""
        Available Data
        - Available tickets: {self.available_tickets}
        - General info: {self.general_info}
        - Traveller categories: {self.categories}

        Response Format:
        
        {{
        {('"cot_output": "chain of thought reasoning",' if include_cot_output else "")}
        "status": "completed",
        "recommended_ticket": {{
            "name": "ticket name in english",
            "category": "traveler category in english",
            "number_of_zones": "number of zones or null if not applicable"
        }},
        "explanation": "reasoning for the recommendation in the users language",
        }}

        If more details are needed:
        
        {{
        {('"cot_output": "chain of thought reasoning",' if include_cot_output else "")}
        "status": "need_more_info",
        "question": "Ask one specific question in the users language."
        }}
        
        Always return pure JSON without formatting.
        """
        
        return f"{combined_prompt}\n\n{data_block.strip()}"
