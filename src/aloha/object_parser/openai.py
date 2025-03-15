from typing import List, Optional
from transformers import pipeline
from aloha.object_parser.base import ObjectParser
from aloha.object_parser.prompts import (
    MULTI_OBJECT_EXAMPLES,
    MULTI_OBJECT_SYSTEM_PROMPT,
    SINGLE_OBJECT_EXAMPLES,
    SINGLE_OBJECT_SYSTEM_PROMPT,
)


class HuggingFaceObjectParser(ObjectParser):
    def __init__(
        self,
        num_target_examples: Optional[int] = 3,
        num_reference_examples: Optional[int] = 3,
        model_name: str = "EleutherAI/gpt-neo-125M",  # Nouveau modèle ici
    ):
        super().__init__(num_target_examples, num_reference_examples)

        # Charger le modèle de Hugging Face
        self.generator = pipeline("text-generation", model=model_name, device_map="auto")

    def generate_response(self, prompt: str) -> str:
        response = self.generator(prompt, max_new_tokens=100, do_sample=False)

        # Gérer le cas où le modèle ne renvoie rien
        if response and "generated_text" in response[0]:
            return response[0]["generated_text"].strip()
        return "Error: No response generated"

    def extract_objects_single_caption(self, caption: str) -> str:
        prompt = SINGLE_OBJECT_SYSTEM_PROMPT
        for idx in range(self._num_target_examples):
            user, system = SINGLE_OBJECT_EXAMPLES[idx]
            prompt += f"\nUser: {user}\nAssistant: {system}"

        prompt += f"\nUser: Caption: {caption}\nObjects:"
        return self.generate_response(prompt)

    def extract_objects_multiple_captions(self, captions: List[str]) -> str:
        prompt = MULTI_OBJECT_SYSTEM_PROMPT
        for idx in range(self._num_reference_examples):
            user, system = MULTI_OBJECT_EXAMPLES[idx]
            prompt += f"\nUser: {user}\nAssistant: {system}"

        formatted_captions = "\n".join(f"- {caption}" for caption in captions)
        prompt += f"\nUser: Captions:\n{formatted_captions}\nObjects:"
        return self.generate_response(prompt)


# Remplacer GPT35TurboObjectParser par HuggingFaceObjectParser
class GPT35TurboObjectParser(HuggingFaceObjectParser):
    def __init__(self, num_target_examples: Optional[int] = 3, num_reference_examples: Optional[int] = 3):
        super().__init__(num_target_examples, num_reference_examples, model_name="EleutherAI/gpt-neo-125M")  # Nouveau modèle
