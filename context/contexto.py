import ollama
import re

class ContextualAssistant:
    def __init__(self):
        self.history = [
            {"role": "system", "content": "Always answer briefly, directly, and objectively, with no more than 3 sentences."}
        ]


    def load_context_from_file(self, filename):
        with open(filename, 'r') as file:
            context = file.read()
            self.history.append({"role": "system", "content": context})

    def update_context(self, new_message):
        self.history.append({"role": "user", "content": new_message})

    def clean_answer(self, answer):
        answer = re.sub(r"^(There is no mention of.*?RoboCup.*?Home competition.*?realistic home environment)", "", answer)
        answer = re.sub(r"^.*?the goal of RoboCup.*?and others", "", answer)     
        answer = re.sub(r"(?:\s+|\n+)", " ", answer).strip()  # Remove quebras de linha ou espaços extras

        return answer.strip()

    def ask_question(self, question):
        self.update_context(question)
        
        response = ollama.chat(
            model="llama3.1:8b",
            messages=self.history
        )
        
        answer = response['message']['content']
        cleaned_answer = self.clean_answer(answer)
        
        self.history.append({"role": "assistant", "content": cleaned_answer})

        print(f"[Ollama]: {cleaned_answer}")
        return cleaned_answer


assistant = ContextualAssistant() # Uso do assistente com contexto carregado de um arquivo

assistant.load_context_from_file('context.txt')

assistant.ask_question("how large is the territory of the state") # Pergunta para o assistente
