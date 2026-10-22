from groq import Groq
from dotenv import load_dotenv
from src.generation.prompt import build_prompt

load_dotenv()



client = Groq()


MODEL_NAME = "openai/gpt-oss-20b"


def generate_answer(question: str, documents: list[str]) -> str:

    prompt = build_prompt(question, documents)

    response = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {
                "role" : "user" , 
                "content" : prompt
            }
        ],
        temperature =0.1
    )

    return response.choices[0].message.content