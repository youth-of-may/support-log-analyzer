from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def explain_classification(ticket_text: str, predicted_label: str, top_chunks: list[tuple] = None) -> str:
    chunk_context = ""
    if top_chunks:
        terms = ", ".join([chunk for chunk, _ in top_chunks[:5]])
        chunk_context = f"\nCommon terms in {predicted_label} tickets: {terms}."

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a support ticket analyst. Be concise and specific."},
                {"role": "user", "content": f"""A support ticket was classified into the category: "{predicted_label}".{chunk_context}

Ticket text:
\"\"\"{ticket_text}\"\"\"

In 2-3 sentences, explain why this ticket belongs to "{predicted_label}". Reference specific words or phrases from the ticket."""}
            ]
        )
        return response.choices[0].message.content
    except RateLimitError:
        return f"⚠️ Explanation unavailable — API rate limit reached. The ticket was classified as **{predicted_label}**."