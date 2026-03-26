# explainer.py
from openai import OpenAI
from openai import RateLimitError
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def explain_classification(ticket_text: str, predicted_label: str, top_chunks: list[tuple] = None) -> str:
    chunk_context = ""
    if top_chunks:
        terms = ", ".join([chunk for chunk, _ in top_chunks[:5]])
        chunk_context = f"\nCommon terms in {predicted_label} tickets: {terms}."

    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash-lite",
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
