"""
Legacy Gemini-based version.

This script was used in the earlier stage of the project when the generation
backend relied on the Gemini free tier. It is kept for reference only.
The active pipeline now uses AGICTO due to rate-limit issues encountered
during evaluation and demo development.
"""

from __future__ import annotations

import os
from google import genai


GEMINI_MODEL = "gemini-2.5-flash"


def ask_gemini_baseline(query: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = f"""
You are a course assistant for a Generative AI course.

Answer the student's question as clearly and concisely as possible.
If you are unsure, say so.
Keep the answer short and direct.

Student question:
{query}
""".strip()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def main() -> None:
    if "GEMINI_API_KEY" not in os.environ:
        print("Missing GEMINI_API_KEY in environment.")
        return

    while True:
        query = input("\nAsk a course question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        answer = ask_gemini_baseline(query)
        print("\nBaseline answer:\n")
        print(answer)


if __name__ == "__main__":
    main()