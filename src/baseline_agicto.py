from __future__ import annotations

import os
from openai import OpenAI


AGICTO_MODEL = "gemini-2.5-flash"


def ask_agicto_baseline(query: str) -> str:
    client = OpenAI(
        api_key=os.environ["AGICTO_API_KEY"],
        base_url="https://api.agicto.cn/v1",
    )

    prompt = f"""
You are a course assistant for a Generative AI course.

Answer the student's question as clearly and concisely as possible.
If you are unsure, say so.

Student question:
{query}
""".strip()

    response = client.chat.completions.create(
        model=AGICTO_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main() -> None:
    if "AGICTO_API_KEY" not in os.environ:
        print("Missing AGICTO_API_KEY in environment.")
        return

    while True:
        query = input("\nAsk a course question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        answer = ask_agicto_baseline(query)
        print("\nBaseline answer:\n")
        print(answer)


if __name__ == "__main__":
    main()