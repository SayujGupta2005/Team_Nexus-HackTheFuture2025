import pandas as pd
import os
from groq import Groq
import numpy as np

# === Set up GROQ client ===
os.environ["GROQ_API_KEY"] = "YOUR_API_KEY"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# === Load CSV Data ===
hh_train = pd.read_csv("data/hh-level-train-data.csv")
hh_test = pd.read_csv("data/hh-level-test-data.csv")
person_train = pd.read_csv("data/person-level-train-data.csv")
person_test = pd.read_csv("data/person-level-test-data.csv")

for df in [hh_train, hh_test, person_train, person_test]:
    df.fillna("", inplace=True)

# === Groq LLM Fallback ===
def generate_reply(current_prompt: str) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant capable of analyzing and reasoning over structured household and personal economic data."
            },
            {"role": "user", "content": current_prompt}
        ],
        model="llama-3.2-11b-vision-preview",
    )
    return chat_completion.choices[0].message.content.strip()

# === Convert subset of data into RAG-friendly prompt ===
def build_rag_prompt(question: str, hh_df: pd.DataFrame, person_df: pd.DataFrame) -> str:
    hh_sample = hh_df.sample(min(5, len(hh_df))).to_string(index=False)
    person_sample = person_df.sample(min(5, len(person_df))).to_string(index=False)

    prompt = f"""
                You are an assistant answering questions about household and person-level data.

                Here is a sample from the Household Data:
                {hh_test}

                Here is a sample from the Person Data:
                {person_test}

                Now answer the following question based on patterns and your general knowledge and do not give code or your reasoning steps, directlly give the answer to the question
                Also some questions may require you to look at both household and person data.
                Some may require you to think as they may not be directly answerable from the data.

                Question: {question}
            """
    return prompt

# === Main Chat Logic (with RAG) ===
def chat(question, hh_test=hh_test, person_test=person_test):
    try:
        # Use sampled rows from data as context
        prompt = build_rag_prompt(question, hh_test, person_test)
        return generate_reply(prompt)
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

# # === Dev Testing ===
# if __name__ == "__main__":
#     test_questions = [
#         "What is MPCE?",
#         "Compare expenses of female-headed and male-headed households.",
#         "How many working people are in the data?",
#         "Is there any relation between sector and average expense?",
#         "Tell me the impact of marital status on expense.",
#         "Give insights into MPCE variations."
#     ]

#     for q in test_questions:
#         print(f"\n🧠 Q: {q}")
#         print(f"💬 {chat(q)}")
