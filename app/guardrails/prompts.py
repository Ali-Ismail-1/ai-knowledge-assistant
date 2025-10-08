# app/guardrails/prompts.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

BASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful assistant. Use the provided context to answer questions.
    If the context or chat history does not contain the answer, say "I don't know" and 
    politely suggest to the user that they can ask a different question.
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Question: {input}\n\nContext: \n{context}"),
])