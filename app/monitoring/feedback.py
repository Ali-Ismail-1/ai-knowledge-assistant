# app/monitoring/feedback.py
def record_feedback(session_id: str, question: str, answer: str, rating: int):
    # Stub: wire to DB or file in production
    print(f"[FEEDBACK] sid={session_id} rating={rating} q={question[:60]} a={answer[:60]}")
