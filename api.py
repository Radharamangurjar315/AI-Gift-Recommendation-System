# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from llm import recommend_gifts

app = FastAPI(title="Gift Recommender")

class UserInput(BaseModel):
    occasion: str
    age: int
    gender: str
    interests: str
    budget_min: int
    budget_max: int

@app.post("/recommend")
def recommend(user_input: UserInput):
    try:
        recs = recommend_gifts(user_input.dict())
        return {"recommendations": recs}
    except Exception as e:
        return {"error": f"Server error: {e}"}
