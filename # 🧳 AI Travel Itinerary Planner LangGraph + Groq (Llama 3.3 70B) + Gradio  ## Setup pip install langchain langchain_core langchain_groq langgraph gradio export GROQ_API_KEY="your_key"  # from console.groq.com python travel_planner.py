# ============================================================
# AI Travel Itinerary Planner
# Built using LangChain + LangGraph + Groq (Llama 3.3 70B) + Gradio
# ============================================================
# Install dependencies:
# pip install langchain langchain_core langchain_groq langgraph gradio

import os
import time
from typing import TypedDict, Annotated, List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import gradio as gr

# ============================================================
# STEP 1: API Key Setup
# Set your Groq API key as environment variable:
# export GROQ_API_KEY="your_actual_api_key"
# Get free API key at: https://console.groq.com
# ============================================================

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set. "
        "Please set it before running: export GROQ_API_KEY='your_key'"
    )

# ============================================================
# STEP 2: State Definition (LangGraph)
# ============================================================

class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str

# ============================================================
# STEP 3: LLM Initialization (Groq — Llama 3.3 70B)
# ============================================================

llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# ============================================================
# STEP 4: Prompt Template
# ============================================================

itinerary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful travel assistant. Create a detailed day trip itinerary "
        "for {city} based on the user's interests: {interests}. "
        "Provide a well-structured, bulleted itinerary with timing for each activity."
    ),
    ("human", "Create an itinerary for my day trip."),
])

# ============================================================
# STEP 5: LangGraph Node Functions
# ============================================================

def input_city(city: str, state: PlannerState) -> PlannerState:
    """Node: Accept city input from user."""
    return {
        **state,
        "city": city,
        "messages": state.get("messages", []) + [HumanMessage(content=city)],
    }

def input_interests(interests: str, state: PlannerState) -> PlannerState:
    """Node: Accept interests input from user."""
    return {
        **state,
        "interests": [i.strip() for i in interests.split(",")],
        "messages": state["messages"] + [HumanMessage(content=interests)],
    }

def create_itinerary(state: PlannerState) -> str:
    """Node: Generate itinerary using Groq LLM with retry logic."""
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            response = llm.invoke(
                itinerary_prompt.format_messages(
                    city=state["city"],
                    interests=", ".join(state["interests"])
                )
            )
            state["itinerary"] = response.content
            state["messages"] += [AIMessage(content=response.content)]
            return response.content

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return "Error: Could not generate itinerary. Please try again later."

# ============================================================
# STEP 6: LangGraph Workflow
# ============================================================

def build_graph():
    workflow = StateGraph(PlannerState)
    workflow.add_node("input_city", lambda state: state)       # Handled externally via Gradio
    workflow.add_node("input_interest", lambda state: state)   # Handled externally via Gradio
    workflow.add_node("create_itinerary", create_itinerary)
    workflow.set_entry_point("input_city")
    workflow.add_edge("input_city", "input_interest")
    workflow.add_edge("input_interest", "create_itinerary")
    workflow.add_edge("create_itinerary", END)
    return workflow.compile()

app = build_graph()

# ============================================================
# STEP 7: Main Travel Planner Function
# ============================================================

def travel_planner(city: str, interests: str) -> str:
    """Main function: takes city + interests, returns itinerary."""
    if not city.strip():
        return "Please enter a city name."
    if not interests.strip():
        return "Please enter at least one interest."

    state: PlannerState = {
        "messages": [],
        "city": "",
        "interests": [],
        "itinerary": "",
    }

    state = input_city(city, state)
    state = input_interests(interests, state)
    itinerary = create_itinerary(state)
    return itinerary

# ============================================================
# STEP 8: Gradio UI
# ============================================================

interface = gr.Interface(
    fn=travel_planner,
    inputs=[
        gr.Textbox(label="🌍 City", placeholder="e.g. Paris, Tokyo, Jaipur"),
        gr.Textbox(label="🎯 Your Interests (comma-separated)", placeholder="e.g. art, history, food, adventure"),
    ],
    outputs=gr.Textbox(label="📅 Your Personalized Itinerary", lines=20),
    title="🧳 AI Travel Itinerary Planner",
    description="Powered by LangGraph + Groq (Llama 3.3 70B). Enter a city and your interests to get a personalized day trip plan.",
    examples=[
        ["Paris", "art, history, cafes"],
        ["Tokyo", "anime, food, technology"],
        ["Jaipur", "history, architecture, street food"],
    ]
)

if __name__ == "__main__":
    interface.launch()
