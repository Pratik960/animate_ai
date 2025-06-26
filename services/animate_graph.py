import os
import re
import json
from typing import Tuple, TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from services.conversion_service import ConversionService
from prompts.prompts import CODE_GENERATION_PROMPT, INTENT_CLASSIFICATION_PROMPT, SCENE_PLANNER_PROMPT

# Load environment variables
load_dotenv()

# LLM Client initialization
client = ChatGroq(temperature=0, model_name="mistral-saba-24b", verbose=True)
code_generation_client = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", verbose=True)
# Define the custom state
class AppState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    scene_description: str
    scenes: List[dict]
    intent: str
    manim_code: str
    video_path: str


def classify_intent(state: AppState) -> AppState:
    """Classify the intent of the scene description."""
    messages = INTENT_CLASSIFICATION_PROMPT.format_messages(scene_description=state["scene_description"])
    
    # Invoke the language model
    response = client.invoke(messages)
    intent = response.content.strip()
    intent = response.content.strip()
    valid_intents = ["GREETINGS", "GUARDRAILS", "MANIM_NOT_POSSIBLE", "MANIM_VIDEO"]
    if intent not in valid_intents:
        print(f"Unexpected intent: {intent}. Defaulting to GUARDRAILS.")
        intent = "GUARDRAILS"
    return {"intent": intent}

def greetings_response(state: AppState) -> AppState:
    """Return a greeting response."""
    return {
        "messages": state["messages"] + [
            AIMessage(content="Hello! How can I assist you with your animation today?")
        ]
    }

def guardrails_response(state: AppState) -> AppState:
    """Return a guardrails response."""
    return {
        "messages": state["messages"] + [
            AIMessage(content="I'm specialized in helping you with manim animation. Please provide a video description for me to generate the code.")
        ]
    }

def manim_not_possible_response(state: AppState) -> AppState:
    """Return a response indicating the request is not possible."""
    return {
        "messages": state["messages"] + [
            AIMessage(content="I'm sorry, but I cannot generate a video for that description. Please provide a different scene description that is feasible with Manim.")
        ]
    }


def scene_planner(state: AppState) -> AppState:
    """Generate a structured scene plan based on the description."""
    prompt = SCENE_PLANNER_PROMPT + f"\n\nDescription: {state['scene_description']}"
    reply = client.invoke(prompt).content.strip()
    print(f"Scene planner reply: {reply}")
    # Remove markdown code block markers if present
    cleaned_reply = reply
    if reply.startswith("```json"):
        cleaned_reply = reply[len("```json"):].rstrip("```").strip()
    try:
        response = json.loads(cleaned_reply)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse response as JSON: {e}")
    if not isinstance(response, list):
        raise ValueError("Scene planner response is not a list.")
    if not all(isinstance(item, dict) and "scene_number" in item and "description" in item for item in response):
        raise ValueError("Scene planner response does not contain valid scene dictionaries.")
    return {
        "messages": state["messages"] + [
            AIMessage(content="Scene plan generated successfully.")
        ],
        "scenes": response,
        "intent": "MANIM_VIDEO"
    }


def generate_manim_code(state: AppState) -> AppState:
    """Generate Manim code and convert it to a video."""
    # Include conversation history in the prompt for context
    prompt = CODE_GENERATION_PROMPT + f"\n\nScenes: {json.dumps(state['scenes'], indent=2)}"
    reply = code_generation_client.invoke(prompt).content.strip()
    print(f"Generated Manim code: {reply}")

    # Extract the code block
    pattern = re.compile(r"```python\n([\s\S]*?)```")
    match = pattern.search(reply)
    # m = re.search(r"<ani-ai-code>(.*?)</ani-ai-code>", reply, flags=re.DOTALL)
    if not match:
        return {
            "messages": state["messages"] + [
                AIMessage(content="No code found in assistant reply.")
            ]
        }
    correct_code = match.group(1).strip()

    # Render to video
    converter = ConversionService()
    video_path = converter.convert(correct_code)

    if not os.path.exists(video_path):
        return {
            "messages": state["messages"] + [
                AIMessage(content=f"Video {video_path} not found.")
            ]
        }

    return {
        "messages": state["messages"] + [
            AIMessage(content="Please wait while your video is getting rendered.")
        ],
        "manim_code": correct_code,
        "video_path": video_path
    }

# Set up the workflow
workflow = StateGraph(AppState)

# Add nodes
workflow.add_node("intent_classifier", classify_intent)
workflow.add_node("greetings", greetings_response)
workflow.add_node("guardrails", guardrails_response)
workflow.add_node("manim_not_possible", manim_not_possible_response)
workflow.add_node("scene_planner", scene_planner)
workflow.add_node("code_generator", generate_manim_code)

# Define edges
workflow.add_edge(START, "intent_classifier")
workflow.add_conditional_edges(
    "intent_classifier",
    lambda state: state["intent"],
    {
        "GREETINGS": "greetings",
        "GUARDRAILS": "guardrails",
        "MANIM_NOT_POSSIBLE": "manim_not_possible",
        "MANIM_VIDEO": "scene_planner"
    }
)
workflow.add_edge("scene_planner", "code_generator")
workflow.add_edge("greetings", END)
workflow.add_edge("guardrails", END)
workflow.add_edge("manim_not_possible", END)
workflow.add_edge("code_generator", END)

# Configure memory for session persistence
checkpointer = MemorySaver()

# Compile the graph
app = workflow.compile(checkpointer=checkpointer)



def process_scene_description(scene_description: str, convId: str) -> Tuple[str,str,str]:
    initial_state = AppState(
        messages=[],
        scene_description=scene_description,
        scenes=[],
        intent="",
        manim_code="",
        video_path=""
    )

    final_state = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": convId}}
    )
    print(f"Final state: {final_state}")
    return final_state["messages"][-1].content, final_state.get("video_path", ""),final_state.get("manim_code","")

def get_video_by_name(video_name: str):
    """Get the video by its name."""
    video_path = os.path.join("temp", "videos", video_name)
    if os.path.exists(video_path):
        with open(video_path, "rb") as video_file:
            while True:
                data = video_file.read(1024 * 1024)
                if not data:
                    break
                yield data
    else:
        raise FileNotFoundError(f"Video {video_name} not found.")