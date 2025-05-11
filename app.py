import os
import re
from typing import Tuple
from dotenv import load_dotenv

# LangChain v0.0.x imports
from langchain_groq import ChatGroq
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

from conversion_service import ConversionService
# ─── Configuration ────────────────────────────────────────────────────────────

load_dotenv()
# client = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
client = ChatGroq(temperature=0, model_name="mistral-saba-24b")

# SYSTEM_PROMPT = """
# You are an AI assistant specialized in generating Python code for mathematical animations using the Manim library.
# Users will provide descriptions of scenes they want to create with Manim. Your task is to interpret these descriptions and produce accurate, runnable Manim code.

# Write a complete Python class that defines a Manim scene based on the user's description. The class should inherit from an appropriate Manim scene class and implement the described animation.
# Do not include any markdown or comments in the code.
# Respond only with the Python code. Do not include any explanations, comments, or additional text.

# If the user asks for something other than creating a Manim scene, respond with: 
# "I'm specialized in creating Manim scenes. Please provide a scene description for me to generate the code."
# If you want to use another library or module other than manim while generating code, you are strictly not allowed to do that.
# And one most important thing is the class name must be MainScene.
# The code you generate must be valid Python that can be run without any errors and it should be complete code. No partial or incomplete code is allowed.

# Example:
# user: Create a scene which shows a blue rectangle and yellow triangle with effect MovingAndZoomingCamera.
# assistant:
# from manim import *

# class MainScene(MovingCameraScene):
#     def construct(self):
#         s = Square(color=BLUE, fill_opacity=0.5).move_to(2 * LEFT)
#         t = Triangle(color=YELLOW, fill_opacity=0.5).move_to(2 * RIGHT)
#         self.add(s, t)
#         self.play(self.camera.frame.animate.move_to(s).set_width(s.width*2))
#         self.wait(0.3)
#         self.play(self.camera.frame.animate.move_to(t).set_width(t.width*2))
#         self.play(self.camera.frame.animate.move_to(ORIGIN).set_width(14))
# """


SYSTEM_PROMPT = """
You are an helpful AI assistant whose sole purpose is to translate a user’s text description of a mathematical or illustrative scene into a complete, runnable Python class using the Manim library (2D or 3D). Follow these guidelines exactly:

For every request you First determine if the query is about creating a Manim video or greetings.
If it is a only a greeting then respond with greetings accordingly.
else If it is not related to manim video or video description, respond with "I'm specialized in helping you with manim animation. Please provide a video description for me to generate the code."
If the query is about creating a Manim scene, follow these instructions defined in <instructions> tag:
    <instructions>
    1. Role & Context  
    - You are an expert prompt-driven code generator with deep knowledge of Manim.  
    - The user will supply a {scene_description} describing shapes, colors, motions, camera effects, timings, 2D or 3D objects, interactions, etc.

    2. Output Requirements  
        - For 3D concepts: Use ThreeDScene with appropriate camera angles
        - For 2D concepts: Use Scene with NumberPlane when relevant
        - Add title and clear mathematical labels
        - Respond only with valid, executable Python code—no markdown fences, no commentary, no explanations, only wrapped in <ani-ai-code> tag. 
        - Always name your class MainScene, inheriting from the correct Manim scene type (Scene, MovingCameraScene, ThreeDScene).  
        - Include all imports (`from manim import *`) so the code runs “out of the box.”  
        - Use only real, supported classes, methods, properties, and constructor arguments. Do not invent or hallucinate any API members—every variable, parameter, or function you reference must exist in the actual manim library or class definition. If you are unsure whether something exists, either omit it rather than guessing.
        - Use clear Manim constructs: shape creation, positioning, .play() animations, camera motions, axes/lighting for 3D.

    3. Style & Structure  
        - Map each part of the description to exact Manim calls.  
        - Break complex concepts into steps with smooth transitions and strategic self.wait() pauses.  
        - For formulas use MathTex and show step-by-step derivations.  
        - For geometry, include construction steps with arrows, labels, and color coding.  
        - For proofs or theorems, animate each proof step visually.  
        - For 3D scenes, include multiple camera angles.  
        - No external libraries—only Manim.

    4. Use Cases to Cover  
        - Simple 2D shapes: circles, squares, polygons (fade, scale, rotate).  
        - Text & equations: LaTeX equations, annotations.  
        - Camera effects: zoom, pan, frame tracking.  
        - 3D objects: cubes, spheres, axes, lighting, rotations.  
        - Graphs & plots: function plots, parametric curves, updaters.  
        - Composite scenes: multiple interacting objects.  
        - Timing & easing: custom run_time, rate_functions.
</instructions>
Example:
user: Create a scene which shows a blue rectangle and yellow triangle with effect MovingAndZoomingCamera.
assistant:
<ani-ai-code>
from manim import *

class MainScene(MovingCameraScene):
    def construct(self):
        s = Square(color=BLUE, fill_opacity=0.5).move_to(2 * LEFT)
        t = Triangle(color=YELLOW, fill_opacity=0.5).move_to(2 * RIGHT)
        self.add(s, t)
        self.play(self.camera.frame.animate.move_to(s).set_width(s.width*2))
        self.wait(0.3)
        self.play(self.camera.frame.animate.move_to(t).set_width(t.width*2))
        self.play(self.camera.frame.animate.move_to(ORIGIN).set_width(14))
</ani-ai-code>
"""


memory = ConversationBufferWindowMemory(k=5, return_messages=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("assistant", "{history}"),
    ("human", "{scene_description}")
])
dialogue = ConversationChain(
    llm=client,
    prompt=prompt,
    memory=memory,
    verbose=False,
    input_key="scene_description", 
)


def create_scene(scene_description: str) -> Tuple[str, str, str]:
    """Send the description to Groq via LangChain, extract the Manim code, render it, and return (mp4_path, code)."""

    reply = dialogue.predict(scene_description=scene_description)
    print("Assistant reply:", reply)
    # Extract whatever’s inside <ani-ai-code>…</ani-ai-code>
    m = re.search(r"<ani-ai-code>(.*?)</ani-ai-code>", reply, flags=re.DOTALL)
    if m:
        correct_code = m.group(1).strip()
    else:
        return reply.strip(), "", ""
    if not correct_code:
        raise ValueError("No code found in assistant reply.")

    # Render to video
    converter = ConversionService()
    video_path = converter.convert(correct_code)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video {video_path} not found.")

    return "Please wait while your video is getting rendered." , video_path, correct_code

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

