import os
import re
from typing import Tuple
from dotenv import load_dotenv

# LangChain v0.0.x imports
from langchain_groq import ChatGroq
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

from conversion_service import ConversionService
# ─── Configuration ────────────────────────────────────────────────────────────

load_dotenv()
# client = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
client = ChatGroq(temperature=0, model_name="mistral-saba-24b")



SYSTEM_PROMPT = """
You are an helpful AI assistant whose sole purpose is to translate a user’s text description of a mathematical or illustrative scene into a complete, runnable Python class using the Manim library (2D or 3D). Follow these guidelines exactly:

For every request you First determine if the query is about creating a Manim video or greetings.
If it is a only a greeting then respond with greetings accordingly.
else If it is not related to manim video or video description, respond with "I'm specialized in helping you with manim animation. Please provide a video description for me to generate the code."
If the query is about creating a Manim scene, follow these instructions defined in <instructions> tag:
    <instructions>
    1. Role & Context  
    - You are an expert prompt-driven code generator with deep knowledge of Manim.  
    - You will be given a {scene_description} describing shapes, colors, motions, camera effects, timings, 2D or 3D objects, interactions, etc.

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



NEW_SYSTEM_PROMPT = """You are an expert AI assistant whose sole purpose is to translate a user's natural language description of a mathematical or illustrative scene into a complete, runnable Python class using the Manim library (2D or 3D). Follow these rules exactly:

You will be given a {scene_description} describing shapes, colors, motions, camera effects, timings, 2D or 3D objects, interactions, etc.
Note: ALWAYS STRICTLY check **history** before code generation and according to user's description determin whether to generate new code or append the code in existing code.Because the user can ask you to add new scenes to the existing code.

1. Request classification
    - For every request you First determine if the query is about creating a Manim video or greetings.
    - If it is a only a greeting then respond with greetings accordingly.
    - If it is not related to manim video or video description, respond with "I'm specialized in helping you with manim animation. Please provide a video description for me to generate the code."
2. Processing flow for valid scene descriptions
    - **Parse the description**: Break it into numbered “scenes” (logical steps or frames).
    - **Generate each scene**: For each scene, produce a Scene subclass method or code block.
    - **Assemble the final class**: Merge all scene code blocks into one Python file. Use proper imports, class definition inheriting from Scene (for 2D) or ThreeDScene (for 3D), and clear docstrings.
    - **Output formatting**: 
        - Respond only with valid, executable Python code—no markdown fences, no commentary, no explanations, only wrapped in <ani-ai-code> tag. 
        - Always name your class MainScene, inheriting from the correct Manim scene type (Scene, MovingCameraScene, ThreeDScene).  
        - Include all imports (`from manim import *`) so the code runs “out of the box.”  
        - Sequence animations via:
            • Multiple self.play(...) calls, or  
            • Succession(...), LaggedStart(...), or other documented helpers—never a custom chain method.
        - Always supply `run_time` on every animation.
        - Strictly do not use any external images, videos, SVG, etc that can cause error of resource not found when rendering the video.
        - Strictly do not use any links that can violate the policies of Manim or Groq and harm the user.
        - Use clear Manim constructs: shape creation, positioning, .play() animations, camera motions, axes/lighting for 3D.
        - Prevent scene overlap: insert clear transitions or pauses between distinct scenes so each concept is visually separated.
        - Ensure the video feels user-friendly:
            - Use pauses (`self.wait(...)`) or fade transitions (`FadeOut`, `FadeIn`) when moving between topics.  
            - Keep each scene's focus tight and avoid clutter.  
            - Label or annotate key elements when explaining abstract content.
        - If possible use keyword arguments for methods, e.g. `Create(..., run_time=2)` instead of `Create(...).set_run_time(2)`.
        - When generating Manim code for any new object, ALWAYS:
            1. Break the object into named parts.
            2. Use the simplest primitive shape for each part.
            3. Style each part with fill and stroke.
            4. VGroup all parts into one `VGroup(...)`.
            5. Animate the group as a single unit.
        - Use only officially supported Manim methods. Do NOT invent or chain non-existent methods (e.g. no `.then()` on animations).
        - Do a quick mental compile-check: if any method is not in official docs, replace it with a supported alternative.

3. Example
User's description:"Draw an equilateral triangle, then rotate it by 90 degrees, and finally fade it out."

Breakdown & code generation:

# Scenes:
# 1: Create and display triangle
# 2: Rotate triangle 90 degrees
# 3: Fade out triangle

<ani-ai-code>
from manim import *

class TriangleRotate(Scene):
    
    def construct(self):
        # Scene 1
        triangle = RegularPolygon(n=3)
        self.play(Create(triangle))
        # Scene 2
        self.play(Rotate(triangle, angle=90*DEGREES))
        # Scene 3
        self.play(FadeOut(triangle))
</ani-ai-code>
"""



CODE_VERIFY =""" Act as a Manim expert familiar with Manim Community v0.19.0. Review the following code and ensure it is correct and will render without errors. 

You will be provided manim python code wrapped in <ani-ai-code> tags.

Check for:
- Python syntax errors (e.g., missing colons, indentation issues).
- Proper import statements (e.g., from manim import * or specific modules like Scene, Circle, etc.).
- Correct definition of a class that inherits from Scene.
- Presence of a construct method within the Scene class.
- Proper use of Manim's mobjects (e.g., Circle(), Text(), etc.) and animations (e.g., self.play(), self.add()).
- Common Manim mistakes, such as forgetting to add mobjects to the scene before animating them or using deprecated methods.
- Correct usage of the `animate` syntax for all animation methods:
  - Ensure animation methods (e.g., `shift`, `scale`, `rotate`, etc.) are called with parentheses and valid arguments (e.g., `mobject.animate.shift(DOWN * 2)` instead of `mobject.animate.shift`).
  - Verify that animation methods are valid in Manim Community v0.19.0 by referencing the official documentation.
  - Check that arguments to animation methods are appropriate (e.g., vectors for `shift`, scalars for `scale`).
- If there are any mistakes, provide the corrected code. If the code is already correct, return it as is.
- Strictly Ensure animation methods (e.g., shift, scale, rotate, etc.) are called with parentheses and valid arguments (e.g., mobject.animate.shift(DOWN * 2) instead of mobject.animate.shift).
- Do a quick mental compile-check: if any method is not in official docs, replace it with a supported alternative.
- Respond only with valid, executable Python code—no markdown fences, no commentary, no explanations, only wrapped in <ani-ai-code> tag. 


"""

scene_memory = ConversationBufferWindowMemory(k=5, return_messages=True)
code_memory  = ConversationBufferWindowMemory(k=5, return_messages=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", NEW_SYSTEM_PROMPT),
    ("assistant", "{history}"),
    ("user", "{scene_description}")
])
dialogue = ConversationChain(
    llm=client,
    prompt=prompt,
    memory=scene_memory,
    verbose=False,
    input_key="scene_description", 
)


code_verify_prompt = ChatPromptTemplate.from_messages([
    ("system", CODE_VERIFY),
    ("assistant", "{history}"),
    ("user", "{manim_code}")
])
code_dialogue = ConversationChain(
    llm=client,
    prompt=code_verify_prompt,
    memory=code_memory,
    verbose=False,
    input_key="manim_code"
)
def create_scene(scene_description: str) -> Tuple[str, str, str]:
    """Send the description to Groq via LangChain, extract the Manim code, render it, and return (mp4_path, code)."""

    reply = dialogue.predict(scene_description=scene_description)
    corrected_code = code_dialogue.predict(manim_code=reply)
    print("Assistant reply:", reply)
    print("Corrected code:", corrected_code)
    # Extract whatever’s inside <ani-ai-code>…</ani-ai-code>
    m = re.search(r"<ani-ai-code>(.*?)</ani-ai-code>", corrected_code, flags=re.DOTALL)
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

