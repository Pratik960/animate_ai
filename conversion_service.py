from datetime import datetime
import random
import shutil
import string
import os
import subprocess
import imageio_ffmpeg as _ffmpeg

class ConversionService:
    def __init__(self, base_dir: str = None):
        """
        :param base_dir: Base working directory for temp files.
                         Defaults to current working directory.
        """
        # Use an explicit base directory so that cwd is predictable
        self.base_dir = base_dir or os.getcwd()
        self.code_dir = os.path.join(self.base_dir, 'temp', 'code')
        self.video_dir = os.path.join(self.base_dir, 'temp', 'videos')
        os.makedirs(self.code_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

    def convert(self, code: str) -> str:
        """
        Convert a Manim scene (as Python code) to an MP4 video.

        :param code: The full content of the .py file, defining a Scene subclass called MainScene
        :return: The path to the generated MP4 file, or an error message.
        """

        if not code.strip():
            raise ValueError("Code cannot be empty.")

        # Ensure that the bundled ffmpeg lives on PATH for subprocess, Manim, pydub, etc.
        ffmpeg_exe = _ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        # Generate a unique filename and output name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        script_name = f"manim_scene_{timestamp}_{rand}.py"
        output_filename = f"MainScene_{timestamp}_{rand}.mp4"

        # Paths on the filesystem
        code_path = os.path.join(self.code_dir, script_name)
        output_path = os.path.join(self.video_dir, output_filename)

        # Write out the user’s Manim script
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code)

        # Build the manim command
        # We specify absolute paths for code_path and output_dir,
        # and run in the project root so that manim can find its own assets.
        command = [
            'manim',      # ensure this is on your PATH, or use full path
            'render',
            '-qm',        # medium quality (720p30)
            '--format', 'mp4',
            '-o', output_path,
            '--media_dir', 'temp/media',
            code_path,
            'MainScene'

        ]

        try:
            # Run Manim from the base directory so that relative imports and configs work
            result = subprocess.run(
                command,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )

            if result.returncode != 0:
                # Propagate stderr for debugging
                return f"Manim command failed (exit code {result.returncode}):\n{result.stderr.strip()}"

            # Confirm output file exists
            if not os.path.isfile(output_path):
                return "Error: Expected output file not found after rendering."

            return output_path

        except subprocess.TimeoutExpired:
            return "Error: Video generation timed out after 5 minutes."
        except FileNotFoundError as e:
            # Likely manim not installed or missing ffmpeg; report clearly
            return f"Error: {e.strerror}. Command or dependencies may be missing."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
        
        # finally:
        #     # Cleanup temporary directory
        #     shutil.rmtree('temp', ignore_errors=True)
