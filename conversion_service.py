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

        # Ensure FFmpeg is available
        ffmpeg_exe = _ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

        # Generate unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        script_name = f"manim_scene_{timestamp}_{rand}.py"
        output_filename = f"MainScene_{timestamp}_{rand}.mp4"

        # Define file paths
        code_path = os.path.join(self.code_dir, script_name)
        output_path = os.path.join(self.video_dir, output_filename)

        try:
            # Prepend configuration lines to enable parallel rendering and set FFmpeg path
            config_lines = f"""
from manim import config
config.num_workers = 4
config.ffmpeg_executable = r'{ffmpeg_exe}'
"""
            modified_code = config_lines + code

            # Write the modified code to the temporary script file
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(modified_code)

            # Build the Manim command without --config_file
            command = [
                'manim',
                'render',
                '-qm',  # Medium quality (720p30)
                '--format', 'mp4',
                '-o', output_path,
                '--media_dir', 'temp/media',
                code_path,
                'MainScene'
            ]

            # Execute the Manim command
            result = subprocess.run(
                command,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=300  # 5-minute timeout
            )

            if result.returncode != 0:
                return f"Manim command failed (exit code {result.returncode}):\n{result.stderr.strip()}"

            if not os.path.isfile(output_path):
                return "Error: Expected output file not found after rendering."

            return output_path

        except subprocess.TimeoutExpired:
            return "Error: Video generation timed out after 5 minutes."
        except FileNotFoundError as e:
            return f"Error: {e.strerror}. Command or dependencies may be missing."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

        finally:
            # Clean up temporary script file
            if os.path.exists(code_path):
                os.remove(code_path)
            # Remove the specific media directory for this conversion
            script_base_name = os.path.splitext(os.path.basename(code_path))[0]
            media_subdir = os.path.join(self.base_dir, 'temp', 'media', 'videos', script_base_name)
            if os.path.exists(media_subdir):
                shutil.rmtree(media_subdir, ignore_errors=True)
               