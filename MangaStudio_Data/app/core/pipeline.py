import os
import json
import subprocess
import gc
import shutil

try:
    import torch
except ImportError:
    torch = None


class Pipeline:
    """Handles the execution of the backend translation process."""

    def __init__(self, app, python_executable, temp_dir):
        """Handles the execution of the backend translation process."""
        self.app = app
        self.python_executable = python_executable
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        self.process = None
        self._stopped_by_user = False

    def run(self, job, output_path, config_dict, log_callback, is_verbose=False, output_format='png'):
        """
        Runs a job using a provided configuration dictionary.
        Now supports passing all command line arguments.
        """
        log_callback("PIPELINE", f"Starting job '{os.path.basename(job['source_path'])}'.")
        self._stopped_by_user = False
        config_path = ""

        try:
            # --- Extract all potential CLI arguments from the config dict ---
            font_path = config_dict.pop('font_path', None)
            pre_dict_path = config_dict.pop('pre_dict_path', None)
            post_dict_path = config_dict.pop('post_dict_path', None)

            config_path = os.path.join(self.temp_dir, f"temp_config_job_{job['id']}.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4)

            command = [self.python_executable, "-m", "manga_translator", "local", "-i", job['source_path'], "-o", output_path, "--config-file", config_path]
            
            # --- Add all optional arguments if they exist ---
            if is_verbose:
                command.append("-v")
            if output_format:
                command.extend(["--format", output_format])
            if font_path:
                # The path should be absolute for the backend
                command.extend(["--font-path", os.path.join(self.app.project_base_dir, font_path)])
            if pre_dict_path:
                command.extend(["--pre-dict", os.path.join(self.app.project_base_dir, pre_dict_path)])
            if post_dict_path:
                command.extend(["--post-dict", os.path.join(self.app.project_base_dir, post_dict_path)])

            if job['settings'].get('processing_device') == 'NVIDIA GPU':
                command.append("--use-gpu")
            
            return self._execute_subprocess(log_callback, command)
        except Exception as e:
            log_callback("ERROR", f"Critical error preparing job: {e}")
            return False
        finally:
            if config_path and os.path.exists(config_path):
                try: os.remove(config_path)
                except: pass
            self._cleanup_memory(log_callback)

    def run_single_image_test(self, test_image_path, output_path, config_dict, log_callback, is_verbose=False):
        """Runs the pipeline for a single test image using a provided config."""
        log_callback("PIPELINE", f"Starting visual test for: {os.path.basename(test_image_path)}")
        self._stopped_by_user = False

        temp_input_dir = os.path.join(self.temp_dir, "visual_test_input")
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
        os.makedirs(temp_input_dir)

        config_path = ""

        try:
            shutil.copy(test_image_path, temp_input_dir)

            output_format = config_dict.pop('output_format_cli', None)

            config_path = os.path.join(self.temp_dir, "temp_config_visual_test.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4)

            command = [self.python_executable, "-m", "manga_translator", "local", "-i", temp_input_dir, "-o", output_path, "--config-file", config_path]

            if is_verbose:
                command.append("-v")

            if output_format:
                command.extend(["-f", output_format])

            if config_dict.get('processing_device') == 'NVIDIA GPU':
                command.append("--use-gpu")

            return self._execute_subprocess(log_callback, command)
        except Exception as e:
            log_callback("ERROR", f"Visual test preparation failed: {e}")
            return False
        finally:
            if os.path.exists(temp_input_dir):
                shutil.rmtree(temp_input_dir)
            if config_path and os.path.exists(config_path):
                try:
                    os.remove(config_path)
                except:
                    pass
            self._cleanup_memory(log_callback)

    def _execute_subprocess(self, log_callback, command):
        """
        Executes the given command, now also checks the output stream for error keywords
        to determine the true success of the operation.
        """
        log_callback("DEBUG", f"Executing: {' '.join(command)}")
        my_env = os.environ.copy()
        my_env["PYTHONUTF8"] = "1"

        has_failed = False  # Flag to track if we've seen an error message

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=my_env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            for line in iter(self.process.stdout.readline, ''):
                if self._stopped_by_user:
                    log_callback("WARNING", "User interrupt received, stopping process.")
                    break

                stripped_line = line.strip()
                if stripped_line:
                    log_callback("RAW", stripped_line)
                    # --- NEW ERROR CHECKING ---
                    # Check for keywords that indicate a failure, even if the process exits cleanly.
                    if stripped_line.startswith("ERROR:") or "Traceback (most recent call last)" in stripped_line:
                        has_failed = True

            return_code = self.process.wait()

            if self._stopped_by_user:
                return False

            # The job is only successful if the return code is 0 AND we haven't seen any error flags.
            return return_code == 0 and not has_failed

        except Exception as e:
            if not self._stopped_by_user:
                log_callback("ERROR", f"Subprocess execution failed: {e}")
            return False
        finally:
            self.process = None

    def stop(self, log_callback):
        """Stops the currently running subprocess."""
        if self.process and self.process.poll() is None:
            log_callback("PIPELINE", "Attempting to terminate running process...")
            self._stopped_by_user = True
            try:
                self.process.kill()
                log_callback("SUCCESS", "Process terminated by user.")
                return True
            except Exception as e:
                log_callback("ERROR", f"Failed to kill process: {e}")
                return False
        return False

    def _cleanup_memory(self, log_callback):
        log_callback("DEBUG", "Performing memory cleanup...")
        gc.collect()
        if torch and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
