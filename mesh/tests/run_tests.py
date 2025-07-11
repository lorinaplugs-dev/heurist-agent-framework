import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


class PythonScriptRunner:
    def __init__(self, max_workers=5):
        self.max_workers = max_workers
        self.success_count = 0
        self.failure_count = 0
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.current_script = os.path.basename(__file__)
        self.exclude_files = [self.current_script]  # Add more filenames here to exclude
        self.files_to_run = self._discover_scripts()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename=f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        self.logger = logging.getLogger(__name__)

    def _discover_scripts(self):
        return sorted([f for f in os.listdir(self.script_dir) if f.endswith(".py") and f not in self.exclude_files])

    def run_script(self, file_name):
        file_path = os.path.join(self.script_dir, file_name)
        log_name = os.path.splitext(file_name)[0]

        print(f"FILE: {file_name}")
        print(f"LOG FOR {log_name}:")

        try:
            process = subprocess.Popen(
                ["python", "-u", file_path],
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
            )
            process.wait()

            if process.returncode == 0:
                self.logger.info(f"Successfully executed {file_name}")
                print(f"✅ Success: {file_name}")
                return True
            else:
                self.logger.error(f"Failed to execute {file_name} with return code {process.returncode}")
                print(f"❌ Failed: {file_name}")
                return False
        except Exception as e:
            self.logger.error(f"Error running {file_name}: {str(e)}")
            print(f"❌ Error running {file_name}: {str(e)}")
            return False
        finally:
            print(f"Completed agent {log_name}\n")

    def execute_all(self):
        if not self.files_to_run:
            print("No Python files found to execute.")
            return

        print(f"Discovered {len(self.files_to_run)} Python files.")
        futures = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for file in self.files_to_run:
                futures.append(executor.submit(self.run_script, file))

            for future in as_completed(futures):
                if future.result():
                    self.success_count += 1
                else:
                    self.failure_count += 1

        self._print_summary()

    def _print_summary(self):
        print("\nExecution Summary:")
        print(f"✅ Success: {self.success_count}")
        print(f"❌ Failed:  {self.failure_count}")
        self.logger.info(f"Execution Summary - Success: {self.success_count}, Failed: {self.failure_count}")


if __name__ == "__main__":
    runner = PythonScriptRunner(max_workers=5)
    runner.execute_all()
