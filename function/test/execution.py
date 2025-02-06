from typing import Optional, Dict
import os
import tempfile
import subprocess

def check_correctness(problem: Dict, completion: str, timeout: float, test: list, requirement: str = None, ) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 
    """

    result = []

    try:
        # Construct the check program
        check_program = completion + "\n" + "\n".join(test) + "\n"

        temp_file_path = tempfile.mktemp(suffix=".py")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        with open(temp_file_path, mode="w") as temp_file:
            temp_file.write(check_program)

        # Execute the code using a subprocess
        process = subprocess.run(
            ["python", temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True
        )
        # Check the process results
        if process.returncode == 0:
            result.append("passed")
        else:
            result.append(f"failed: {process.stderr.strip()}")

    except subprocess.TimeoutExpired:
        result.append("timed out")
    except BaseException as e:
        result.append(f"failed: {e}")
    finally:
        # Remove the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # Return the result in the required format
    return dict(
        task_id=problem["task_id"],
        passed=result[0] == "passed",
        result=result[0],
        requirement=requirement,
    )


# Custom exception for timeout
class TimeoutException(Exception):
    pass


# Simplified context manager to mimic a temporary directory (not really used here)
class DummyTempDir:
    def __enter__(self):
        # Just a placeholder, no temp dir needed
        return "tempdir"

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# A simple check function to replace reliability_guard (remove destructive functions)
def reliability_guard():
    # No special actions for now, we keep it simple
    pass


def chdir(root):
    # Simplified version: No directory switching needed
    pass
