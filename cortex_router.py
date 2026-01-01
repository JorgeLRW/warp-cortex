import re

class CortexRouter:
    """
    The Dynamic Delegator.
    Instead of hardcoded roles, it extracts 'Tasks' from the stream.
    The Side Agent is a generic worker that executes whatever task is delegated.
    """
    def __init__(self):
        # Regex patterns to capture the task description
        self.delegation_patterns = [
            # Explicit: [TASK: Check the math]
            r"\[TASK:\s*(.*?)\]",
            # Explicit: [DELEGATE: Write a script]
            r"\[DELEGATE:\s*(.*?)\]",
            # Implicit: "I need to [SEARCH] for X" -> Task: "Search for X"
            # We'll keep it simple for now and just detect the intent keyword
        ]
        
        # Fallback implicit triggers
        self.implicit_triggers = {
            r"\[SEARCH\]": "Perform a search to verify this information.",
            r"\[CODE\]": "Write and verify code for this problem.",
            r"\[CHECK\]": "Double check the logic and facts.",
            r"write.*script": "Write a script to solve this.",
            r"check.*facts": "Verify these facts."
        }

    def check_for_triggers(self, text_stream):
        """
        Scans for delegation triggers.
        Returns: task_description (str) or None
        """
        # 1. Check Explicit Delegation [TASK: ...]
        for pattern in self.delegation_patterns:
            match = re.search(pattern, text_stream, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # 2. Check Implicit Keywords
        for pattern, task_desc in self.implicit_triggers.items():
            if re.search(pattern, text_stream, re.IGNORECASE):
                return task_desc
                
        return None

