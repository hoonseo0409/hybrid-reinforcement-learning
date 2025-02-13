import utils, sensor
import secures
from copy import deepcopy
from sympy import sympify
# import win32gui
# from win32process import GetWindowThreadProcessId
import ctypes

import base64
import cv2
from anthropic import Anthropic
import numpy as np
from typing import Optional
import json
import os
import threading
from datetime import datetime, timedelta
import traceback

from typing import List, Dict, Tuple
from dataclasses import dataclass

def if_current_keyboard_is_english():
    """
        Gets the keyboard language in use by the current
        active window process.

        https://stackoverflow.com/questions/42047253/how-to-detect-current-keyboard-language-in-python
    """

    user32 = ctypes.WinDLL('user32', use_last_error=True)

    # Get the current active window handle
    curr_window = user32.GetForegroundWindow()

    # Get the thread id from that window handle
    threadid = user32.GetWindowThreadProcessId(curr_window, 0)

    # Get the keyboard layout id from the threadid
    layout_id = user32.GetKeyboardLayout(threadid)

    # Extract the keyboard language id from the keyboard layout id
    language_id = layout_id & (2 ** 16 - 1)

    # Convert the keyboard language id from decimal to hexadecimal
    language_id_hex = hex(language_id)

    # Check if the hex value is in the dictionary.
    # if language_id_hex in languages.keys():
        # return languages[language_id_hex]
    if language_id_hex == '0x409':
        return True
    else:
        # Return language id hexadecimal value if not found.
        return False

# Track API usage across all agents
_api_call_times = []
_api_call_lock = threading.Lock()

def get_agent_source_code(agent_type: str) -> str:
    """Get the source code of the agent implementation file
    
    Args:
        agent_type: Type of agent
    """
    agent_path = os.path.join(os.path.dirname(__file__), f'agents/agent_{agent_type}.py')
    with open(agent_path, 'r', encoding='utf-8') as f:
        return f.read()

class ScreenEmbeddingModel:
    """Template class for screen embedding model that converts sequence of frames to fixed-length vector"""
    
    def __init__(self, embedding_dim: int = 512):
        """Initialize model
        
        Args:
            embedding_dim: Dimension of output embedding vector
        """
        self.embedding_dim = embedding_dim
        # Model weights and architecture will be implemented later
        
    def predict(self, screen_frames: List[np.ndarray]) -> np.ndarray:
        """Convert sequence of frames into fixed dimension embedding
        
        Args:
            screen_frames: List of screen frames as numpy arrays
            
        Returns:
            embedding: Fixed length embedding vector as numpy array
        """
        # Actual implementation will come later
        return np.zeros(self.embedding_dim)
    
@dataclass
class ScenarioData:
    """Container for scenario data including screens and explanation"""
    screens: List[np.ndarray]
    explanation: str
    embedding: np.ndarray

class RAGRetriever:
    """Retrieves relevant scenarios using embedding similarity"""
    
    def __init__(self, 
                 embedding_model: ScreenEmbeddingModel,
                 scenarios: List[ScenarioData]):
        """Initialize retriever
        
        Args:
            embedding_model: Model to convert screens to embeddings
            scenarios: List of available scenarios with screens and explanations
        """
        self.embedding_model = embedding_model
        self.scenarios = scenarios
        
        # Stack all scenario embeddings into 2D array
        self.embedding_matrix = np.stack([s.embedding for s in scenarios])
        
    def retrieve(self, 
                query_screens: List[np.ndarray],
                top_k: int = 1,
                similarity_threshold: float = 0.7
                ) -> List[ScenarioData]:
        """Retrieve most relevant scenarios
        
        Args:
            query_screens: List of query screen frames
            top_k: Number of scenarios to retrieve
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            relevant_scenarios: List of most relevant scenarios
        """
        # Get embedding for query screens
        query_embedding = self.embedding_model.predict(query_screens)
        
        # Calculate similarity scores with all scenarios
        # Using dot product similarity
        similarity_scores = np.dot(self.embedding_matrix, query_embedding)
        
        # Get indices of top K most similar scenarios above threshold
        top_indices = np.argsort(similarity_scores)[-top_k:]
        relevant_indices = [i for i in top_indices 
                          if similarity_scores[i] >= similarity_threshold]
        
        # Return relevant scenarios
        return [self.scenarios[i] for i in relevant_indices]

def handle_with_LLM(screenshots: List[np.ndarray],
                    traceback: str, 
                    tr: int,
                    agent_type: str,
                    agent_pos: tuple,
                    agent_size: tuple,  
                    eng_name: str,
                    rag_retriever: RAGRetriever,
                    previous_code: Optional[str] = None,
                    previous_result: Optional[dict] = None) -> Optional[str]:
    """
    Generate error handling code using RAG-enhanced Claude API
    
    Args:
        screenshots: List of recent game screenshots
        traceback: Error traceback string
        tr: Current trial number 
        agent_type: Type of agent
        agent_pos: Position of agent window
        agent_size: Size of agent window
        eng_name: English name of agent
        rag_retriever: RAG retrieval system
        previous_code: Previously generated code that failed
        previous_result: Result from executing previous code
        
    Returns:
        Python code string to handle error state
    """
    # Check trial number
    if tr >= 5:
        return None
    utils.telegram_bot.send_message(secures.telegram_chat_id, 
                                  f"{eng_name}: Trying to call Anthropic API...")
        
    # Check API rate limit
    current_time = datetime.now()
    with _api_call_lock:
        cutoff_time = current_time - timedelta(minutes=30)
        _api_call_times[:] = [t for t in _api_call_times if t > cutoff_time]
        
        if len(_api_call_times) >= 15:
            utils.telegram_bot.send_message(secures.telegram_chat_id,
                f"{eng_name}: Anthropic API rate limit reached (15 calls/30min)")
            return None
            
        _api_call_times.append(current_time)

    # Retrieve relevant scenarios using RAG
    relevant_scenarios = rag_retriever.retrieve(screenshots)
    
    # Convert current screenshots to base64
    # Using most recent screenshot for LLM
    success, buffer = cv2.imencode('.png', screenshots[-1])
    if not success:
        raise ValueError("Failed to encode screenshot")
    screenshot_b64 = base64.b64encode(buffer).decode('utf-8')

    # Get truncated agent source code
    agent_source = get_agent_source_code(agent_type)
    agent_source_lines = agent_source.split('\n')[:1300]
    agent_source_truncated = '\n'.join(agent_source_lines)
    
    # System message with source code and tools
    system_message = f"""You are an expert at analyzing game screenshots and debugging automation systems.
Below is the first part of the source code of the Agent class you're working with:

{agent_source_truncated}

You have access to the following core functions:
"""

    # Add tools definition
    tools_definition = """
<tools>
[
    {
        "name": "click",
        "description": "Click the mouse at specified coordinates relative to game window's upper left corner",
        "parameters": {
            "type": "object", 
            "properties": {
                "x": {"type": "integer", "description": "X coordinate relative to game window"},
                "y": {"type": "integer", "description": "Y coordinate relative to game window"}
            },
            "required": ["x", "y"]
        }
    },
    {
        "name": "press_key",
        "description": "Press a specified keyboard key",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key to press (e.g. 'm', 'esc', 'space')"}
            },
            "required": ["key"]
        }
    }
]
</tools>

When generating code to handle uncertain game states:
1. Use the provided click and press_key functions for core interactions
2. Follow existing error handling patterns from the source code
3. Focus on safety and robustness
4. Return results in the same format as other Agent methods"""

    system_message = system_message + tools_definition

    # Build user prompt with RAG context
    prompt = f"""Help debug and handle an uncertain state in the game automation system.

Current screenshot has been provided (base64 encoded PNG).

Error traceback:
{traceback}

The game window position is at pos = ({agent_pos[0]}, {agent_pos[1]}) and size is size = ({agent_size[0]}, {agent_size[1]}).
"""

    # Add RAG context if relevant scenarios found
    if relevant_scenarios:
        prompt += "\nRelevant previous scenarios:\n"
        for i, scenario in enumerate(relevant_scenarios):
            prompt += f"\nScenario {i+1}:\n{scenario.explanation}\n"

    prompt += """
Please analyze the screenshot and error and generate Python code to handle this uncertain state. Your code should:
1. Use appropriate Agent methods to return the game to a stable state, such as ingame state.
2. Include error handling and logging.
3. Return a dict with 'success' key indicating if the handling was successful.
4. Do not purchase items, destroy items, or use the Agent's inventory. These characters are expensive, do not incur any financial risk.

You have access to all Agent methods shown in the source code, plus utils, sensor, and cv2 modules."""

    # Add context about previous attempts if applicable
    if previous_code is not None:
        prompt += f"\nPreviously tried code:\n{previous_code}\n\nResult:\n{json.dumps(previous_result, indent=2)}"

    # Call Claude API
    try:
        client = Anthropic(api_key=secures.anthropic_api_key)
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.7,
            system=system_message,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt + "\n\nPlease make use of the click() and press_key() functions defined in the tools section above to interact with the game."
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64
                        }
                    }
                ]
            }])

        # Extract code from response
        response = message.content[0].text
        utils.telegram_bot.send_message(secures.telegram_chat_id, 
                                      f"{eng_name}: LLM message received: {response}")
        
        code_start = response.find("```python")
        code_end = response.find("```", code_start + 8)
        
        if code_start == -1 or code_end == -1:
            raise ValueError("No Python code found in Claude response")
            
        code = response[code_start + 8:code_end].strip()
        
        # Security checks
        forbidden_terms = [
            "import os", "import sys", "subprocess", "exec(", "eval(",
            "open(", "__import__", "system(", "popen("
        ]
        
        for term in forbidden_terms:
            if term in code.lower():
                raise ValueError(f"Generated code contains forbidden term: {term}")

        return code

    except Exception as e:
        utils.telegram_bot.send_message(secures.telegram_chat_id,
            f"{eng_name}: LLM API call failed: {str(e)}")
        return None

def handle_error_state(agent, code: str) -> dict:
    """
    Execute generated error handling code safely.
    
    Args:
        agent: Agent instance
        code: Python code string to execute
    
    Returns:
        Dict containing execution results and any errors
    """
    try:
        # Create function from code
        namespace = {
            'self': agent,
            'utils': utils,
            'cv2': cv2,
            'sensor': sensor,
            'np': np
        }
        
        # Indent code for function body
        indented_code = '\n    '.join(code.split('\n'))
        full_code = f"def error_handler(self):\n    {indented_code}"
        
        exec(full_code, namespace)
        handler = namespace["error_handler"]
        
        # Execute handler with timeout
        result = handler(agent)
        
        if isinstance(result, dict) and 'success' in result:
            return result
        else:
            return {
                'success': False,
                'error': 'Handler did not return expected result format'
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
