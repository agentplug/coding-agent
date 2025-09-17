#!/usr/bin/env python3
"""
Agent Hub Agent: coding-agent
Generates Python code based on natural language prompts.
"""

import json
import sys
import os
from typing import Dict, Any
import os

class CodingAgent:
    """Python code generation agent."""
    
    def __init__(self):
        """Initialize the coding agent."""
        self.config = self._load_config()

    def _load_model_name(self) -> str:
        if os.getenv("OPENAI_API_KEY"):
            return "openai:gpt-4.1"
        elif os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic:claude-3.5-sonnet"
        elif os.getenv("GOOGLE_API_KEY"):
            return "google:gemini-2.0-flash"
        elif os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek:deepseek-chat"
        elif os.getenv("FIREWORKS_API_KEY"):
            return "fireworks:accounts/fireworks/models/llama-v3p2-3b-instruct"
        else:
            return "openai:gpt-4.1"
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback to default configuration if config.json doesn't exist
            return {
                "ai": {
                    "model": self._load_model_name(),
                    "temperature": 0.0,
                    "max_tokens": None,
                    "timeout": 30
                },
                "system_prompts": {
                    "generate_code": "You are a Python code generator. Generate only valid, working Python code. Do not include explanations, just the code.",
                    "explain_code": "You are a Python code explainer. Explain what the code does in simple terms.",
                    "validate_code": "You are a code validator. Check the provided code against the given criteria. Your answer must have a concise analysis, then based on that to answer PASS or FAIL."
                },
                "error_messages": {
                    "generate_code": "# Error generating code: {error}\n# Please check your API key and internet connection.",
                    "explain_code": "Error explaining code: {error}. Please check your API key and internet connection.",
                    "validate_code": "Error validating code: {error}. Please check your API key and internet connection."
                }
            }
    
    def generate_code(self, prompt: str) -> str:
        """
        Generate Python code based on a prompt using AI.
        
        Args:
            prompt: Natural language description of code to generate
            
        Returns:
            Generated Python code as a string
        """
        try:
            import aisuite as ai
            from dotenv import load_dotenv
            load_dotenv()
            
            client = ai.Client()
            messages = [
                {"role": "system", "content": self.config["system_prompts"]["generate_code"]},
                {"role": "user", "content": prompt}
            ]
            
            # Prepare API call parameters
            api_params = {
                "model": self.config["ai"]["model"],
                "messages": messages,
                "temperature": self.config["ai"]["temperature"]
            }
            
            # Add optional parameters if they exist
            if self.config["ai"]["max_tokens"]:
                api_params["max_tokens"] = self.config["ai"]["max_tokens"]
            
            response = client.chat.completions.create(**api_params)
            
            return response.choices[0].message.content
        except Exception as e:
            return self.config["error_messages"]["generate_code"].format(error=str(e))
    
    def explain_code(self, code: str) -> str:
        """
        Explain what a piece of Python code does using AI.
        
        Args:
            code: Python code to explain
            
        Returns:
            Explanation of what the code does
        """
        try:
            import aisuite as ai
            from dotenv import load_dotenv
            load_dotenv()
            
            client = ai.Client()
            messages = [
                {"role": "system", "content": self.config["system_prompts"]["explain_code"]},
                {"role": "user", "content": f"Explain this Python code:\n{code}"}
            ]
            
            # Prepare API call parameters
            api_params = {
                "model": self.config["ai"]["model"],
                "messages": messages,
                "temperature": self.config["ai"]["temperature"]
            }
            
            # Add optional parameters if they exist
            if self.config["ai"]["max_tokens"]:
                api_params["max_tokens"] = self.config["ai"]["max_tokens"]
            
            response = client.chat.completions.create(**api_params)
            
            return response.choices[0].message.content
        except Exception as e:
            return self.config["error_messages"]["explain_code"].format(error=str(e))
    
    def validate_code(self, code: str, criteria: str) -> str:
        """
        Validate code against specified criteria using AI.
        
        Args:
            code: Code to validate
            criteria: Validation criteria or requirements
            
        Returns:
            Validation result including pass/fail status and feedback
        """
        try:
            import aisuite as ai
            from dotenv import load_dotenv
            load_dotenv()
            
            client = ai.Client()
            messages = [
                {"role": "system", "content": self.config["system_prompts"]["validate_code"]},
                {"role": "user", "content": f"Validate this code against the following criteria:\n\nCriteria: {criteria}\n\nCode to validate:\n{code}"}
            ]
            
            # Prepare API call parameters
            api_params = {
                "model": self.config["ai"]["model"],
                "messages": messages,
                "temperature": self.config["ai"]["temperature"]
            }
            
            # Add optional parameters if they exist
            if self.config["ai"]["max_tokens"]:
                api_params["max_tokens"] = self.config["ai"]["max_tokens"]
            
            response = client.chat.completions.create(**api_params)
            
            return response.choices[0].message.content
        except Exception as e:
            return self.config["error_messages"]["validate_code"].format(error=str(e))

def main():
    """Main entry point for agent execution."""
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Invalid arguments"}))
        sys.exit(1)
    
    try:
        # Parse input from command line
        input_data = json.loads(sys.argv[1])
        method = input_data.get("method")
        parameters = input_data.get("parameters", {})
        
        # Create agent instance
        agent = CodingAgent()
        
        # Execute requested method
        if method == "generate_code":
            result = agent.generate_code(parameters.get("prompt", ""))
            print(json.dumps({"result": result}))
        elif method == "explain_code":
            result = agent.explain_code(parameters.get("code", ""))
            print(json.dumps({"result": result}))
        elif method == "validate_code":
            result = agent.validate_code(parameters.get("code", ""), parameters.get("criteria", ""))
            print(json.dumps({"result": result}))
        else:
            print(json.dumps({"error": f"Unknown method: {method}"}))
            sys.exit(1)
            
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
