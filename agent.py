#!/usr/bin/env python3
"""
Agent Hub Agent: coding-agent
Generates Python code based on natural language prompts.
"""

import json
import sys
import os
import logging
from typing import Dict, Any

# Import our modular LLM service
from llm_service import CoreLLMService, get_shared_llm_service

logger = logging.getLogger(__name__)


class CodingAgent:
    """Python code generation agent."""
    
    def __init__(self):
        """Initialize the coding agent."""
        self.config = self._load_config()
        self.llm_service = get_shared_llm_service()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config
        except FileNotFoundError:
            # Fallback to default configuration if config.json doesn't exist
            return {
                "ai": {
                    "temperature": 0.1,
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
            system_prompt = self.config["system_prompts"]["generate_code"]
            
            # Prepare API call parameters
            api_params = {
                "temperature": self.config["ai"]["temperature"]
            }
            
            # Add optional parameters if they exist
            if self.config["ai"]["max_tokens"]:
                api_params["max_tokens"] = self.config["ai"]["max_tokens"]
            
            response = self.llm_service.generate(
                prompt, 
                system_prompt=system_prompt,
                **api_params
            )
            
            return response
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
            system_prompt = self.config["system_prompts"]["explain_code"]
            user_prompt = f"Explain this Python code:\n{code}"
            
            # Prepare API call parameters
            api_params = {
                "temperature": self.config["ai"]["temperature"]
            }
            
            # Add optional parameters if they exist
            if self.config["ai"]["max_tokens"]:
                api_params["max_tokens"] = self.config["ai"]["max_tokens"]
            
            response = self.llm_service.generate(
                user_prompt, 
                system_prompt=system_prompt,
                **api_params
            )
            
            return response
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
            system_prompt = self.config["system_prompts"]["validate_code"]
            user_prompt = f"Validate this code against the following criteria:\n\nCriteria: {criteria}\n\nCode to validate:\n{code}"
            
            # Prepare API call parameters
            api_params = {
                "temperature": self.config["ai"]["temperature"]
            }
            
            # Add optional parameters if they exist
            if self.config["ai"]["max_tokens"]:
                api_params["max_tokens"] = self.config["ai"]["max_tokens"]
            
            response = self.llm_service.generate(
                user_prompt, 
                system_prompt=system_prompt,
                **api_params
            )
            
            return response
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