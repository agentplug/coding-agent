#!/usr/bin/env python3
"""
Agent Hub Agent: coding-agent
Generates Python code based on natural language prompts.
"""

import json
import sys
from typing import Dict, Any

class CodingAgent:
    """Python code generation agent."""
    
    def __init__(self):
        """Initialize the coding agent."""
        pass
    
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
                {"role": "system", "content": "You are a Python code generator. Generate only valid, working Python code. Do not include explanations, just the code."},
                {"role": "user", "content": prompt}
            ]
            
            response = client.chat.completions.create(
                model="openai:gpt-4.1",
                messages=messages,
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"# Error generating code: {str(e)}\n# Please check your API key and internet connection."
    
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
                {"role": "system", "content": "You are a Python code explainer. Explain what the code does in simple terms."},
                {"role": "user", "content": f"Explain this Python code:\n{code}"}
            ]
            
            response = client.chat.completions.create(
                model="openai:gpt-4.1",
                messages=messages,
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error explaining code: {str(e)}. Please check your API key and internet connection."
    
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
                {"role": "system", "content": "You are a code validator. Check the provided code against the given criteria. Your answer must have a concise analysis, then based on that to answer PASS or FAIL."},
                {"role": "user", "content": f"Validate this code against the following criteria:\n\nCriteria: {criteria}\n\nCode to validate:\n{code}"}
            ]
            
            response = client.chat.completions.create(
                model="openai:gpt-4.1",
                messages=messages,
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error validating code: {str(e)}. Please check your API key and internet connection."

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
