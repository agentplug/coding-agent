#!/usr/bin/env python3
"""
Agent Hub Agent: coding-agent
Generates Python code based on natural language prompts.
"""

import json
import sys
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# CORE LLM SERVICE IMPLEMENTATION (Mimicking agenthub.core.llm)
# =============================================================================

class ModelConfig:
    """Configuration constants for model selection and scoring."""

    # Preferred models for different use cases
    PREFERRED_MODELS = [
        "gpt-oss:120b",
        "gpt-oss:20b",  # OpenAI open-weight (highest priority)
        "deepseek-r1:70b",
        "deepseek-r1:32b",  # DeepSeek reasoning models
        "gemma:latest",
        "llama3:latest",  # General purpose models
        "qwen:latest",
        "mistral:latest",  # Alternative models
    ]

    # Model family scoring (higher = better for agentic tasks)
    FAMILY_SCORES = {
        "gpt-oss": 60,  # OpenAI's open-weight models (highest priority)
        "deepseek": 50,
        "gemma": 35,
        "llama": 40,
        "qwen": 50,
        "mistral": 35,
        "codellama": 30,
        "phind": 25,
        "wizard": 20,
        "vicuna": 15,
        "claude": 45,
        "gpt": 40,
    }

    # Size scoring (larger is generally better)
    SIZE_SCORES = {
        "120b": 120,
        "70b": 100,
        "65b": 95,
        "32b": 80,
        "latest": 80,
        "20b": 75,
        "13b": 60,
        "7b": 40,
        "3b": 20,
        "1b": 10,
    }

    # Common Ollama URLs for auto-detection
    OLLAMA_URLS = [
        "http://localhost:11434",  # Default Ollama
        "http://127.0.0.1:11434",  # Alternative localhost
        "http://0.0.0.0:11434",  # All interfaces
    ]

    # Cloud provider models (fallback when no local models)
    CLOUD_MODELS = {
        "OPENAI_API_KEY": "openai:gpt-4o",
        "ANTHROPIC_API_KEY": "anthropic:claude-3-5-sonnet-20241022",
        "GOOGLE_API_KEY": "google:gemini-1.5-pro",
        "DEEPSEEK_API_KEY": "deepseek:deepseek-chat",
        "FIREWORKS_API_KEY": "fireworks:accounts/fireworks/models/llama-v3p2-3b-instruct",
        "COHERE_API_KEY": "cohere:command-r-plus",
        "MISTRAL_API_KEY": "mistral:mistral-large-latest",
        "GROQ_API_KEY": "groq:llama-3.1-70b-versatile",
        "REPLICATE_API_TOKEN": "replicate:meta/llama-2-70b-chat",
        "HUGGINGFACE_API_KEY": "huggingface:microsoft/DialoGPT-large",
        "AZURE_OPENAI_API_KEY": "azure:gpt-4o",
    }

    # Special case for AWS (requires multiple env vars)
    AWS_MODEL = "aws:anthropic.claude-3-5-sonnet-20241022-v2:0"
    AWS_REQUIRED_VARS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]


@dataclass
class ModelInfo:
    """Information about a detected model"""

    name: str
    provider: str
    score: int
    is_local: bool
    is_available: bool


class CoreLLMService:
    """
    Comprehensive LLM Service for AgentHub
    Mimicking the agenthub.core.llm implementation
    """

    def __init__(
        self,
        aisuite_client: Any = None,
        model: str | None = None,
        auto_detect: bool = True,
    ) -> None:
        """Initialize Core LLM Service"""
        # Initialize caching first (needed by model detection)
        self.cache: Dict[str, Any] = {}
        self._model_info: Optional[ModelInfo] = None
        self._ollama_url_cache: Optional[str] = None

        # Model selection
        if model:
            self.model = model
            print(f"🎯 Using specified model: {model}")
        elif auto_detect:
            self.model = self._detect_best_model()
        else:
            self.model = ModelConfig.CLOUD_MODELS.get("OPENAI_API_KEY", "openai:gpt-4o")
            print(f"⚠️ No model specified, using default: {self.model}")

        # Initialize client with appropriate configuration
        if aisuite_client is None:
            self.client = self._initialize_aisuite_with_config(self.model)
        else:
            self.client = aisuite_client

    def _detect_best_model(self) -> str:
        """Automatically detect and return the best available model."""
        # Priority 1: Check for local models first (auto-detection)
        local_model = self._detect_running_local_model()
        if local_model:
            print(f"🎯 Selected model: {local_model}")
            return local_model

        # Priority 2: Check API keys and return corresponding cloud model
        cloud_model = self._detect_cloud_model()
        if cloud_model:
            print(f"☁️ Selected cloud model: {cloud_model}")
            return cloud_model

        # Default fallback
        default_model = "openai:gpt-4o"
        print(f"⚠️ No models detected, using default: {default_model}")
        return default_model

    def _detect_cloud_model(self) -> Optional[str]:
        """Detect available cloud model based on API keys."""
        # Check AWS Bedrock (special case - requires multiple env vars)
        if all(os.getenv(var) for var in ModelConfig.AWS_REQUIRED_VARS):
            return ModelConfig.AWS_MODEL

        # Check other cloud providers
        for env_var, model in ModelConfig.CLOUD_MODELS.items():
            if os.getenv(env_var):
                return model

        return None

    def _detect_running_local_model(self) -> Optional[str]:
        """Detect running local models with auto-detection."""
        # Check if Ollama is available
        ollama_url = self._detect_ollama_url()

        if self._check_ollama_available(ollama_url):
            # Get available models
            models = self._get_ollama_models(ollama_url)
            if models:
                best_model = self._select_best_ollama_model(models)
                selected_model = f"ollama:{best_model}"
                print(f"🤖 Local model detected: {selected_model} (from {len(models)} available models)")
                return selected_model

        return None

    def _detect_ollama_url(self) -> str:
        """Auto-detect Ollama API URL with fallback options (cached)."""
        # Return cached URL if available
        if self._ollama_url_cache is not None:
            return self._ollama_url_cache

        # 1. Environment variable (user override)
        if os.getenv("OLLAMA_API_URL"):
            url = os.getenv("OLLAMA_API_URL")
            print(f"🔧 Using Ollama URL from environment: {url}")
            self._ollama_url_cache = url
            return url

        # 2. Try to find running Ollama instance
        for url in ModelConfig.OLLAMA_URLS:
            if self._check_ollama_available(url):
                print(f"🔍 Auto-detected Ollama URL: {url}")
                self._ollama_url_cache = url
                return url

        # 3. Default fallback
        url = "http://localhost:11434"
        self._ollama_url_cache = url
        return url

    def _check_ollama_available(self, url: str) -> bool:
        """Check if Ollama is running at the given URL."""
        try:
            import requests
            response = requests.get(f"{url}/api/tags", timeout=1)
            return response.status_code == 200
        except Exception:
            return False

    def _get_ollama_models(self, url: str) -> list[dict]:
        """Get available models from Ollama."""
        try:
            import requests
            response = requests.get(f"{url}/api/tags", timeout=2)
            if response.status_code == 200:
                return response.json().get("models", [])
        except Exception:
            pass
        return []

    def _select_best_ollama_model(self, available_models: list[dict]) -> str:
        """Select the best model from available Ollama models."""
        model_names = [model.get("name", "") for model in available_models]

        if not model_names:
            return "llama3:latest"

        # If only one model, return it
        if len(model_names) == 1:
            model_name = model_names[0]
            print(f"🎯 Single model available: {model_name}")
            return model_name

        # Score each model and select the best one
        print(f"🔍 Evaluating {len(model_names)} models: {', '.join(model_names)}")

        best_model = self._score_and_select_best(model_names)
        print(f"🏆 Best model selected: {best_model}")
        return best_model

    def _score_and_select_best(self, model_names: list[str]) -> str:
        """Score models and return the best one."""
        scored_models = []

        for model_name in model_names:
            score = self._calculate_model_score(model_name)
            scored_models.append((model_name, score))

        # Sort by score (highest first) and return the best
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models[0][0]

    def _calculate_model_score(self, model_name: str) -> int:
        """Calculate a score for a model (higher is better)."""
        score = 0
        model_lower = model_name.lower()

        # Size scoring (larger is better)
        for size, points in ModelConfig.SIZE_SCORES.items():
            if size in model_lower:
                score += points
                break

        # Model family scoring (known good models)
        for family, points in ModelConfig.FAMILY_SCORES.items():
            if family in model_lower:
                score += points
                break

        # Penalty for poor models
        poor_indicators = ["tiny", "small", "test", "demo"]
        for indicator in poor_indicators:
            if indicator in model_lower:
                score -= 30
                break

        # Bonus for latest/stable versions
        if "latest" in model_lower or "stable" in model_lower:
            score += 10

        return score

    def _initialize_aisuite_with_config(self, model: str) -> Any:
        """Initialize AISuite client with appropriate configuration."""
        import aisuite as ai

        # Check if it's a local model
        if model.startswith("ollama:"):
            return self._initialize_ollama_client(model)
        else:
            # Cloud models - no special config needed
            return ai.Client()

    def _initialize_ollama_client(self, model: str) -> Any:
        """Initialize AISuite client for Ollama."""
        import aisuite as ai

        # Get Ollama configuration
        api_url = self._detect_ollama_url()
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))

        return ai.Client(
            provider_configs={
                "ollama": {
                    "api_url": api_url,
                    "timeout": timeout,
                }
            }
        )

    def generate(
        self,
        input_data: str | list[dict],
        system_prompt: str | None = None,
        return_json: bool = False,
        **kwargs: Any,
    ) -> str:
        """Adaptive LLM generation using AISuite"""
        if not self.client:
            return self._fallback_response()

        try:
            # Prepare request parameters
            request_kwargs = kwargs.copy()
            if return_json:
                request_kwargs["response_format"] = {"type": "json_object"}

            if isinstance(input_data, str):
                # Single prompt - convert to messages format
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": input_data})

                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, **request_kwargs
                )
                return str(response.choices[0].message.content)

            elif isinstance(input_data, list):
                # Messages - organize into context and focus on current
                messages = self._organize_messages_to_aisuite_format(
                    input_data, system_prompt
                )

                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, **request_kwargs
                )
                return str(response.choices[0].message.content)
            else:
                raise ValueError("input_data must be string or list")
        except Exception as e:
            print(f"AISuite generation failed: {e}")
            return self._fallback_response()

    def _organize_messages_to_aisuite_format(
        self, messages: list[dict], system_prompt: str | None = None
    ) -> list[dict]:
        """Convert conversation messages to AISuite messages format with context management"""
        if not messages:
            return []

        # Separate context (previous messages) from current message
        context_messages = messages[:-1] if len(messages) > 1 else []
        current_message = messages[-1]

        # Build messages list for AISuite
        aisuite_messages = []

        # Add system prompt if provided
        if system_prompt:
            aisuite_messages.append({"role": "system", "content": system_prompt})

        # Add context messages (limit to last 3-4 to avoid overwhelming)
        if context_messages:
            recent_messages = (
                context_messages[-3:] if len(context_messages) > 3 else context_messages
            )
            for msg in recent_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Truncate long messages
                if len(content) > 200:
                    content = content[:200] + "..."
                aisuite_messages.append({"role": role, "content": content})

        # Add current message
        current_content = current_message.get("content", "")
        current_role = current_message.get("role", "user")
        aisuite_messages.append({"role": current_role, "content": current_content})

        return aisuite_messages

    def _fallback_response(self) -> str:
        """Fallback response when AISuite is not available"""
        return "AISuite not available"

    def get_current_model(self) -> str:
        """Get the currently selected model."""
        return self.model

    def is_local_model(self) -> bool:
        """Check if current model is local (Ollama)."""
        return self.model.startswith("ollama:")


# =============================================================================
# CODING AGENT IMPLEMENTATION
# =============================================================================

class CodingAgent:
    """Python code generation agent."""
    
    def __init__(self):
        """Initialize the coding agent."""
        self.config = self._load_config()
        self.llm_service = CoreLLMService()

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
