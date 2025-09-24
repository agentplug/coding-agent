#!/usr/bin/env python3
"""
Modular LLM Service Implementation
Self-contained LLM service without external agenthub dependencies
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a detected model"""
    name: str
    provider: str
    score: int
    is_local: bool
    is_available: bool


class ModelConfig:
    """Configuration constants for model selection and scoring"""

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

    # Common LM Studio URLs for auto-detection
    LMSTUDIO_URLS = [
        "http://localhost:1234",  # LM Studio default port
        "http://localhost:8080",  # Alternative port
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


class OllamaDetector:
    """Handles Ollama local model detection and management"""
    
    def __init__(self):
        self._ollama_url_cache: Optional[str] = None
    
    def detect_ollama_url(self) -> str:
        """Auto-detect Ollama API URL with fallback options (cached)"""
        # Return cached URL if available
        if self._ollama_url_cache is not None:
            return self._ollama_url_cache

        # 1. Environment variable (user override)
        if os.getenv("OLLAMA_API_URL"):
            url = os.getenv("OLLAMA_API_URL")
            print(f"🔧 Using Ollama URL from environment: {url}", file=sys.stderr)
            self._ollama_url_cache = url
            return url

        # 2. Try to find running Ollama instance
        for url in ModelConfig.OLLAMA_URLS:
            if self.check_ollama_available(url):
                print(f"🔍 Auto-detected Ollama URL: {url}", file=sys.stderr)
                self._ollama_url_cache = url
                return url

        # 3. Default fallback
        url = "http://localhost:11434"
        self._ollama_url_cache = url
        return url

    def check_ollama_available(self, url: str) -> bool:
        """Check if Ollama is running at the given URL"""
        try:
            # Try requests first (if available)
            import requests
            response = requests.get(f"{url}/api/tags", timeout=1)
            return response.status_code == 200
        except ImportError:
            # Fallback to urllib (standard library)
            try:
                import urllib.request
                import urllib.error
                req = urllib.request.Request(f"{url}/api/tags")
                response = urllib.request.urlopen(req, timeout=1)
                return response.status == 200
            except Exception:
                return False
        except Exception:
            return False

    def get_ollama_models(self, url: str) -> List[Dict]:
        """Get available models from Ollama"""
        try:
            # Try requests first (if available)
            import requests
            response = requests.get(f"{url}/api/tags", timeout=2)
            if response.status_code == 200:
                return response.json().get("models", [])
        except ImportError:
            # Fallback to urllib (standard library)
            try:
                import urllib.request
                import json
                req = urllib.request.Request(f"{url}/api/tags")
                response = urllib.request.urlopen(req, timeout=2)
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    return data.get("models", [])
            except Exception:
                pass
        except Exception:
            pass
        return []


class LMStudioDetector:
    """Handles LM Studio local model detection and management"""
    
    def __init__(self):
        self._lmstudio_url_cache: Optional[str] = None
    
    def detect_lmstudio_url(self) -> str:
        """Auto-detect LM Studio API URL with fallback options (cached)"""
        # Return cached URL if available
        if self._lmstudio_url_cache is not None:
            return self._lmstudio_url_cache

        # 1. Environment variable (user override)
        if os.getenv("LMSTUDIO_API_URL"):
            url = os.getenv("LMSTUDIO_API_URL")
            print(f"🔧 Using LM Studio URL from environment: {url}", file=sys.stderr)
            self._lmstudio_url_cache = url
            return url

        # 2. Try to find running LM Studio instance
        for url in ModelConfig.LMSTUDIO_URLS:
            if self.check_lmstudio_available(url):
                print(f"🔍 Auto-detected LM Studio URL: {url}", file=sys.stderr)
                self._lmstudio_url_cache = url
                return url

        # 3. Default fallback
        url = "http://localhost:1234"
        self._lmstudio_url_cache = url
        return url

    def check_lmstudio_available(self, url: str) -> bool:
        """Check if LM Studio is running at the given URL"""
        try:
            # Try requests first (if available)
            import requests
            # If URL already has /v1, use it directly, otherwise add it
            if url.endswith("/v1"):
                models_url = f"{url}/models"
            else:
                models_url = f"{url}/v1/models"
            
            response = requests.get(models_url, timeout=1)
            return response.status_code == 200
        except ImportError:
            # Fallback to urllib (standard library)
            try:
                import urllib.request
                import urllib.error
                # If URL already has /v1, use it directly, otherwise add it
                if url.endswith("/v1"):
                    models_url = f"{url}/models"
                else:
                    models_url = f"{url}/v1/models"
                
                req = urllib.request.Request(models_url)
                response = urllib.request.urlopen(req, timeout=1)
                return response.status == 200
            except Exception:
                return False
        except Exception:
            return False

    def get_lmstudio_models(self, url: str) -> List[str]:
        """Get available models from LM Studio"""
        try:
            # Try requests first (if available)
            import requests
            # If URL already has /v1, use it directly, otherwise add it
            if url.endswith("/v1"):
                models_url = f"{url}/models"
            else:
                models_url = f"{url}/v1/models"
            
            response = requests.get(models_url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]
        except ImportError:
            # Fallback to urllib (standard library)
            try:
                import urllib.request
                import json
                # If URL already has /v1, use it directly, otherwise add it
                if url.endswith("/v1"):
                    models_url = f"{url}/models"
                else:
                    models_url = f"{url}/v1/models"
                
                req = urllib.request.Request(models_url)
                response = urllib.request.urlopen(req, timeout=2)
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    return [model["id"] for model in data.get("data", [])]
            except Exception:
                pass
        except Exception:
            pass
        return []


class ModelScorer:
    """Handles model scoring and selection logic"""
    
    @staticmethod
    def calculate_model_score(model_name: str) -> int:
        """Calculate a score for a model (higher is better)"""
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

    @staticmethod
    def select_best_ollama_model(available_models: List[Dict]) -> str:
        """Select the best model from available Ollama models"""
        model_names = [model.get("name", "") for model in available_models]

        if not model_names:
            return "llama3:latest"

        # If only one model, return it
        if len(model_names) == 1:
            model_name = model_names[0]
            print(f"🎯 Single model available: {model_name}", file=sys.stderr)
            return model_name

        # Score each model and select the best one
        print(f"🔍 Evaluating {len(model_names)} models: {', '.join(model_names)}", file=sys.stderr)

        scored_models = []
        for model_name in model_names:
            score = ModelScorer.calculate_model_score(model_name)
            scored_models.append((model_name, score))

        # Sort by score (highest first) and return the best
        scored_models.sort(key=lambda x: x[1], reverse=True)
        best_model = scored_models[0][0]
        print(f"🏆 Best model selected: {best_model}", file=sys.stderr)
        return best_model

    def select_best_model(self, available_models: List[str], is_lmstudio: bool = False) -> str:
        """Select the best model from available models (unified for Ollama and LM Studio)"""
        if not available_models:
            fallback = "llama3:latest" if not is_lmstudio else "meta-llama/llama-3.2-1b-instruct"
            print(f"No models available, using fallback: {fallback}", file=sys.stderr)
            return fallback

        # If only one model, return it
        if len(available_models) == 1:
            model_name = available_models[0]
            print(f"🎯 Single model available: {model_name}", file=sys.stderr)
            return model_name

        # Score each model and select the best one
        print(f"🔍 Evaluating {len(available_models)} models: {', '.join(available_models)}", file=sys.stderr)

        # Log scoring details
        for model_name in available_models:
            score = self.calculate_model_score(model_name)
            print(f"📊 {model_name}: {score} points", file=sys.stderr)

        best_model = self._score_and_select_best(available_models)
        print(f"🏆 Best model selected: {best_model}", file=sys.stderr)
        return best_model

    def _score_and_select_best(self, model_names: List[str]) -> str:
        """Score models and return the best one"""
        scored_models = []
        for model_name in model_names:
            score = ModelScorer.calculate_model_score(model_name)
            scored_models.append((model_name, score))

        # Sort by score (highest first) and return the best
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models[0][0]


class ModelDetector:
    """Handles automatic model detection and selection"""
    
    def __init__(self):
        self.ollama_detector = OllamaDetector()
        self.lmstudio_detector = LMStudioDetector()
        self.model_scorer = ModelScorer()
    
    def detect_cloud_model(self) -> Optional[str]:
        """Detect available cloud model based on API keys"""
        # Check AWS Bedrock (special case - requires multiple env vars)
        if all(os.getenv(var) for var in ModelConfig.AWS_REQUIRED_VARS):
            return ModelConfig.AWS_MODEL

        # Check other cloud providers
        for env_var, model in ModelConfig.CLOUD_MODELS.items():
            if os.getenv(env_var):
                return model

        return None

    def detect_running_local_model(self) -> Optional[str]:
        """Detect running local models with auto-detection (preferring Ollama)"""
        # Try Ollama first (preferred)
        ollama_model = self.detect_ollama_model()
        if ollama_model:
            return ollama_model
        
        # Fallback to LM Studio if Ollama not available
        lmstudio_model = self.detect_lmstudio_model()
        if lmstudio_model:
            return lmstudio_model
        
        print("No local models detected", file=sys.stderr)
        return None

    def detect_ollama_model(self) -> Optional[str]:
        """Detect Ollama models"""
        ollama_url = self.ollama_detector.detect_ollama_url()

        if self.ollama_detector.check_ollama_available(ollama_url):
            # Get available models
            models = self.ollama_detector.get_ollama_models(ollama_url)
            if models:
                best_model = self.model_scorer.select_best_ollama_model(models)
                selected_model = f"ollama:{best_model}"
                print(f"🤖 Ollama model detected: {selected_model} (from {len(models)} available models)", file=sys.stderr)
                return selected_model
        return None

    def detect_lmstudio_model(self) -> Optional[str]:
        """Detect LM Studio models as fallback"""
        lmstudio_urls = [
            "http://localhost:1234",  # LM Studio default port
            "http://localhost:8080",  # Alternative port
        ]
        
        for url in lmstudio_urls:
            if self.lmstudio_detector.check_lmstudio_available(url):
                models = self.lmstudio_detector.get_lmstudio_models(url)
                if models:
                    best_model = self.model_scorer.select_best_model(models, is_lmstudio=True)
                    selected_model = f"lmstudio:{best_model}"
                    print(f"🤖 LM Studio model detected: {selected_model} (from {len(models)} available models)", file=sys.stderr)
                    return selected_model
        return None

    def detect_best_model(self) -> str:
        """Automatically detect and return the best available model"""
        # Priority 1: Check for local models first (auto-detection)
        local_model = self.detect_running_local_model()
        if local_model:
            print(f"🎯 Selected model: {local_model}", file=sys.stderr)
            return local_model

        # Priority 2: Check API keys and return corresponding cloud model
        cloud_model = self.detect_cloud_model()
        if cloud_model:
            print(f"☁️ Selected cloud model: {cloud_model}", file=sys.stderr)
            return cloud_model

        # Default fallback
        default_model = "openai:gpt-4o"
        print(f"⚠️ No models detected, using default: {default_model}", file=sys.stderr)
        return default_model


class CoreLLMService:
    """
    Comprehensive LLM Service Implementation
    Self-contained without external agenthub dependencies
    """

    def __init__(
        self,
        aisuite_client: Any = None,
        model: str | None = None,
        auto_detect: bool = True,
    ) -> None:
        """Initialize Core LLM Service"""
        # Initialize components
        self.model_detector = ModelDetector()
        self.ollama_detector = OllamaDetector()
        
        # Initialize caching
        self.cache: Dict[str, Any] = {}
        self._model_info: Optional[ModelInfo] = None

        # Model selection
        if model:
            self.model = model
            print(f"🎯 Using specified model: {model}", file=sys.stderr)
        elif auto_detect:
            self.model = self.model_detector.detect_best_model()
        else:
            self.model = ModelConfig.CLOUD_MODELS.get("OPENAI_API_KEY", "openai:gpt-4o")
            print(f"⚠️ No model specified, using default: {self.model}", file=sys.stderr)

        # Initialize client with appropriate configuration
        if aisuite_client is None:
            self.client = self._initialize_aisuite_with_config(self.model)
        else:
            self.client = aisuite_client

    def _initialize_aisuite_with_config(self, model: str) -> Any:
        """Initialize AISuite client with appropriate configuration"""
        import aisuite as ai

        # Check if it's a local model
        if model.startswith("ollama:"):
            return self._initialize_ollama_client(model)
        elif model.startswith("lmstudio:"):
            return self._initialize_lmstudio_client(model)
        else:
            # Cloud models - no special config needed
            return ai.Client()

    def _initialize_ollama_client(self, model: str) -> Any:
        """Initialize AISuite client for Ollama"""
        import aisuite as ai

        # Get Ollama configuration
        api_url = self.ollama_detector.detect_ollama_url()
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))

        return ai.Client(
            provider_configs={
                "ollama": {
                    "api_url": api_url,
                    "timeout": timeout,
                }
            }
        )

    def _initialize_lmstudio_client(self, model: str) -> Any:
        """Initialize AISuite client for LM Studio using OpenAI-compatible API"""
        import aisuite as ai

        # Get LM Studio configuration
        api_url = self._detect_lmstudio_url()
        timeout = int(os.getenv("LMSTUDIO_TIMEOUT", "300"))

        # LM Studio provides OpenAI-compatible API, so use openai provider with custom base_url
        return ai.Client(
            provider_configs={
                "openai": {
                    "base_url": api_url,
                    "timeout": timeout,
                }
            }
        )

    def _detect_lmstudio_url(self) -> str:
        """Detect LM Studio API URL"""
        # Check environment variable first
        if os.getenv("LMSTUDIO_API_URL"):
            return os.getenv("LMSTUDIO_API_URL")
        
        # Check common LM Studio ports
        lmstudio_urls = [
            "http://localhost:1234/v1",  # LM Studio default with /v1
            "http://localhost:8080/v1",  # Alternative port with /v1
        ]
        
        for url in lmstudio_urls:
            if self._check_lmstudio_available(url):
                return url
        
        # Fallback to default
        return "http://localhost:1234/v1"

    def _check_lmstudio_available(self, url: str) -> bool:
        """Check if LM Studio is running at the given URL"""
        try:
            # Try requests first (if available)
            import requests
            # If URL already has /v1, use it directly, otherwise add it
            if url.endswith("/v1"):
                models_url = f"{url}/models"
            else:
                models_url = f"{url}/v1/models"
            
            response = requests.get(models_url, timeout=1)
            return response.status_code == 200
        except ImportError:
            # Fallback to urllib (standard library)
            try:
                import urllib.request
                import urllib.error
                # If URL already has /v1, use it directly, otherwise add it
                if url.endswith("/v1"):
                    models_url = f"{url}/models"
                else:
                    models_url = f"{url}/v1/models"
                
                req = urllib.request.Request(models_url)
                response = urllib.request.urlopen(req, timeout=1)
                return response.status == 200
            except Exception:
                return False
        except Exception:
            return False

    def generate(
        self,
        input_data: str | List[Dict],
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
                    model=self._get_actual_model_name(), messages=messages, **request_kwargs
                )
                return str(response.choices[0].message.content)

            elif isinstance(input_data, list):
                # Messages - organize into context and focus on current
                messages = self._organize_messages_to_aisuite_format(
                    input_data, system_prompt
                )

                response = self.client.chat.completions.create(
                    model=self._get_actual_model_name(), messages=messages, **request_kwargs
                )
                return str(response.choices[0].message.content)
            else:
                raise ValueError("input_data must be string or list")
        except Exception as e:
            print(f"AISuite generation failed: {e}", file=sys.stderr)
            return self._fallback_response()

    def _organize_messages_to_aisuite_format(
        self, messages: List[Dict], system_prompt: str | None = None
    ) -> List[Dict]:
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
        """Get the currently selected model"""
        return self.model

    def is_local_model(self) -> bool:
        """Check if current model is local (Ollama or LM Studio)"""
        return self.model.startswith("ollama:") or self.model.startswith("lmstudio:")

    def _get_actual_model_name(self) -> str:
        """Get the actual model name for API calls (removes platform prefix)"""
        if self.model.startswith("ollama:"):
            # For Ollama, keep the full format for AISuite
            return self.model  # Keep "ollama:model" format
        elif self.model.startswith("lmstudio:"):
            # For LM Studio, use openai provider format
            model_name = self.model[9:]  # Remove "lmstudio:" prefix
            return f"openai:{model_name}"
        else:
            return self.model

    def get_model_info(self) -> ModelInfo:
        """Get detailed information about the current model"""
        if self._model_info is None:
            self._model_info = self._create_model_info()
        return self._model_info

    def _create_model_info(self) -> ModelInfo:
        """Create ModelInfo object for current model"""
        provider, model_name = (
            self.model.split(":", 1) if ":" in self.model else ("unknown", self.model)
        )
        is_local = provider in ["ollama", "lmstudio"]
        score = (
            self.model_scorer.calculate_model_score(model_name) if is_local else 100
        )  # Cloud models get default score

        return ModelInfo(
            name=model_name,
            provider=provider,
            score=score,
            is_local=is_local,
            is_available=True,
        )


# Global shared instance to prevent duplicate model detection logs
_shared_llm_service: Optional[CoreLLMService] = None


def get_shared_llm_service(
    model: str | None = None, auto_detect: bool = True
) -> CoreLLMService:
    """
    Get a shared CoreLLMService instance to avoid duplicate model detection logs.
    """
    global _shared_llm_service

    # If no shared instance exists, create one
    if _shared_llm_service is None:
        _shared_llm_service = CoreLLMService(model=model, auto_detect=auto_detect)
        return _shared_llm_service

    # If shared instance exists but different model requested, create new instance
    if model and model != _shared_llm_service.model:
        return CoreLLMService(model=model, auto_detect=auto_detect)

    # Return existing shared instance
    return _shared_llm_service


def reset_shared_llm_service() -> None:
    """Reset the shared LLM service instance"""
    global _shared_llm_service
    _shared_llm_service = None
