"""
LLM client for generating and judging samples.
Supports OpenAI, Anthropic, and OpenRouter APIs.
"""

import os
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root (parent directory)
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    # Also try loading from current directory
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly


class LLMClient(ABC):
    """Base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_json(self, prompt: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate structured JSON output."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = model
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate text using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def generate_json(self, prompt: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate structured JSON."""
        import json
        
        # Add schema enforcement to prompt
        if schema:
            prompt += f"\n\nYou MUST return valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if schema else None,
            temperature=0.3  # Lower temperature for structured output
        )
        
        content = response.choices[0].message.content
        return json.loads(content)


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, model: str = "claude-3-opus-20240229", api_key: Optional[str] = None):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = model
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate text using Anthropic."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def generate_json(self, prompt: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate structured JSON."""
        import json
        
        if schema:
            prompt += f"\n\nReturn valid JSON matching:\n{json.dumps(schema, indent=2)}"
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)


class GeminiClient(LLMClient):
    """
    Google Gemini API client using google-genai package.
    
    Uses the official google-genai package (replacement for deprecated google-generativeai).
    Supports all available Gemini models including newer ones.
    """
    
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        try:
            import google.genai as genai
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")
        
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.api_key = api_key
        self.model_name = model
        
        # Initialize the client
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {e}")
        
        # Verify the model exists and find the best available model
        try:
            available_models = list(self.client.models.list())
            model_names = [m.name for m in available_models if hasattr(m, 'name')]
            
            # Check if requested model exists
            if model in model_names or f"models/{model}" in model_names:
                self.model_name = model if model.startswith("models/") else f"models/{model}"
            else:
                # Try to find a similar model (e.g., gemini-2.5-flash instead of gemini-3-flash)
                preferred_models = [
                    "models/gemini-2.5-flash",  # Latest flash model
                    "models/gemini-2.5-pro",    # Latest pro model
                    "models/gemini-2.0-flash",  # Previous flash
                    "models/gemini-1.5-flash",  # Older flash
                ]
                
                for preferred in preferred_models:
                    if preferred in model_names:
                        self.model_name = preferred
                        print(f"Note: Using {preferred} instead of {model}")
                        break
                
                if not self.model_name:
                    raise RuntimeError(
                        f"Model '{model}' not found. Available models: {', '.join(model_names[:15])}"
                    )
        except Exception as e:
            raise RuntimeError(
                f"Failed to access models. Error: {e}. "
                "Please check your API key and ensure the Gemini API is enabled."
            ) from e
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate text using Gemini API."""
        try:
            # google-genai API format
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if len(parts) > 0 and hasattr(parts[0], 'text'):
                        return parts[0].text
            raise ValueError(f"Unexpected response format: {response}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate content: {e}") from e
    
    def generate_json(self, prompt: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate structured JSON using Gemini API."""
        import json
        
        # Add schema instruction to prompt
        if schema:
            prompt += f"\n\nYou MUST return ONLY valid JSON matching this schema:\n{json.dumps(schema, indent=2)}\n\nReturn ONLY the JSON object, no markdown, no explanation."
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={
                    "temperature": 0.3,  # Lower temperature for structured output
                    "max_output_tokens": 2000,
                    "response_mime_type": "application/json",  # Request JSON response
                }
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                content = response.text.strip()
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if len(parts) > 0 and hasattr(parts[0], 'text'):
                        content = parts[0].text.strip()
                    else:
                        raise ValueError(f"Unexpected response format: {response}")
                else:
                    raise ValueError(f"Unexpected response format: {response}")
            else:
                raise ValueError(f"Unexpected response format: {response}")
            
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Try to parse as single JSON object first
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try to extract the first valid JSON object
                # (in case the response contains multiple JSON objects)
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            return json.loads(line)
                        except json.JSONDecodeError:
                            continue
                # If still no valid JSON, try parsing the first line
                if lines:
                    try:
                        return json.loads(lines[0].strip())
                    except json.JSONDecodeError:
                        pass
                raise
            
        except json.JSONDecodeError as e:
            content_preview = content[:200] if 'content' in locals() else "N/A"
            raise RuntimeError(f"Failed to parse JSON response: {str(e)}. Response: {content_preview}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate JSON content: {e}") from e


def get_llm_client(provider: str = "openai", model: Optional[str] = None) -> LLMClient:
    """Factory function to get LLM client."""
    provider_lower = provider.lower()
    
    if provider_lower == "openai":
        return OpenAIClient(model=model or "gpt-4")
    elif provider_lower == "anthropic":
        return AnthropicClient(model=model or "claude-3-opus-20240229")
    elif provider_lower == "gemini" or provider_lower == "google":
        # Use gemini-2.5-flash as default (latest available)
        # Will auto-detect working model if specified model doesn't work
        return GeminiClient(model=model or "gemini-2.5-flash")
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, anthropic, gemini")

