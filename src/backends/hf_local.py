from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
import gc
import os
import logging

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:  # pragma: no cover
    raise RuntimeError("transformers must be installed to use hf_local backend") from e

from ..config import Config
from ..utils.logger import get_logger, log_exception
from ..utils.memory_manager import MemoryManager, prepare_for_model_loading


class HFLocalClient:
    """
    Local Hugging Face backend that mirrors APIClient's async context and generate() contract.

    Load order (to honor the user's requested snippet while remaining robust on low-VRAM GPUs):
      1) Try user's exact preferred load:
           model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="cuda")
         This can OOM on small GPUs (e.g., 4 GB). If it fails, we fallback automatically:
      2) device_map="auto" (Accelerate dispatch / CPU+offload as available)
      3) CPU-only load as last resort.

    Returns a dict shaped like APIClient.generate():
      {
        "output_text": str,
        "raw": dict,
        "meta": { "cached": bool, "cache_key": str }
      }

    Notes:
      - Caching is not implemented here (disk cache lives in APIClient). meta.cached=False always.
      - Tools are ignored (no tool execution in local backend).
    """

    def __init__(self, config: Config, logger_name: str = "hf-local") -> None:
        self.config = config
        self.logger: logging.Logger = get_logger(logger_name)
        self._tokenizer = None
        self._model = None
        self._started = False

    async def __aenter__(self) -> "HFLocalClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def start(self) -> None:
        if self._started:
            return
        configured = (self.config.model or "openai/gpt-oss-20b").strip()
        model_id = configured if "/" in configured else f"openai/{configured}"

        # Optional HF auth token (for gated/private repos)
        import os
        token = getattr(self.config, "hf_token", None) or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

        # Prepare memory before loading
        model_size_gb = 20.0 if "20b" in model_id.lower() else 10.0
        can_load, memory_message = prepare_for_model_loading(model_size_gb)
        self.logger.info(f"[hf_local] Memory preparation: {memory_message}")

        # Load tokenizer
        self.logger.info(f"[hf_local] Loading tokenizer: {model_id}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        # Some tokenizers/models don't define pad token; align to eos if missing
        if getattr(self._tokenizer, "pad_token_id", None) is None and getattr(self._tokenizer, "eos_token_id", None) is not None:
            try:
                self._tokenizer.pad_token = self._tokenizer.eos_token  # type: ignore[attr-defined]
            except Exception:
                pass

        # Optional: 8-bit quantized load if configured
        if bool(getattr(self.config, "hf_load_in_8bit", False)):
            try:
                self.logger.info("[hf_local] Attempting 8-bit quantized load")
                
                # Clear memory before loading
                MemoryManager.clear_memory()
                
                # Check if bitsandbytes is available and properly configured
                try:
                    import bitsandbytes as bnb
                    self.logger.info("[hf_local] bitsandbytes library found")
                    
                    # Try creating BitsAndBytesConfig manually for better compatibility
                    from transformers import BitsAndBytesConfig
                    
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16 if torch is not None else None,
                        bnb_8bit_use_double_quant=True,
                        bnb_8bit_quant_type="nf4"
                    )
                    
                    kwargs: Dict[str, Any] = {
                        "quantization_config": bnb_config,
                        "device_map": "auto",
                        "low_cpu_mem_usage": True,
                        "token": token,
                    }
                    
                except ImportError:
                    self.logger.warning("[hf_local] bitsandbytes not available, trying direct 8-bit load")
                    # Fallback to direct parameters (may not work with newer transformers)
                    kwargs: Dict[str, Any] = {
                        "device_map": "auto",
                        "low_cpu_mem_usage": True,
                        "token": token,
                        "torch_dtype": torch.float16 if torch is not None else None,
                    }
                    
                    # Only add load_in_8bit if we're sure it won't cause issues
                    import transformers
                    if hasattr(transformers, "__version__"):
                        version = transformers.__version__.split(".")
                        # Only use load_in_8bit for older versions
                        if int(version[0]) < 4 or (int(version[0]) == 4 and int(version[1]) < 30):
                            kwargs["load_in_8bit"] = True
                        else:
                            self.logger.warning("[hf_local] Skipping 8-bit quantization due to compatibility issues")
                            raise Exception("8-bit quantization not compatible with this transformers version")
                
                # Add memory limits if specified
                max_memory = getattr(self.config, "hf_max_memory", None)
                if max_memory:
                    kwargs["max_memory"] = max_memory
                
                self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
                self.logger.info("[hf_local] Successfully loaded in 8-bit mode")
                self._started = True
                
                # Log memory after loading
                MemoryManager.log_memory_status("After 8-bit loading")
                return
                
            except Exception as e_8bit:
                log_exception(self.logger, "[hf_local] 8-bit load failed", e_8bit)
                MemoryManager.clear_memory()
        
        # Optional: 4-bit quantized load if configured (preferred on Kaggle T4×2)
        if bool(getattr(self.config, "hf_load_in_4bit", False)):
            try:
                self.logger.info("[hf_local] Attempting 4-bit quantized load")
                
                # Clear memory before loading
                MemoryManager.clear_memory()
                
                # Check if bitsandbytes is available and properly configured
                try:
                    import bitsandbytes as bnb
                    self.logger.info("[hf_local] bitsandbytes library found for 4-bit")
                    
                    # Try creating BitsAndBytesConfig manually for better compatibility
                    from transformers import BitsAndBytesConfig
                    
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16 if torch is not None else None,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    kwargs: Dict[str, Any] = {
                        "quantization_config": bnb_config,
                        "device_map": "auto",
                        "low_cpu_mem_usage": True,
                        "token": token,
                    }
                    
                except ImportError:
                    self.logger.warning("[hf_local] bitsandbytes not available, trying direct 4-bit load")
                    # Fallback to direct parameters (may not work with newer transformers)
                    kwargs: Dict[str, Any] = {
                        "device_map": "auto",
                        "low_cpu_mem_usage": True,
                        "token": token,
                        "torch_dtype": torch.float16 if torch is not None else None,
                    }
                    
                    # Only add load_in_4bit if we're sure it won't cause issues
                    import transformers
                    if hasattr(transformers, "__version__"):
                        version = transformers.__version__.split(".")
                        # Only use load_in_4bit for older versions
                        if int(version[0]) < 4 or (int(version[0]) == 4 and int(version[1]) < 30):
                            kwargs["load_in_4bit"] = True
                        else:
                            self.logger.warning("[hf_local] Skipping 4-bit quantization due to compatibility issues")
                            raise Exception("4-bit quantization not compatible with this transformers version")
                
                # Add memory limits if specified
                max_memory = getattr(self.config, "hf_max_memory", None)
                if max_memory:
                    kwargs["max_memory"] = max_memory
                    
                self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
                self.logger.info("[hf_local] Successfully loaded in 4-bit mode")
                self._started = True
                
                # Log memory after loading
                MemoryManager.log_memory_status("After 4-bit loading")
                return
                
            except Exception as e_4bit:
                log_exception(self.logger, "[hf_local] 4-bit load failed; falling back to non-quantized path", e_4bit)
                MemoryManager.clear_memory()

        # Attempt user-requested load first (as provided in the task)
        # NOTE: This often OOMs on 4 GB GPUs; we catch and fallback automatically.
        self.logger.info("[hf_local] Attempting preferred load: device_map='cuda', torch_dtype='auto'")
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="cuda",
                token=token,
            )
            self.logger.info("[hf_local] Loaded with device_map='cuda'")
        except Exception as e1:
            log_exception(self.logger, "[hf_local] Preferred load failed, trying device_map='auto'", e1)
            try:
                kwargs: Dict[str, Any] = {
                    "torch_dtype": "auto",
                    "device_map": "auto",
                    "token": token,
                }
                max_memory = getattr(self.config, "hf_max_memory", None)
                if max_memory:
                    kwargs["max_memory"] = max_memory
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **kwargs,
                )
                self.logger.info("[hf_local] Loaded with device_map='auto' (may use CPU/offload)")
            except Exception as e2:
                log_exception(self.logger, "[hf_local] device_map='auto' failed, falling back to CPU", e2)
                # Last resort: CPU-only load
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype="auto",
                    token=token,
                )
                self.logger.warning("[hf_local] Loaded on CPU. Generation will be slow.")

        self._started = True
        self.logger.info("[hf_local] Ready")

    async def close(self) -> None:
        # Nothing special to close for HF models; allow GC to reclaim
        self._started = False
        self.logger.info("[hf_local] Closed")

    def _build_text(self, prompt: str, system: Optional[str]) -> str:
        """
        Build a plain text prompt. The rest of the pipeline already collapses multi-turn.
        """
        if system:
            return f"System:\n{system}\n\nUser:\n{prompt}\n\nAssistant:"
        return prompt

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        tools: Optional[list] = None,  # ignored
        metadata: Optional[dict] = None,  # passthrough ignored
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self._started:
            await self.start()
        assert self._tokenizer is not None and self._model is not None

        text = self._build_text(prompt, system)
        max_new_tokens = int(max_tokens) if max_tokens is not None else int(getattr(self.config, "hf_max_new_tokens", 256))

        do_sample = bool(getattr(self.config, "hf_do_sample", True))
        top_p = float(getattr(self.config, "hf_top_p", 0.95))
        temp = float(getattr(self.config, "hf_temperature", temperature if temperature is not None else 0.7))

        # Tokenize (remain on CPU; Transformers/Accelerate will dispatch as needed)
        inputs = self._tokenizer(text, return_tensors="pt")
        
        # Move inputs to the same device as the model
        if hasattr(self._model, 'device'):
            device = self._model.device
        elif hasattr(self._model, 'module') and hasattr(self._model.module, 'device'):
            device = self._model.module.device
        else:
            # Try to get device from model parameters
            try:
                device = next(self._model.parameters()).device
            except:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move all input tensors to the correct device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Ensure pad/eos tokens are set
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        pad_id = getattr(self._tokenizer, "pad_token_id", None) or eos_id

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=max(0.0, temp),
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

        try:
            # Run generation in a thread to avoid blocking the event loop
            outputs = await asyncio.to_thread(self._model.generate, **inputs, **gen_kwargs)
        except Exception as e:
            log_exception(self.logger, "[hf_local] Generation failed", e)
            raise

        try:
            decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Heuristic: return everything after the original text if present
            if decoded.startswith(text):
                output_text = decoded[len(text):].lstrip()
            else:
                output_text = decoded
        except Exception:
            output_text = ""

        raw = {
            "model": self.config.model,
            "backend": "hf_local",
            "generated_text": output_text,
            "tokens": getattr(outputs, "shape", None),
        }
        return {"output_text": output_text, "raw": raw, "meta": {"cached": False, "cache_key": ""}}


__all__ = ["HFLocalClient"]
