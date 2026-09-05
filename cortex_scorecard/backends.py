from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

from .schema import BackendResponse, TraceCase, ValidationResult


class BackendUnavailable(RuntimeError):
    """Raised when an optional backend is not configured on this machine."""


@dataclass
class BackendContext:
    max_tokens: int
    temperature: float
    device: str = "auto"
    hf_home: str = ""
    offline: bool = True
    timeout_seconds: float = 60.0


class BaseBackend:
    backend_name = "base"
    model_id = ""

    def generate(self, case: TraceCase, context: BackendContext) -> BackendResponse:
        raise NotImplementedError


class DeterministicBackend(BaseBackend):
    backend_name = "deterministic"
    model_id = "fixture"

    def __init__(self, *, broken: bool = False):
        self.broken = broken
        if broken:
            self.backend_name = "deterministic_bad"
            self.model_id = "fixture_bad"

    def generate(self, case: TraceCase, context: BackendContext) -> BackendResponse:
        start = time.perf_counter()
        if self.broken:
            keys = list(case.expected.keys())
            payload = {keys[0]: case.expected[keys[0]]} if keys else {"status": "bad"}
        else:
            payload = case.expected
        return BackendResponse(
            text=json.dumps(payload, sort_keys=True),
            elapsed_s=time.perf_counter() - start,
            output_tokens=len(payload) * 6,
            metadata={"fixture": True},
        )


class LocalHFBackend(BaseBackend):
    backend_name = "local_hf"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def generate(self, case: TraceCase, context: BackendContext) -> BackendResponse:
        self._ensure_loaded(context)
        prompt = self._format_prompt(case)
        return self._generate_from_prompt(prompt, context)

    def repair(self, case: TraceCase, failed_output: str, validation: ValidationResult, context: BackendContext) -> BackendResponse:
        self._ensure_loaded(context)
        prompt = self._format_repair_prompt(case, failed_output, validation)
        return self._generate_from_prompt(prompt, context)

    def _generate_from_prompt(self, prompt: str, context: BackendContext) -> BackendResponse:
        assert self._tokenizer is not None and self._model is not None

        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        start = time.perf_counter()
        generation_kwargs = {
            **inputs,
            "max_new_tokens": context.max_tokens,
            "do_sample": context.temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if context.temperature > 0:
            generation_kwargs["temperature"] = context.temperature
        else:
            generation_config = copy.deepcopy(getattr(self._model, "generation_config", None))
            if generation_config is not None:
                for attr, value in (("temperature", None), ("top_p", None), ("top_k", None)):
                    if hasattr(generation_config, attr):
                        setattr(generation_config, attr, value)
                generation_kwargs["generation_config"] = generation_config
        with torch.no_grad():
            generated = self._model.generate(**generation_kwargs)
        elapsed = time.perf_counter() - start
        new_tokens = generated[:, inputs.input_ids.shape[1]:]
        text = self._tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        return BackendResponse(
            text=text,
            elapsed_s=elapsed,
            input_tokens=int(inputs.input_ids.numel()),
            output_tokens=int(new_tokens.numel()),
            metadata={"runtime": "transformers", "device": self._device},
        )

    def _ensure_loaded(self, context: BackendContext):
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from cortex_core.hf_utils import prepare_hf_cache, resolve_local_model_source
        from cortex_core.settings import project_root

        if context.hf_home:
            prepare_hf_cache(str(project_root()), preferred_root=context.hf_home)
        if context.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        source, local_files_only = resolve_local_model_source(self.model_id, os.environ.get("HF_HOME"))
        local_files_only = bool(local_files_only or context.offline)
        model_id_lower = str(self.model_id).lower()
        is_bitnet = any(key in model_id_lower for key in ("bitnet", "b1.58"))
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                source,
                local_files_only=local_files_only,
                trust_remote_code=True,
                fix_mistral_regex=True,
            )
        except TypeError:
            self._tokenizer = AutoTokenizer.from_pretrained(
                source,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
        except AttributeError as exc:
            if not is_bitnet and "endswith" not in str(exc):
                raise
            self._tokenizer = self._load_bitnet_tokenizer(source)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if context.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = context.device
        dtype = torch.bfloat16 if is_bitnet else (torch.float16 if self._device.startswith("cuda") else torch.float32)
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                source,
                local_files_only=local_files_only,
                trust_remote_code=True,
                dtype=dtype,
            ).to(self._device)
        except OSError as exc:
            if not is_bitnet and "configuration_bitnet" not in str(exc).lower() and "bitnet" not in str(exc).lower():
                raise
            from transformers import BitNetForCausalLM

            self._model = BitNetForCausalLM.from_pretrained(
                source,
                local_files_only=local_files_only,
                dtype=torch.bfloat16,
            ).to(self._device)
        self._model.eval()
        self._loaded = True

    def _load_bitnet_tokenizer(self, source: str):
        from transformers import PreTrainedTokenizerFast

        source_path = Path(source)
        tokenizer_file = source_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"BitNet tokenizer.json not found at {tokenizer_file}")

        tokenizer_config_path = source_path / "tokenizer_config.json"
        special_tokens_map_path = source_path / "special_tokens_map.json"

        tokenizer_config = {}
        special_tokens_map = {}
        if tokenizer_config_path.exists():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        if special_tokens_map_path.exists():
            special_tokens_map = json.loads(special_tokens_map_path.read_text(encoding="utf-8"))

        eos_token = tokenizer_config.get("eos_token") or special_tokens_map.get("eos_token")
        bos_token = tokenizer_config.get("bos_token") or special_tokens_map.get("bos_token")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            bos_token=bos_token,
            eos_token=eos_token,
        )
        chat_template = tokenizer_config.get("chat_template")
        if chat_template:
            tokenizer.chat_template = chat_template
        return tokenizer

    def _format_prompt(self, case: TraceCase) -> str:
        tokenizer = self._tokenizer
        field_contract = self._build_field_contract(case.expected)
        messages = [
            {
                "role": "system",
                "content": (
                    "Return only the requested JSON object. Do not add prose or markdown fences. "
                    "Use exactly the required top-level keys and no others. "
                    "Do not wrap the required keys under helper objects like 'data', 'result', or 'keys'. "
                    "Return atomic field values rather than explanatory phrases.\n\n"
                    f"Field contract:\n{field_contract}"
                ),
            },
            {"role": "user", "content": case.prompt},
        ]
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return "\n\n".join(f"{message['role'].title()}: {message['content']}" for message in messages) + "\nAssistant:"

    def _format_repair_prompt(self, case: TraceCase, failed_output: str, validation: ValidationResult) -> str:
        tokenizer = self._tokenizer
        field_contract = self._build_field_contract(case.expected)
        issue_summary = self._summarize_validation_issues(validation)
        messages = [
            {
                "role": "system",
                "content": (
                    "Repair the JSON extraction. Return only one corrected JSON object with exactly the required "
                    "top-level keys and no extras. Keep atomic values short and canonical. Do not add prose or markdown fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original extraction task:\n{case.prompt}\n\n"
                    f"Required field contract:\n{field_contract}\n\n"
                    f"Previous JSON output:\n{failed_output}\n\n"
                    f"Issues to fix:\n{issue_summary}\n\n"
                    "Rewrite the JSON so it satisfies the task."
                ),
            },
        ]
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return "\n\n".join(f"{message['role'].title()}: {message['content']}" for message in messages) + "\nAssistant:"

    def _build_field_contract(self, expected: Dict[str, object]) -> str:
        lines = []
        for key, value in expected.items():
            requirement = self._describe_expected_type(value)
            guidance = self._field_guidance(key, value)
            if guidance:
                requirement = f"{requirement}; {guidance}"
            lines.append(f"- {key}: {requirement}")
        return "\n".join(lines)

    def _describe_expected_type(self, value: object) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int) and not isinstance(value, bool):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "value"

    def _field_guidance(self, key: str, value: object) -> str:
        key_lower = key.lower()
        if isinstance(value, str):
            if any(token in key_lower for token in ("id", "header", "token", "seal", "code", "key")):
                return "copy the exact literal value from the prompt"
            if any(token in key_lower for token in ("action", "operation")):
                return "return the exact action verb assigned to this field, not a nearby descriptive phrase"
            if any(token in key_lower for token in ("channel", "queue", "speed")):
                return "copy the exact short label assigned to this field"
            if any(token in key_lower for token in ("reason", "status", "type", "category", "label")):
                return "return the shortest canonical label supported by the prompt, not a longer phrase"
        return ""

    def _summarize_validation_issues(self, validation: ValidationResult) -> str:
        lines = []
        if validation.missing_fields:
            lines.append("- Add these missing top-level keys: " + ", ".join(validation.missing_fields))
        if validation.mismatches:
            lines.append("- Correct these fields without adding explanations: " + ", ".join(sorted(validation.mismatches)))
        if validation.failed_checks and not lines:
            lines.append("- Fix these validation checks: " + ", ".join(validation.failed_checks))
        return "\n".join(lines) if lines else "- Return a corrected JSON object that matches the task."


class WarpBitNetBackend(LocalHFBackend):
    backend_name = "warp_bitnet"

    def __init__(self, model_id: str, *, model_dir: str = ""):
        super().__init__(model_id)
        self._checkpoint_path = Path(model_id).expanduser()
        self._model_dir = Path(model_dir).expanduser() if model_dir else self._checkpoint_path.parent

    def _ensure_loaded(self, context: BackendContext):
        if self._loaded:
            return

        import sys
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if not self._checkpoint_path.exists():
            raise FileNotFoundError(f"Warp BitNet checkpoint not found: {self._checkpoint_path}")

        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from warp_bitnet.kernel.bit_linear import BitLinear, unpack_ternary_weights

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_dir),
                trust_remote_code=True,
                fix_mistral_regex=True,
            )
        except TypeError:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_dir),
                trust_remote_code=True,
            )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if context.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = context.device

        config = AutoConfig.from_pretrained(str(self._model_dir), trust_remote_code=True)
        state = torch.load(str(self._checkpoint_path), map_location="cpu", weights_only=False)
        uses_packed_weights = any(key.endswith("packed_weight") for key in state)

        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        if uses_packed_weights:
            self._replace_linears_with_bitlinear(model, BitLinear, skip_names={"lm_head"})
            state = self._prepare_packed_state(model, state, unpack_ternary_weights)

        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        allowed_missing_suffixes = (
            "rotary_emb.inv_freq",
        )
        bad_missing_keys = [
            key for key in missing_keys if not any(key.endswith(suffix) for suffix in allowed_missing_suffixes)
        ]
        if bad_missing_keys or unexpected_keys:
            problems = []
            if bad_missing_keys:
                problems.append(f"missing_keys={bad_missing_keys[:12]}")
            if unexpected_keys:
                problems.append(f"unexpected_keys={unexpected_keys[:12]}")
            raise RuntimeError("Warp BitNet checkpoint load mismatch: " + "; ".join(problems))

        if getattr(model.config, "tie_word_embeddings", False):
            model.tie_weights()

        target_dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        self._model = model.to(device=self._device, dtype=target_dtype)
        self._model.eval()
        self._loaded = True

    def _replace_linears_with_bitlinear(self, module, bitlinear_cls, *, skip_names=None):
        import torch.nn as nn

        skip_names = set(skip_names or ())
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                if name in skip_names:
                    continue
                replacement = bitlinear_cls(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                )
                if child.bias is not None and replacement.bias is not None:
                    replacement.bias.data.copy_(child.bias.data.to(dtype=replacement.bias.dtype))
                setattr(module, name, replacement)
            else:
                self._replace_linears_with_bitlinear(child, bitlinear_cls, skip_names=skip_names)

    def _prepare_packed_state(self, model, state, unpack_ternary_weights):
        import torch

        prepared_state = dict(state)
        packed_lm_head = prepared_state.pop("lm_head.packed_weight", None)
        lm_head_scale = prepared_state.pop("lm_head.weight_scale", None)
        if packed_lm_head is None:
            return prepared_state

        lm_head_module = getattr(model, "lm_head", None)
        if lm_head_module is None or not hasattr(lm_head_module, "weight"):
            return prepared_state

        if getattr(model.config, "tie_word_embeddings", False):
            embed_weight = prepared_state.get("model.embed_tokens.weight")
            if embed_weight is not None:
                prepared_state["lm_head.weight"] = embed_weight.to(dtype=lm_head_module.weight.dtype)
                return prepared_state

        dense_weight = unpack_ternary_weights(
            packed_lm_head,
            (lm_head_module.out_features, lm_head_module.in_features),
        ).to(dtype=torch.float32)

        if lm_head_scale is not None:
            scale = lm_head_scale.to(dtype=torch.float32)
            if scale.numel() == 1:
                dense_weight = dense_weight * scale.item()
            else:
                dense_weight = dense_weight * scale.reshape(-1, 1)

        prepared_state["lm_head.weight"] = dense_weight.to(dtype=lm_head_module.weight.dtype)
        return prepared_state


class OpenAIBackend(BaseBackend):
    backend_name = "api_openai"

    def __init__(self, model_id: str, *, base_url: str = ""):
        self.model_id = model_id
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self._client = None

    def generate(self, case: TraceCase, context: BackendContext) -> BackendResponse:
        self._ensure_client()
        assert self._client is not None
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "Return only the requested JSON object. Do not add prose."},
                {"role": "user", "content": case.prompt},
            ],
            temperature=context.temperature,
            max_tokens=context.max_tokens,
        )
        elapsed = time.perf_counter() - start
        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return BackendResponse(
            text=text,
            elapsed_s=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            remote_calls=1,
            cost_usd=estimate_openai_cost(self.model_id, input_tokens, output_tokens),
            metadata={"runtime": "openai_chat_completions", "base_url": self.base_url or "default"},
        )

    def _ensure_client(self):
        if self._client is not None:
            return
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise BackendUnavailable("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise BackendUnavailable("openai package is not installed; run pip install -e .[api]") from exc
        kwargs = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)


def estimate_openai_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    per_million: Dict[str, Tuple[float, float]] = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
    }
    in_rate, out_rate = per_million.get(model_id, (0.0, 0.0))
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0