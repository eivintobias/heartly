import logging
import re
from typing import Optional, Dict, Any
import asyncio

from litellm import acompletion
from litellm.exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError
)

from config import settings
from core.interfaces import CodeGeneratorInterface
from code_generator import heartly_parser
from code_generator import local_model as local_model_module

logger = logging.getLogger(__name__)

class CodeGeneratorAgent(CodeGeneratorInterface):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_name = settings.LITELLM_DEFAULT_MODEL
        self.generation_config = {
            "temperature": settings.LITELLM_TEMPERATURE,
            "top_p": settings.LITELLM_TOP_P,
            "top_k": settings.LITELLM_TOP_K,
            "max_tokens": settings.LITELLM_MAX_TOKENS,
        }
        self.litellm_extra_params = {
            "base_url": settings.LITELLM_DEFAULT_BASE_URL,
        }

        # Local model (Heartly-Qwen-Coder) setup
        self.local_model = None
        self.local_tokenizer = None
        self.boundary_head = None
        if settings.USE_LOCAL_MODEL:
            logger.info(f"Loading local Heartly model from {settings.LOCAL_MODEL_PATH}")
            self.local_model, self.local_tokenizer = local_model_module.load_model(
                model_path=settings.LOCAL_MODEL_PATH,
                device=settings.LOCAL_MODEL_DEVICE,
                dtype=settings.LOCAL_MODEL_DTYPE,
                use_qlora=settings.LOCAL_MODEL_USE_QLORA,
            )
            # Optionally load boundary head
            if settings.USE_BOUNDARY_HEAD:
                try:
                    from code_generator.boundary_head import BoundaryHead
                    self.boundary_head = BoundaryHead(settings.BOUNDARY_HEAD_PATH)
                    logger.info(f"Boundary head loaded from {settings.BOUNDARY_HEAD_PATH}")
                except Exception as e:
                    logger.warning(f"Could not load boundary head: {e}")

        logger.info(f"CodeGeneratorAgent initialized with model: {self.model_name}")
        if settings.USE_LOCAL_MODEL:
            logger.info(f"Local Heartly model mode ACTIVE (path: {settings.LOCAL_MODEL_PATH})")

    async def generate_code(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code", litellm_extra_params: Optional[Dict[str, Any]] = None) -> str:
        effective_model_name = model_name if model_name else self.model_name
        litellm_extra_params = litellm_extra_params or self.litellm_extra_params
        logger.info(f"Attempting to generate code using model: {effective_model_name}, output_format: {output_format}")

        # ---- LOCAL MODEL MODE ----
        if settings.USE_LOCAL_MODEL and self.local_model is not None:
            logger.info("Using local Heartly model (replacing external API call)")

            # Retry configuration for when the model says "I don't know"
            max_retries = 3
            retry_encouragements = [
                "",
                "\n\nPlease try your best to solve this problem. Even a partial solution is better than no solution. Write the code now.",
                "\n\nYou MUST provide a solution. Do not say you don't know. Write a Python function that attempts to solve this problem, even if it's not perfect.",
            ]

            for retry_attempt in range(max_retries):
                logger.info(f"Heartly model attempt {retry_attempt + 1}/{max_retries}")

                # Build the prompt, adding encouragement on retries
                retry_prompt = prompt + retry_encouragements[retry_attempt]

                # If output_format is "diff", append diff format instructions to the prompt
                # so the Heartly model knows to produce SEARCH/REPLACE blocks
                if output_format == "diff":
                    retry_prompt += '''

I need you to provide your changes as a sequence of diff blocks in the following format:

<<<<<<< SEARCH
# Original code block to be found and replaced (COPY EXACTLY from original)
=======
# New code block to replace the original
>>>>>>> REPLACE

IMPORTANT DIFF GUIDELINES:
1. The SEARCH block MUST be an EXACT copy of code from the original - match whitespace, indentation, and line breaks precisely
2. Each SEARCH block should be large enough (3-5 lines minimum) to uniquely identify where the change should be made
3. Include context around the specific line(s) you want to change
4. Make multiple separate diff blocks if you need to change different parts of the code
5. For each diff, the SEARCH and REPLACE blocks must be complete, valid code segments
6. Pay special attention to matching the exact original indentation of the code in your SEARCH block, as this is crucial for correct application in environments sensitive to indentation (like Python).

Example of a good diff:
<<<<<<< SEARCH
def calculate_sum(numbers):
    result = 0
    for num in numbers:
        result += num
    return result
=======
def calculate_sum(numbers):
    if not numbers:
        return 0
    result = 0
    for num in numbers:
        result += num
    return result
>>>>>>> REPLACE

Make sure your diff can be applied correctly!
'''
                    logger.info("Appended diff format instructions to Heartly model prompt")

                heartly_prompt = heartly_parser.format_prompt(retry_prompt)

                # Generate with the local model
                # Increase temperature slightly on retries to encourage different output
                base_temp = temperature if temperature is not None else settings.LOCAL_MODEL_TEMPERATURE
                if base_temp is not None:
                    base_temp = float(base_temp)
                    # On retries, increase temperature to encourage generation
                    if retry_attempt > 0:
                        base_temp = min(base_temp + 0.1 * retry_attempt, 1.5)
                        logger.info(f"Retry {retry_attempt}: increased temperature to {base_temp}")

                local_top_p = float(settings.LOCAL_MODEL_TOP_P) if settings.LOCAL_MODEL_TOP_P else None
                local_top_k = int(settings.LOCAL_MODEL_TOP_K) if settings.LOCAL_MODEL_TOP_K else None

                raw_output = local_model_module.generate(
                    self.local_model,
                    self.local_tokenizer,
                    heartly_prompt,
                    max_new_tokens=settings.LOCAL_MODEL_MAX_TOKENS,
                    temperature=base_temp,
                    top_p=local_top_p,
                    top_k=local_top_k,
                )

                # Extract just the assistant portion (after "Assistant: ")
                assistant_idx = raw_output.find("Assistant: ")
                if assistant_idx >= 0:
                    raw_output = raw_output[assistant_idx + len("Assistant: "):]

                # Optionally run boundary head as a quality gate
                if self.boundary_head is not None:
                    try:
                        verify_idx = raw_output.find("<verify>")
                        if verify_idx >= 0:
                            hidden_state = local_model_module.extract_hidden_state_at_verify(
                                self.local_model, self.local_tokenizer,
                                heartly_prompt + raw_output, verify_idx,
                                max_length=settings.LOCAL_MODEL_MAX_TOKENS,
                            )
                            if hidden_state is not None:
                                is_known = self.boundary_head.is_known(
                                    hidden_state.cpu().numpy(),
                                    threshold=settings.BOUNDARY_HEAD_THRESHOLD,
                                )
                                if not is_known:
                                    logger.info(f"Boundary head says model doesn't know (attempt {retry_attempt + 1})")
                                    if retry_attempt < max_retries - 1:
                                        logger.info("Retrying with encouragement...")
                                        continue
                                    logger.info("All retries exhausted - returning empty code")
                                    return ""
                    except Exception as e:
                        logger.warning(f"Boundary head check failed: {e}")

                # Parse the Heartly output to extract code
                if output_format == "code":
                    code = heartly_parser.extract_code(raw_output)
                    if code is None:
                        logger.info(f"Heartly parser returned no code (attempt {retry_attempt + 1}) - trying fallback")
                        # Fallback: try to extract code directly from raw output
                        fallback_code = self._extract_code_fallback(raw_output)
                        if fallback_code:
                            logger.info("Fallback code extraction succeeded")
                            logger.debug(f"Fallback code output:\n--CODE START--\n{fallback_code}\n--CODE END--")
                            return fallback_code

                        # If this was the last retry, return empty
                        if retry_attempt < max_retries - 1:
                            logger.info(f"Fallback failed on attempt {retry_attempt + 1}, retrying with encouragement...")
                            continue
                        logger.info("All retries exhausted - returning empty code")
                        return ""
                    logger.debug(f"Heartly model code output:\n--CODE START--\n{code}\n--CODE END--")
                    return code
                else:
                    # For diff format, pass through the raw output
                    logger.debug(f"Heartly model raw output:\n{raw_output}")
                    return raw_output

            # If we get here, all retries failed
            logger.warning("All Heartly model retries exhausted - returning empty code")
            return ""

        # ---- EXTERNAL API MODE (existing) ----
        logger.info(f"Attempting to generate code using model: {effective_model_name}, output_format: {output_format}")
        
        if output_format == "diff":
            prompt += '''

I need you to provide your changes as a sequence of diff blocks in the following format:

<<<<<<< SEARCH
# Original code block to be found and replaced (COPY EXACTLY from original)
=======
# New code block to replace the original
>>>>>>> REPLACE

IMPORTANT DIFF GUIDELINES:
1. The SEARCH block MUST be an EXACT copy of code from the original - match whitespace, indentation, and line breaks precisely
2. Each SEARCH block should be large enough (3-5 lines minimum) to uniquely identify where the change should be made
3. Include context around the specific line(s) you want to change
4. Make multiple separate diff blocks if you need to change different parts of the code
5. For each diff, the SEARCH and REPLACE blocks must be complete, valid code segments
6. Pay special attention to matching the exact original indentation of the code in your SEARCH block, as this is crucial for correct application in environments sensitive to indentation (like Python).

Example of a good diff:
<<<<<<< SEARCH
def calculate_sum(numbers):
    result = 0
    for num in numbers:
        result += num
    return result
=======
def calculate_sum(numbers):
    if not numbers:
        return 0
    result = 0
    for num in numbers:
        result += num
    return result
>>>>>>> REPLACE

Make sure your diff can be applied correctly!
'''
        
        logger.debug(f"Received prompt for code generation (format: {output_format}):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        
        current_generation_config = self.generation_config.copy()
        if temperature is not None:
            current_generation_config["temperature"] = temperature
            logger.debug(f"Using temperature override: {temperature}")

        retries = settings.API_MAX_RETRIES
        delay = settings.API_RETRY_DELAY_SECONDS
        
        for attempt in range(retries):
            try:
                logger.debug(f"API Call Attempt {attempt + 1} of {retries} to {effective_model_name}.")
                response = await acompletion(
                    model=effective_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **(current_generation_config or {}),
                    **(litellm_extra_params or {})
                )
                
                if not response.choices:
                    logger.warning("LLM API returned no choices.")
                    return ""

                generated_text = response.choices[0].message.content
                logger.debug(f"Raw response from LLM API:\n--RESPONSE START--\n{generated_text}\n--RESPONSE END--")
                
                if output_format == "code":
                    if "```python" in generated_text:
                        pass
                    cleaned_code = self._clean_llm_output(generated_text)
                    logger.debug(f"Cleaned code:\n--CLEANED CODE START--\n{cleaned_code}\n--CLEANED CODE END--")
                    return cleaned_code
                else:                           
                    logger.debug(f"Returning raw diff text:\n--DIFF TEXT START--\n{generated_text}\n--DIFF TEXT END--")
                    return generated_text                        
            except (APIError, InternalServerError, TimeoutError, RateLimitError, AuthenticationError, BadRequestError) as e:
                logger.warning(f"LLM API error on attempt {attempt + 1}: {type(e).__name__} - {e}. Retrying in {delay}s...")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2 
                else:
                    logger.error(f"LLM API call failed after {retries} retries for model {effective_model_name}.")
                    raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during code generation with {effective_model_name}: {e}", exc_info=True)
                raise
        
        logger.error(f"Code generation failed for model {effective_model_name} after all retries.")
        return ""

    def _extract_code_fallback(self, raw_output: str) -> Optional[str]:
        """
        Fallback method to extract code from model output when Heartly parser fails.
        Tries multiple strategies:
        1. Extract from markdown code fences (```python ... ```)
        2. Extract from markdown code fences (``` ... ```)
        3. Find Python function definitions and extract the code block
        """
        if not raw_output or not raw_output.strip():
            return None
        
        text = raw_output.strip()
        
        # Strategy 1: Extract from ```python ... ``` fences
        python_fence_pattern = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)
        matches = python_fence_pattern.findall(text)
        if matches:
            best_match = max(matches, key=len)
            code = best_match.strip()
            if code:
                logger.debug("Fallback: extracted code from ```python fences")
                return code
        
        # Strategy 2: Extract from ``` ... ``` fences
        generic_fence_pattern = re.compile(r'```\s*\n(.*?)```', re.DOTALL)
        matches = generic_fence_pattern.findall(text)
        if matches:
            best_match = max(matches, key=len)
            code = best_match.strip()
            if code:
                logger.debug("Fallback: extracted code from generic ``` fences")
                return code
        
        # Strategy 3: Find Python function definitions
        lines = text.split('\n')
        code_lines = []
        in_function = False
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('import ') or line.strip().startswith('from '):
                in_function = True
                code_lines.append(line)
            elif in_function:
                if line.strip() == '' or line.startswith('    ') or line.startswith('\t'):
                    code_lines.append(line)
                else:
                    if line.strip().startswith('def '):
                        code_lines.append(line)
                    else:
                        break
        
        if code_lines:
            code = '\n'.join(code_lines).strip()
            if code and 'def ' in code:
                logger.debug("Fallback: extracted code from function definition pattern")
                return code
        
        # Strategy 4: If the entire output looks like code (starts with def/import/from)
        if text.startswith('def ') or text.startswith('import ') or text.startswith('from '):
            logger.debug("Fallback: entire output appears to be code")
            return text
        
        return None

    def _clean_llm_output(self, raw_code: str) -> str:
        """
        Cleans the raw output from the LLM, typically removing markdown code fences.
        """
        logger.debug(f"Attempting to clean raw LLM output. Input length: {len(raw_code)}")
        code = raw_code.strip()
        
        if code.startswith("```python") and code.endswith("```"):
            cleaned = code[len("```python"): -len("```")].strip()
            logger.debug("Cleaned Python markdown fences.")
            return cleaned
        elif code.startswith("```") and code.endswith("```"):
            cleaned = code[len("```"): -len("```")].strip()
            logger.debug("Cleaned generic markdown fences.")
            return cleaned
            
        logger.debug("No markdown fences found or standard cleaning applied to the stripped code.")
        return code

    def _apply_diff(self, parent_code: str, diff_text: str) -> str:
        """
        Applies a diff in the AlphaEvolve format to the parent code.
        """
        logger.info("Attempting to apply diff.")
        logger.debug(f"Parent code length: {len(parent_code)}")
        logger.debug(f"Diff text:\n{diff_text}")

        modified_code = parent_code
        diff_pattern = re.compile(r"<<<<<<< SEARCH\s*?\n(.*?)\n=======\s*?\n(.*?)\n>>>>>>> REPLACE", re.DOTALL)
        
        replacements_made = []
        
        for match in diff_pattern.finditer(diff_text):
            search_block = match.group(1)
            replace_block = match.group(2)
            
            search_block_normalized = search_block.replace('\r\n', '\n').replace('\r', '\n').strip()
            
            try:
                if search_block_normalized in modified_code:
                    logger.debug(f"Found exact match for SEARCH block")
                    modified_code = modified_code.replace(search_block_normalized, replace_block, 1)
                    logger.debug(f"Applied one diff block. SEARCH:\n{search_block_normalized}\nREPLACE:\n{replace_block}")
                else:
                    normalized_search = re.sub(r'\s+', ' ', search_block_normalized)
                    normalized_code = re.sub(r'\s+', ' ', modified_code)
                    
                    if normalized_search in normalized_code:
                        logger.debug(f"Found match after whitespace normalization")
                        
                        start_pos = normalized_code.find(normalized_search)
                        
                        original_pos = 0
                        norm_pos = 0
                        
                        while norm_pos < start_pos and original_pos < len(modified_code):
                            if not modified_code[original_pos].isspace() or (
                                original_pos > 0 and 
                                modified_code[original_pos].isspace() and 
                                not modified_code[original_pos-1].isspace()
                            ):
                                norm_pos += 1
                            original_pos += 1
                        
                        end_pos = original_pos
                        remaining_chars = len(normalized_search)
                        
                        while remaining_chars > 0 and end_pos < len(modified_code):
                            if not modified_code[end_pos].isspace() or (
                                end_pos > 0 and 
                                modified_code[end_pos].isspace() and 
                                not modified_code[end_pos-1].isspace()
                            ):
                                remaining_chars -= 1
                            end_pos += 1
                        
                        overlap = False
                        for start, end in replacements_made:
                            if (start <= original_pos <= end) or (start <= end_pos <= end):
                                overlap = True
                                break
                        
                        if not overlap:
                            actual_segment = modified_code[original_pos:end_pos]
                            logger.debug(f"Replacing segment:\n{actual_segment}\nWith:\n{replace_block}")
                            
                            modified_code = modified_code[:original_pos] + replace_block + modified_code[end_pos:]
                            
                            replacements_made.append((original_pos, original_pos + len(replace_block)))
                        else:
                            logger.warning(f"Diff application: Skipping overlapping replacement")
                    else:
                        search_lines = search_block_normalized.splitlines()
                        parent_lines = modified_code.splitlines()
                        
                        if len(search_lines) >= 3:
                            first_line = search_lines[0].strip()
                            last_line = search_lines[-1].strip()
                            
                            for i, line in enumerate(parent_lines):
                                if first_line in line.strip() and i + len(search_lines) <= len(parent_lines):
                                    if last_line in parent_lines[i + len(search_lines) - 1].strip():
                                        matched_segment = '\n'.join(parent_lines[i:i + len(search_lines)])
                                        
                                        modified_code = '\n'.join(
                                            parent_lines[:i] + 
                                            replace_block.splitlines() + 
                                            parent_lines[i + len(search_lines):]
                                        )
                                        logger.debug(f"Applied line-by-line match. SEARCH:\n{matched_segment}\nREPLACE:\n{replace_block}")
                                        break
                            else:
                                logger.warning(f"Diff application: SEARCH block not found even with line-by-line search:\n{search_block_normalized}")
                        else:
                            logger.warning(f"Diff application: SEARCH block not found in current code state:\n{search_block_normalized}")
            except re.error as e:
                logger.error(f"Regex error during diff application: {e}")
                continue
            except Exception as e:
                logger.error(f"Error during diff application: {e}", exc_info=True)
                continue
        
        if modified_code == parent_code and diff_text.strip():
             logger.warning("Diff text was provided, but no changes were applied. Check SEARCH blocks/diff format.")
        elif modified_code != parent_code:
             logger.info("Diff successfully applied, code has been modified.")
        else:
             logger.info("No diff text provided or diff was empty, code unchanged.")
              
        return modified_code

    async def execute(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code", parent_code_for_diff: Optional[str] = None, litellm_extra_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generic execution method.
        If output_format is 'diff', it generates a diff and applies it to parent_code_for_diff.
        Otherwise, it generates full code.
        """
        logger.debug(f"CodeGeneratorAgent.execute called. Output format: {output_format}")
        
        generated_output = await self.generate_code(
            prompt=prompt, 
            model_name=model_name, 
            temperature=temperature,
            output_format=output_format,
            litellm_extra_params=litellm_extra_params
        )

        if output_format == "diff":
            if not parent_code_for_diff:
                logger.error("Output format is 'diff' but no parent_code_for_diff provided. Returning raw diff.")
                return generated_output 
            
            if not generated_output.strip():
                 logger.info("Generated diff is empty. Returning parent code.")
                 return parent_code_for_diff

            try:
                logger.info("Applying generated diff to parent code.")
                modified_code = self._apply_diff(parent_code_for_diff, generated_output)
                return modified_code
            except Exception as e:
                logger.error(f"Error applying diff: {e}. Returning raw diff text.", exc_info=True)
                return generated_output
        else:         
            return generated_output