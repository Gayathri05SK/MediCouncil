import json
import asyncio
import aiohttp
from typing import Dict
from config.prompts import (
    EMERGENCY_AGENT_PROMPT,
    PRIMARY_CARE_AGENT_PROMPT,
    GUIDELINE_AGENT_PROMPT
)
from config.settings import LLM_CONFIG
from src.utils.logger import logger

class LLMCouncil:
    """Orchestrates YOUR 3 LLMs: DeepSeek-R1 + GPT-OSS-120B + GLM-4.5V"""
    
    def __init__(self):
        self.agents = {
            'emergency': {
                'prompt': EMERGENCY_AGENT_PROMPT,
                'config': LLM_CONFIG['emergency']
            },
            'primary': {
                'prompt': PRIMARY_CARE_AGENT_PROMPT,
                'config': LLM_CONFIG['primary']
            },
            'guideline': {
                'prompt': GUIDELINE_AGENT_PROMPT,
                'config': LLM_CONFIG['guideline']
            }
        }
    
    async def call_agent(self, agent_name: str, case: Dict) -> Dict:
        """Call YOUR specific LLM agent"""
        agent = self.agents[agent_name]
        config = agent['config']
        
        # Format prompt
        prompt = agent['prompt'].format(
            symptoms_text=case['symptoms_text'],
            age=case.get('age', 'unknown'),
            sex=case.get('sex', 'unknown'),
            chronic=', '.join(case.get('chronic_conditions', [])) or 'none',
            red_flags=', '.join(case.get('red_flags', [])) or 'none'
        )
        
        try:
            # Call YOUR specific LLM
            if agent_name == 'emergency':
                result = await self._call_deepseek(prompt, config)
            elif agent_name == 'primary':
                result = await self._call_gpt_oss_120b(prompt, config)
            elif agent_name == 'guideline':
                result = await self._call_glm_45v(prompt, config)
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
            
            result['agent'] = agent_name
            result['model'] = config['name']
            result['status'] = 'success'
            result = self._validate_output(result)
            return result
        
        except asyncio.TimeoutError:
            logger.warning(f"⚠ {config['name']} ({agent_name}) timed out")
            return self._fallback_response(agent_name, config['name'], "timeout")
        except Exception as e:
            logger.warning(f"⚠ {config['name']} ({agent_name}) failed: {e}")
            return self._fallback_response(agent_name, config['name'], "error")
    
    async def _call_deepseek(self, prompt: str, config: Dict) -> Dict:
        """DeepSeek-R1 API (Emergency Agent)"""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config['model'],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config['temperature'],
            "max_tokens": config['max_tokens'],
            "response_format": {"type": "json_object"}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config['base_url']}/chat/completions",
                headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=config['timeout'])
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return json.loads(data['choices'][0]['message']['content'])
    
    async def _call_gpt_oss_120b(self, prompt: str, config: Dict) -> Dict:
        """GPT-OSS-120B via HuggingFace (Primary Care Agent)"""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": config['temperature'],
                "max_new_tokens": config['max_tokens'],
                "return_full_text": False,
                "do_sample": True
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config['base_url']}/models/openai/gpt-oss-120b",
                headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=config['timeout'])
            ) as response:
                response.raise_for_status()
                data = await response.json()
                text = data[0]['generated_text']
                return self._extract_json_from_text(text)
    
    async def _call_glm_45v(self, prompt: str, config: Dict) -> Dict:
        """GLM-4.5V API (Guideline Agent)"""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config['model'],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config['temperature'],
            "max_tokens": config['max_tokens']
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config['base_url']}/chat/completions",
                headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=config['timeout'])
            ) as response:
                response.raise_for_status()
                data = await response.json()
                content = data['choices'][0]['message']['content']
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return self._extract_json_from_text(content)
    
    def _extract_json_from_text(self, text: str) -> Dict:
        """Extract JSON from LLM response"""
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"risk_score": 50, "urgency": "routine", "reasoning": "JSON extraction failed"}
    
    async def run_council(self, case: Dict) -> Dict:
        """Run all 3 agents in parallel"""
        tasks = [self.call_agent(name, case) for name in ['emergency', 'primary', 'guideline']]
        results = await asyncio.gather(*tasks)
        return {r['agent']: r for r in results if 'agent' in r}
    
    def _validate_output(self, output: Dict) -> Dict:
        """Validate agent output"""
        output['risk_score'] = max(0, min(100, int(output.get('risk_score', 50))))
        valid_urgency = ['emergency', 'urgent', 'routine', 'self_care']
        output['urgency'] = output.get('urgency', 'routine') if output.get('urgency') in valid_urgency else 'routine'
        output['confidence'] = max(0.0, min(1.0, float(output.get('confidence', 0.5))))
        output['reasoning'] = output.get('reasoning', 'No reasoning provided')
        output['key_symptoms'] = output.get('key_symptoms', [])
        return output
    
    def _fallback_response(self, agent: str, model: str, error: str) -> Dict:
        return {
            'agent': agent, 'model': model, 'status': 'fallback', 'error_type': error,
            'risk_score': 50, 'urgency': 'routine', 'reasoning': f'{error} fallback',
            'confidence': 0.0, 'key_symptoms': []
        }
