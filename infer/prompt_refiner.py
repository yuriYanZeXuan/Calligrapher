"""
Prompt Refiner: 基于 VLM API 的 prompt 优化接口

委托 VLMAgent 完成实际的 LLM 调用。
"""

from typing import Optional

from VLM_agent import VLMAgent


# 模块级便捷函数使用的默认 agent
_default_agent: Optional[VLMAgent] = None


def _get_default_agent() -> VLMAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = VLMAgent()
    return _default_agent


def refine_prompt(
    prompt: str,
    text_content: Optional[str] = None,
    temperature: float = 0.7,
    num_variants: int = 1,
) -> list[str]:
    """优化用户 prompt（模块级便捷函数）。"""
    agent = _get_default_agent()
    return agent.refine_prompt(
        prompt=prompt,
        text_content=text_content,
        num_variants=num_variants,
        temperature=temperature,
    )


class PromptRefiner:
    """Prompt 优化器类"""

    def __init__(self, temperature: float = 0.7, vlm_agent: Optional[VLMAgent] = None):
        self.temperature = temperature
        self._agent = vlm_agent

    @property
    def agent(self) -> VLMAgent:
        if self._agent is None:
            self._agent = VLMAgent()
        return self._agent

    def refine(
        self,
        prompt: str,
        text_content: Optional[str] = None,
        num_variants: int = 1,
    ) -> list[str]:
        return self.agent.refine_prompt(
            prompt=prompt,
            text_content=text_content,
            num_variants=num_variants,
            temperature=self.temperature,
        )

    def __call__(
        self,
        prompt: str,
        text_content: Optional[str] = None,
        num_variants: int = 1,
    ) -> list[str]:
        return self.refine(prompt, text_content, num_variants)


if __name__ == "__main__":
    test_prompt = "爱因斯坦在黑板前写公式"
    test_text = "x = (-b ± √(b²-4ac)) / 2a"

    print("原始 prompt:", test_prompt)
    print("文字内容:", test_text)
    print("\n优化结果:")

    results = refine_prompt(test_prompt, test_text, num_variants=2)
    for i, r in enumerate(results, 1):
        print(f"\n变体 {i}:")
        print(r)
