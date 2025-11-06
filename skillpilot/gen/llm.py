from ..config import OPENAI_API_KEY, OPENAI_MODEL

def llm(system, prompt, temperature=0.25, max_tokens=800):
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30,  # важный таймаут
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM ERROR] {type(e).__name__}: {e}"
    # офлайн-заглушка
    return f"[OFFLINE]\n{prompt[:500]}"
