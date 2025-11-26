"""
Simple verifier for RLVR.
Rewards completions based on vowel count.
"""

def verifier(completion: str, max_reward: float = 15.0) -> float:
    """
    Simple verifier that counts vowels (a, e, i, o, u) in the completion.
    
    Args:
        completion: Generated text string
        max_reward: Maximum reward cap
    
    Returns:
        Reward score (capped at max_reward)
    
    Constraints:
        - Single EOS token enforced during generation
        - Max tokens: 50
        - Vowels counted: a, e, i, o, u (case-insensitive)
        - Score = min(total vowel count, max_reward)
    """
    # Define vowels
    vowels = set('aeiouAEIOU')
    
    # Count vowels in completion
    count = sum(1 for char in completion if char in vowels)
    
    # Cap at max_reward
    reward = min(float(count), max_reward)
    
    return reward


def verify_batch(completions: list[str], max_reward: float = 10.0) -> list[float]:
    """
    Apply verifier to a batch of completions.
    
    Args:
        completions: List of generated text strings
        max_reward: Maximum reward cap
    
    Returns:
        List of reward scores
    """
    return [verifier(comp, max_reward) for comp in completions]


if __name__ == "__main__":
    # Test the verifier
    test_cases = [
        "Hello world",
        "The quick brown fox",
        "aeiouAEIOU",
        "programming",
        "Beautiful mountains"
    ]
    
    print("Verifier Test:")
    print("-" * 50)
    for text in test_cases:
        score = verifier(text)
        vowel_count = sum(1 for c in text if c in 'aeiouAEIOU')
        print(f"Text: '{text}'")
        print(f"Vowels: {vowel_count}, Score: {score}")
        print()