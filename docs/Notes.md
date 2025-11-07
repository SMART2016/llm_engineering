# Frontier Models
![[Screenshot 2025-11-07 at 10.39.01 AM.png]]
- **Closed source** models and paid to be used like openai , anthropic models.
- Foundation Models
- Anthropic Claude
	- haiku smallest model
	- sonnet medium model
	- opus largest model
- Why Frontier models are great![[Screenshot 2025-11-07 at 10.41.08 AM.png]]
- Limitations: ![[Screenshot 2025-11-07 at 10.49.58 AM.png]]
# Open Source Models (OSS)
- Free to use models
- Open Weight Models
	- Because they have open sourced the weights of the models and not the training data and methodology with which they trained the model.
- LLama from FB
- Qwen from Alibaba
- Phi from MS
- GPT OSS from OpenAI
- Ways to run OSS models locally
	- using Ollama
	- using HuggingFace Transformer

# Types of LLM's
- **Base Model**
	- predictive text in phone keyboards
	- They don't work with system and user prompts.
- **Chat/Instruct Model** (**RLHF** Reinforcement learning from human feedback)
	- System Prompt
	- User Prompt
	- Faster and better for interactive usecase
- **Reasoning/ Thinking Models**
	- **Reasoning Budget**
	- **Budget forcing** make the model think longer and better , inject the word `wait` while model is thinking.
	- Take more time

# Transformers
- Transformers is an architecture / ways that allows efficient way to layout artificial neurons  to create more robust and deep neural networks as layers called the attention layer.
- Neural network where multiple ML models (statistical programs trained on many data to predict outcomes) work together in a mesh to generate more accurate and robust outcomes. 
- Its an optimisation that helps models work/train with more params and data.


## Evolution 
![[Screenshot 2025-11-07 at 2.44.54 PM.png]]

# Training-Time Scaling and the Chinchilla Insight  
- Larger models generally have more parameters, which make them more capable of absorbing training information.  
- Models labeled as small, medium, or large differ primarily in the number of parameters and computational requirements.  
- The concept of training-time scaling refers to increasing model size and compute to train on more data.  
- The Chinchilla scaling laws suggest that the number of parameters should roughly match the amount of training data for optimal performance.  
- In essence, more parameters allow a model to effectively learn from and retain more data.  

# Inference-Time Scaling: Enhancing Model Performance During Use  
- Inference-time scaling focuses on improving model performance while it is being used, not during training.  
- One method is prompting the model to explain its reasoning step-by-step before producing an answer (chain-of-thought prompting).  
- Another method is providing more detailed or context-rich input data for the model to reference, such as external knowledge sources.  
- Retrieval-Augmented Generation (RAG) is an example of this approach, where additional information is fed into the model during inference.  
- Both chain-of-thought prompting and RAG exemplify inference-time scaling, which improves model output quality without retraining.

# Tokens
- Tokens are per model and not common in size.
### Common Words and One-Token Mapping
- Common words like "an", "important", "sentence", "for", "my", "class", "of", "AI", "engineers" each map to a single token.  
- Different models may tokenize text slightly differently.  
- Tokenization efficiency differences between models are minor.  
- Choose a model that suits your investigation or interests when exploring tokens.  

### Two Points About Tokens
- Each word typically maps to one token.  
- Tokens also include the preceding space as part of the token, marking it as the beginning of a new word.  
- Tokens distinguish between beginnings of words and fragments within words.  
- Example: “important” has a different token representation than part of “unimportant”.  

### Less Common Words and Subword Tokenization
- Rare or invented words are split into multiple smaller tokens.  
- Example breakdowns:  
  - “exquisitely” → “X”, “quiz”, “it”, “li”  
  - “handcrafted” → “hand”, “crafted”  
  - “masters” → “master”, “s”  
- This helps the model capture meaning more efficiently based on familiar subparts.  

### Compound and Composed Words
- Multi-part or compound words are split into multiple tokens.  
- Example:  
  - “LLM” → two tokens (since it was less common when trained)  
  - “witchcraft” → “witch”, “craft”  
- Sentence examples show 50–66 characters mapping down to 9–18 tokens, showing compression ratios.  

### Numbers and Tokenization
- Numbers are split into three-digit chunks.  
- Each three-digit sequence corresponds to one token.  
- GPT’s vocabulary includes tokens for all possible three-digit numbers.  
- Early models struggled with longer numbers because they spanned multiple tokens.  
- Modern models handle multi-token numbers much better.  

### Practical Rules of Thumb
- On average, one token represents about four characters.  
- Roughly, a token equals three-quarters of a word (1,000 tokens ≈ 750 words).  
- Complete Works of Shakespeare: ~900,000 words ≈ 1.2 million tokens.  
- Tokens are often measured in cost per million tokens.  
- Technical, mathematical, or code-heavy text consumes more tokens (sometimes close to one per character).  
- Experimenting with a tokenizer tool helps build intuition for token consumption patterns.

# Context Window
- Max number of tokens that the model can consider when generating the next token.



# References
- https://edwarddonner.com/2024/11/13/llm-engineering-resources/
- **Tokenizer**
	- https://platform.openai.com/tokenizer
- **Agents**
	- https://github.com/Aidin-Sahneh/deep-research-agent
- **Fine Tuning**
	- https://www.youtube.com/watch?v=00IK9apncCg