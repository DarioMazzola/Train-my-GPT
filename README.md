# Fine-tuning Generative NLP Models with OpenOrca/SlimOrca Dataset

This project focuses on fine-tuning state-of-the-art generative NLP models, including GPT-2, Gemma-2b, and Llama-3, using the OpenOrca/SlimOrca dataset. The goal was to enable these models to interact in a conversational style similar to ChatGPT, showcasing improved understanding and context-based responses.

The **OpenOrca-SlimOrca** project is designed to create a high-quality, ChatGPT-style conversational model by fine-tuning open-source models on dialogue datasets. This dataset, [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca), is derived from interactions between users and ChatGPT, aiming to train models capable of mimicking ChatGPT's conversational abilities. This fine-tuning task equips the model with improved understanding and response generation in human-chatbot interactions.


## Project Overview

The goal of this project is to fine-tune a conversational model on the Open-Orca SlimOrca dataset. This notebook sets up the training environment, preprocesses the dataset, and implements evaluation functions to test model responses. Key areas include:

- **Dataset Analysis and Preparation**: Conducting exploratory data analysis to understand SlimOrca's structure and properties, including statistics on conversation length, vocabulary usage, and response patterns.
- **Pre-processing and Embedding**: Implementing pre-processing techniques and embedding strategies like Word2Vec, which help create vectorized representations of the dataset’s vocabulary for efficient model training.
- **Model Training and Optimization**:
   - To accommodate model training within Google Colab’s memory limits, I utilized [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) and [Unsloth](https://unsloth.ai/), an efficiency-focused tool that allows for model compression and efficient parameter adaptation.
- **Evaluation Metrics**:
   - Models were fine-tuned using quantization, reducing memory consumption and enhancing the model’s deployment feasibility in limited-resource environments.
   - Various models were trained and evaluated to ensure quality and coherence in chatbot responses, comparing their performance on conversational benchmarks.Implementing functions to calculate model perplexity and compare generated responses to expected outputs.
- **Performance Evaluation**: Using metrics such as BLEU, ROUGE, and other NLP-specific scores, the models' responses were evaluated to ensure they meet conversational benchmarks.

## Installation

To install the necessary packages, run the following commands:

```bash
git clone <train_my_gpt>
cd train_my_gpt
pip install -q datasets plotly transformers accelerate torch
pip install -q unsloth[colab-new] xformers trl peft bitsandbytes
```

## Usage

1. **Load the Dataset**: The dataset is loaded from Hugging Face (https://huggingface.co/datasets/Open-Orca/SlimOrca) and processed for efficient training.
2. **Train the Model**: The model is configured to use 4-bit quantization and prepared for low-resource fine-tuning.
3. **Evaluate Responses**: Utility functions, including `compare_responses` and `calculate_perplexity`, are used to assess model output quality and consistency.

## Project Overview

### Baseline Model: Retrieval-Based Chatbot
The starting point was a retrieval-based chatbot, which generates responses by calculating similarity scores between input sentences and available responses in the dataset, outputting the closest match. This baseline provided a straightforward approach to establishing an initial chatbot.

### Fine-Tuning Distil GPT-2
Next, I fine-tuned Distil GPT-2, testing different generation methods, such as Beam Search and Greedy Decoding, to enhance response quality. The model demonstrated notable improvement in generating coherent and relevant answers.

### GPT-2 with LoRA Optimization
For further improvements, we trained GPT-2 with Low-Rank Adaptation (LoRA), significantly optimizing the model's performance. This setup enabled the model to accurately recognize context and provide appropriate responses.

**Example:**
```plaintext
👨‍🦱: Which are the primary colors?
🤖: Red, Yellow, Green, Blue, Indigo, Violet, and Black.
👨‍🦱: What is the color of the sky?
🤖: Blue.
👨‍🦱: What is the color of the ocean?
🤖: Blue.
```

### Experimenting with Larger Models: Gemma-2b and Llama-3
To explore current large language models (LLMs), we tested Gemma-2b and Llama-3. Due to memory constraints on Google Colab, we used quantized versions of these models, facilitated by the Unsloth library. This approach allowed us to leverage their capabilities for fine-tuning within limited computational resources.

The results were impressive: the models not only grasped context well but also demonstrated sustained conversation abilities and basic reasoning skills, capable of answering simple riddles and making logical connections.

### Chain of Thought (CoT) Reasoning and Riddle Solving
To assess their reasoning abilities, we implemented Chain of Thought (CoT) prompting, where the models think step-by-step before arriving at an answer. This process significantly improved their performance on logic quizzes and complex questions.

**Example of Riddle Solving:**
```plaintext
System: You are an AI assistant. The user will provide a logic quiz as input. Your task is to analyze the provided logic quiz and deliver a precise and accurate answer to the question posed. Think step by step and propose a reasoning. First output the reasoning and then your answer.
👨‍🦱: This is the quiz: Jimmy’s mother has three children. One of them is April and the other is May. What is the name of the other child?
🤖:
1. We are given information about Jimmy's mother having three kids: April, May, and another child.
2. To find the missing child's name, we must analyze which name is not used to describe a month.
3. In the list, "April" and "May" are both names of months, so we can eliminate them.
4. Since the remaining name "Jimmy" is a person's first name and not a name referring to a specific month, it must be the third child in the family.
5. Therefore, the answer is Jimmy is one of three siblings, with April being one, while May is another, making Jimmy the last child.
```

### Story Understanding and Comprehension Testing
We further evaluated the models by testing their comprehension with a short story (`story.pdf`) about a group of friends who share an interest in Formula One racing. The models were able to understand details from the story and answer various questions.

**Example:**
```plaintext
👨‍🦱: What is the text about?
🤖: The text is about five close friends who share a passion and excitement for Formula One racing. They gather in an apartment during the weekend to watch the Italian Grand Prix and enjoy each other's company.

👨‍🦱: What are the names of the protagonists?
🤖: The names are Alex, Elena, Marco, Sara, and Luca.

👨‍🦱: Who knows Formula 1 best?
🤖: Alex is known as a Formula One enthusiast and has encyclopedic knowledge about it.
```

## Key Results
The fine-tuned models displayed:
- **Contextual Awareness:** Accurate responses in context-sensitive scenarios.
- **Conversational Coherence:** Ability to maintain meaningful conversation across multiple exchanges.
- **Enhanced Reasoning:** Solved basic riddles and answered story-related questions, demonstrating understanding beyond surface-level text.

The trained models can be found at: https://huggingface.co/dario-mazzola/train_my_gpt/tree/main
