# 🚀 **AI Research Ideation Enhanced: An Open-Source Framework for Research Idea Generation with Visualized Interface** 🌟

Are you ready to revolutionize your research process? Introducing **AI Researcher Spark**, an upgraded and enhanced version of the research ideation pipeline inspired by Stanford NLP's groundbreaking work. Designed to empower researchers, educators, and students alike, our open-source project seamlessly integrates **domestic large language models** (compatible with OpenAPI-like platforms) and features a **user-friendly, fully visualized interface** for effortless use.

---

## 🧠 **What is AI Researcher Spark?**

AI Researcher Spark takes the research ideation process to a whole new level, building on the original "Research Ideation Agent" pipeline. Whether you're a student brainstorming for your next project or an expert refining innovative ideas, our framework is tailored to provide **detailed project proposals** that are **groundbreaking, executable, and ranked for quality**. 

💡 **Key Features:**
- Fully compatible with **domestic large language models** (e.g., MOSS, Zhipu GPT, and Qwen) for enhanced accessibility in diverse environments.
- **Visualized Interface**: A Gradio-powered, easy-to-use GUI that simplifies every step of the research ideation process.
- Seamless integration with **OpenAI-like APIs** and domestic models for global and local accessibility.
- Modular pipeline design: Use end-to-end or customize each module as standalone tools.
- Comprehensive documentation and ready-to-use scripts to get you started in minutes.

---

## 🔍 **What Can You Do with AI Researcher Spark?**

### 🔗 **1. Related Paper Search**
Effortlessly search for and rank relevant academic papers using advanced retrieval techniques, grounded in your input topic.

### 💡 **2. Grounded Idea Generation**
Generate detailed, novel research ideas—grounded in existing literature—with the option to use retrieval-augmented generation (RAG).

### ✨ **3. Idea Deduplication**
Remove redundancy and refine your ideas by leveraging sentence similarity embeddings, ensuring only the most unique ideas make it forward.

### 📋 **4. Project Proposal Generation**
Transform research ideas into **detailed project proposals**, complete with step-by-step plans for implementation.

### 📊 **5. Project Proposal Ranking**
Rank your project proposals based on their quality, novelty, and feasibility using state-of-the-art ranking models.

### ✅ **6. Project Proposal Filtering**
Automatically filter proposals for **novelty and feasibility** to ensure you're working on truly innovative projects.

---

## 🌐 **Why Choose AI Researcher Spark?**

- **Enhanced Compatibility**: Fully adapted for domestic large language models, making it suitable for users in regions with limited access to OpenAI APIs.
- **Budget-Friendly**: Generate high-quality ideas and proposals while optimizing costs for API usage.
- **Visualized & Intuitive**: No need to navigate complex scripts—our **interactive visual interface** makes the entire workflow accessible to everyone.
- **Research-Driven**: Backed by rigorous evaluation with expert reviewers, ensuring ideas generated are **novel** and **actionable**.

---

## 💻 **Get Started in Minutes**

Setting up AI Researcher Spark is simple! Follow these steps to unleash your research potential:

1. Clone the repo and set up the environment:
   ```bash
   git clone https://github.com/YourRepo/AI-Researcher-Spark.git
   cd AI-Researcher-Spark
   conda create -n ai-researcher python=3.10
   conda activate ai-researcher
   pip install -r requirements.txt
   ```
2. Configure your API keys in `keys.json` for seamless integration with models and APIs.
   ```json
   {
      "api_key": "Your OpenAI-Like API Key",
      "base_url": "Your Base URL (Optional)",
      "organization_id": "Your OpenAI Organization ID (Optional)",
      "s2_key": "Your Semantic Scholar API Key (Optional)",
      "anthropic_key": "Your Anthropic API Key（Optional)"
   }
   ```
3. Launch the **visualized interface**:
   ```bash
   python app.py
   ```
4. Explore, ideate, and innovate!

---

## 🎉 **What’s New in AI Researcher Spark?**

- **Domestic Model Support**: Use cutting-edge local models for paper search, idea generation, ranking, and more.
- **Visualized Workflow**: Fully interactive, easy-to-use Gradio interface for running the entire pipeline or individual modules.
- **Optimized for Feasibility**: Lower inference costs while ensuring superior output quality.

---

## 📝 **Call to Action**

⭐ **Star this repo** to support our open-source initiative!  
📢 **Spread the word** by sharing this project with your peers, teams, and collaborators.  
🛠️ **Contribute**: Got ideas for improvement? We welcome pull requests and feedback!  

---

## 📌 **Citation**

If you find this project useful in your research or workflow, please cite the original paper:

```bibtex
@article{si2024llmideas,
      title={Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers}, 
      author={Chenglei Si and Diyi Yang and Tatsunori Hashimoto},
      year={2024},
      journal={arXiv}
}
```

---

**Unlock the next level of research ideation with AI Researcher Spark. Your journey to groundbreaking research starts here. 🚀**