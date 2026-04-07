SceneChat-Agent

An intelligent agent simulation framework that enables dynamic character interactions within a defined world setting. This project leverages Retrieval-Augmented Generation (RAG) to ground agent behaviors and dialogues in rich contextual documents, allowing for coherent and immersive multi-agent simulations.

📁simulation Example Files

.
├── data/
│   ├── world.md          # World background and rules
│   └── character.md      # Character profiles and traits
├── simulation_log.txt     # Output log of the simulation

🚀 Quick Start

Prerequisites
Python 3.9+
A LLM provider API key(qwen is recommended), for other LLM provider, you need to update the get_llm() function in history.py.
Required packages: llama-index, chromadb, python-dotenv, etc.

Install dependencies:
pip install -r requirements.txt

Setup
Create a .env file in the root directory:
      LLM_API_KEY=your__api_key_here,
   LLM_MODEL=your_llm_name,
   EMBEDDING_MODEL=your_embedding_model (such as text-embedding-v2)
   

Prepare your scenario:
   python main.py
run the main to generate world.md and character.md by user prompt
and start simulate by a specific scenario from user.


Interact with the agent by typing questions or commands. Type quit, exit, or q to stop.

Note: On first run, the system will index your documents using ChromaDB. Subsequent runs will load from persistent storage (./storage_chroma).

📝 Example Scenario

World Setting (data/world.md)(Your world description will appear here)

Character Profile (data/character.md)(Your character definition will appear here)

Sample Interaction(Your example input and output will appear here)

Simulation Log (simulationlog.txt)
The full conversation history is saved to simulationlog.txt after each session for analysis and replay.

🔧 How It Works

Document Ingestion:  
   The system loads world.md and character.md as contextual knowledge.

Intelligent Indexing:  
   Documents are parsed using Markdown structure and embedded into a vector store (ChromaDB) for semantic retrieval.

Context-Aware Querying:  
   When you ask a question, the RAG engine retrieves relevant world/character context and generates a grounded response using the specified LLM.

Persistent Storage:  
   Indexed data is cached locally to avoid reprocessing on every run.

🛠️ Customization

Change LLM: Update LLM_MODEL in .env (e.g., qwen-plus, qwen-turbo)
Add More Characters: Extend character.md with additional profiles (use consistent Markdown headers)
Modify World Rules: Update world.md to change environment dynamics
Adjust Retrieval Depth: Modify similarity_top_k in main.py to control context breadth

⚠️ Notes

This project currently supports DashScope models only (Qwen series),but it's easy to add your LLM provider.
Ensure your .env file is not committed to version control.
The system assumes UTF-8 encoded Markdown files.

> ⚠️ **Note on API Costs**  
> This project uses external LLM services. Providing your own API key enables access to these services, but **you will be billed by the provider for all usage**. Be sure to monitor your quota and set spending limits in your cloud account to avoid unexpected charges.

🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request with improvements or new features.
Built with ❤️ using LlamaIndex and ChromaDB