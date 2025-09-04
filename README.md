🐍 Snake AI with PyTorch

This project implements the classic Snake Game along with an AI agent trained using Reinforcement Learning (PPO - Proximal Policy Optimization). The AI learns to play Snake autonomously and can also be played by a human for comparison.

---

📂 Project Structure

snake-ai-pytorch/
│── agent.py              # AI Agent implementation (RL logic)
│── game.py               # Snake game environment
│── helper.py             # Utility functions
│── model.py              # Neural Network model definition
│── snake_game_human.py   # Play Snake manually as a human
│── model/model.pth       # Pre-trained model weights
│── arial.ttf             # Font for rendering the game

---

🚀 Features

* Classic Snake game implementation in Python.
* AI agent trained with Deep Reinforcement Learning (PPO).
* Human-playable version for testing against the AI.
* Modular design with separated files for agent, environment, and model.
* Pre-trained model included (`model/model.pth`).

---

🛠️ Installation

1. Clone the repository:

   git clone https://github.com/your-username/snake-ai-pytorch.git
   cd snake-ai-pytorch


2. Create a virtual environment (optional but recommended):

   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
  

3. Install dependencies:

   pip install torch pygame numpy

---

🎮 Usage

1. Play as Human

Run the following to play Snake manually:

python snake_game_human.py


2. Train the AI Agent

To train the agent from scratch:

python agent.py


3. Run Pre-trained Model

If you want to watch the pre-trained AI play:

python agent.py --load-model model/model.pth

---

📊 Algorithm

* The agent is trained using **Proximal Policy Optimization (PPO)**.
* The model is a simple **Feedforward Neural Network** built with PyTorch.
* Rewards encourage survival and eating food while discouraging collisions.

---

📈 Future Improvements

* Add DQN implementation for comparison.
* Implement training visualization with TensorBoard.
* Enhance environment with obstacles and advanced states.

---

🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit pull requests.

---

📜 License

This project is licensed under the MIT License.
