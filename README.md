# Closed-Loop CNT Growth using Retrieval-Augmented Generation and Human Feedback

Welcome to the repository for **ARES-Vidura**, a cutting-edge approach to accelerating Carbon Nanotube (CNT) growth via a human-AI hybrid system that leverages advanced generative AI techniques and feedback loops. This project demonstrates how AI can assist researchers—novices and experts alike—in navigating the complex synthesis process of CNTs, combining the best of theory, computation, and experimentation.

## Overview

The synthesis of Carbon Nanotubes (CNTs) is a delicate and highly controlled process, often requiring precise parameter tuning in experimental settings, such as scanning electron microscopy (SEM). To optimize this process, **ARES-Vidura** integrates **Retrieval-Augmented Generation (RAG)** with human feedback, along with machine learning (ML) models, to create an intelligent system capable of guiding CNT growth experiments. This closed-loop system aims to accelerate material discovery by providing real-time recommendations, predictive analytics, and a user-friendly interface for researchers at all levels.

### Core Components

1. **Retrieval-Augmented Generation (RAG)**: This approach augments the generative capabilities of large language models (LLMs) by enhancing the model's responses through retrieval-based techniques. By tapping into a knowledge base of CNT-specific data and simulations, the system delivers context-aware, precise answers that evolve as the model interacts with the user.
   
2. **Regression Models**: These models are used to predict optimal synthesis parameters based on training data—both simulated and real. By mimicking experimental conditions, the system alleviates the challenge of data scarcity, especially in cases where experimental data might be limited or hard to obtain.

3. **Simulation Tools for CNT Synthesis**: Integrated with the system, these tools allow users to simulate CNT growth under various parameters, aiding in the visualization of the outcomes before conducting physical experiments.

4. **Human Feedback Integration**: In this architecture, human feedback plays a critical role. The user can provide input on model predictions, refining the system's recommendations over time and enhancing its learning process. This feedback loop fosters a more dynamic and personalized research process.

### Key Features

- **Knowledge Base**: A comprehensive repository of CNT-related knowledge, including experimental protocols, synthesis parameters, and material properties, ensuring that users can access highly relevant and specific information to guide their work.
- **AI Assistance for Novice and Expert Users**: The RAG-based approach offers over 89% performance compared to public LLMs, such as GPT-3. For novice users, this results in highly accurate and intuitive guidance, while more experienced researchers benefit from refined, advanced suggestions.
- **Predictive Capabilities**: The regression models can predict synthesis parameters with less than 1% error (mean squared error), offering highly accurate recommendations based on a vast dataset of simulated and real-world experiments.
  
### Experimental Results

In our experiments, **ARES-Vidura** demonstrated impressive accuracy and usability:
- **RAG-based Model Performance**: Achieved over 89% accuracy in comparison with widely used LLMs like GPT-3, making it an effective tool for novice users who require expert-level assistance.
- **Regression Model**: Achieved an exceptionally low mean squared error (MSE) of less than 1%, indicating the model's high precision in predicting synthesis parameters based on both simulated and real data.

## Installation

To get started with **ARES-Vidura**, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ARES-Vidura.git
cd ARES-Vidura
```

### 2. Install Dependencies

We recommend using a Python virtual environment for easy management of dependencies. 

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up Simulation Tools and Knowledge Base

The simulation tools and knowledge base require specific datasets and configuration files. Make sure you have access to these resources by following the setup instructions in the **`docs/setup.md`** file.

### 4. Run the System

After installation, you can start interacting with the system by running:

```bash
python main.py
```

This will launch the user interface, where you can input CNT growth parameters and receive feedback, predictions, and recommendations from the system.

## Usage

Once the system is up and running, here’s how you can start using **ARES-Vidura**:

### 1. Start a New CNT Experiment
You’ll be prompted to enter initial CNT growth parameters, such as:

- **Temperature**
- **Pressure**
- **Catalyst type**
- **Gas flow rates**

### 2. Retrieve AI Guidance

Based on the parameters you provide, the system will utilize the **RAG** model to fetch relevant data from its knowledge base and offer recommendations. For example:

- "Given your parameters, the recommended catalyst for optimal CNT growth is [X]. However, altering the temperature slightly could result in a more stable growth pattern."

### 3. Simulate and Predict

If you want to simulate the experiment, the system will use its simulation tools to model the CNT growth process based on your inputs. It will provide you with a predicted outcome (e.g., growth rate, diameter distribution, etc.).

### 4. Provide Feedback

If the system’s prediction or recommendation doesn’t align with your expectations or experimental observations, you can provide feedback. This feedback is used to refine future recommendations and improve the accuracy of predictions for subsequent users.

### 5. Regression-based Prediction

You can also opt to use the regression models to predict optimal synthesis parameters based on a range of desired CNT properties. The system will use the training data to offer predictions with minimal error.

## Contributing

We encourage contributions to this project! If you are interested in improving or expanding **ARES-Vidura**, feel free to fork the repository and submit a pull request. For detailed information on contributing, please refer to **CONTRIBUTING.md**.

## License

This project is licensed under the MIT License.

## Conclusion

**ARES-Vidura** offers an intelligent, closed-loop system for CNT synthesis that seamlessly integrates human feedback, machine learning models, and simulation tools. By reducing data scarcity, enhancing user interaction, and offering predictive analytics, the system provides researchers with a powerful platform to explore the vast possibilities of CNT materials and accelerate material discovery. Whether you're a novice in the field or an experienced researcher, **ARES-Vidura** can significantly enhance your ability to optimize CNT growth and explore new synthesis pathways.

