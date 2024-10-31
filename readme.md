Here's a simplified `README.md` with basic installation instructions focused on libraries and dependencies:

# Story Processor Project

## Overview

The **Story Processor** is a Python-based application that processes text data, extracts entities, generates embeddings, and visualizes results using a variety of NLP and AI tools.

[Read More](description.md)

## Installation Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/story-processor.git
cd story-processor
```

### 2. Set Up Virtual Environment

It is recommended to use a virtual environment to manage your dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Python Libraries

Once the virtual environment is activated, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 4. Install spaCy Model

Download the spaCy model for English, which is required for named entity recognition:

```bash
python -m spacy download en_core_web_sm
```

### 5. Optional: Install Qdrant (For Vector Database)

If you need to use Qdrant, install it locally or use Docker to run it:

#### Install via Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Alternatively, follow the official [Qdrant documentation](https://qdrant.tech/documentation/quick_start/) to install it locally.

### 6. Run the Application

Once all dependencies are installed, you can run the main application:

```bash
python main.py
```

### 7. Additional Dependencies

If any further dependencies are required (such as system-level libraries), they will be specified in the error messages or log output when running the application.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
