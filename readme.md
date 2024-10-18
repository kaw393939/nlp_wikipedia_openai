```markdown
# Story Processor

![Project Logo](path_to_logo.png) <!-- Optional: Add your project logo here -->

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up a Virtual Environment](#2-set-up-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Download spaCy Language Model](#4-download-spacy-language-model)
- [Configuration](#configuration)
  - [1. Create Configuration Files](#1-create-configuration-files)
  - [2. Set Up Environment Variables](#2-set-up-environment-variables)
- [Preparing Your Data](#preparing-your-data)
- [Running the Program](#running-the-program)
- [Understanding the Output](#understanding-the-output)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview

**Story Processor** is a Python-based application designed to analyze and generate comprehensive reports from a collection of story files written in Markdown (`.md`) format. Leveraging Natural Language Processing (NLP) techniques, it extracts key entities from each story, enriches them with information from Wikipedia and Wikidata, generates embeddings using OpenAI's API, stores vector data in Qdrant (a vector database), and compiles detailed reports that include analysis and visualizations.

**Key Functionalities:**

- **Text Preprocessing:** Corrects misspellings and tokenizes text for accurate analysis.
- **Entity Extraction:** Identifies and resolves significant entities within each story.
- **Data Enrichment:** Fetches detailed information about entities from Wikipedia and Wikidata.
- **Embedding Generation:** Creates numerical representations of story content for storage and similarity analysis.
- **Vector Storage:** Utilizes Qdrant to store and manage embeddings efficiently.
- **Report Generation:** Compiles markdown reports with structured data, analysis, and visualizations.
- **Multiprocessing Optimization:** Enhances performance by utilizing multiple CPU cores for concurrent processing.

---

## Features

- **Automated Text Processing:** Streamlines the analysis of large volumes of story content.
- **Comprehensive Data Enrichment:** Integrates information from multiple sources to provide context-rich insights.
- **Efficient Vector Management:** Uses Qdrant for scalable and performant storage of embeddings.
- **Detailed Reporting:** Generates well-structured markdown reports with tables and sections for easy interpretation.
- **Extensible Design:** Modular architecture allows for easy integration of additional features or data sources.
- **Robust Error Handling:** Implements retry mechanisms and detailed logging to handle and diagnose issues effectively.

---

## Prerequisites

Before installing and running **Story Processor**, ensure your system meets the following requirements:

- **Operating System:** Windows, macOS, or Linux.
- **Python:** Version 3.8 or higher.
- **Hardware:** Sufficient CPU cores and memory to handle multiprocessing tasks, especially when processing large datasets.
- **Internet Connection:** Required for API calls to OpenAI, Wikipedia, Wikidata, and NewsAPI (if used).

---

## Installation

Follow these step-by-step instructions to set up **Story Processor** on your local machine.

### 1. Clone the Repository

First, clone the project repository to your local machine using `git`. Replace `your-repo-url.git` with the actual repository URL.

```bash
git clone your-repo-url.git
```

Navigate into the cloned repository directory:

```bash
cd your-repo-directory
```

### 2. Set Up a Virtual Environment

It's recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python packages on your system.

#### a. Using `venv` (Built-in)

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment

# On Windows:
venv\Scripts\activate

# On Unix or MacOS:
source venv/bin/activate
```

#### b. Using `conda` (Alternative)

If you prefer using `conda`, you can create and activate a virtual environment as follows:

```bash
# Create a conda environment named 'story_processor_env' with Python 3.8
conda create -n story_processor_env python=3.8

# Activate the environment
conda activate story_processor_env
```

### 3. Install Dependencies

All required Python packages are listed in the `requirements.txt` file. Install them using `pip`.

```bash
pip install -r requirements.txt
```

**`requirements.txt` Content:**

```plaintext
openai
spacy
wikipedia
httpx
qdrant-client
python-dotenv
PyYAML
aiohttp
beautifulsoup4
fuzzywuzzy
python-Levenshtein
requests
spellchecker
matplotlib
```

**Note:**

- **`python-Levenshtein`:** Improves the performance of `fuzzywuzzy`. It's optional but recommended.
- **`matplotlib`:** Used for generating visualizations like charts.

### 4. Download spaCy Language Model

The application uses `spaCy` for NLP tasks. Download the required English language model.

```bash
python -m spacy download en_core_web_sm
```

**Optional:** For more advanced NLP tasks, you can download the transformer-based model (requires more resources):

```bash
python -m spacy download en_core_web_trf
```

**Note:** Ensure that the model name used in the script (`en_core_web_sm`) matches the downloaded model.

---

## Configuration

Proper configuration is crucial for the seamless operation of **Story Processor**. This involves setting up configuration files and environment variables.

### 1. Create Configuration Files

Ensure that your project has a `config` directory containing a `config.yaml` file with the necessary settings.

**Directory Structure:**

```
story-processor/
│
├── config/
│   └── config.yaml
├── stories/
│   └── ... (your .md files)
├── output/
│   └── report.md
├── main.py
├── requirements.txt
└── README.md
```

**`config/config.yaml` Example:**

```yaml
openai:
  settings:
    api_key: "your_openai_api_key_here"
    embedding_model: "text-embedding-ada-002"
    timeout: 30.0
    chat_model: "gpt-4"
    max_tokens: 800
    temperature: 0.7

qdrant:
  settings:
    host: "localhost"
    port: 6333
    collection:
      name: "stories_collection"
      vector_size: 1536
      distance: "COSINE"

prompts:
  analysis:
    system: "You are an AI assistant that analyzes stories."
    user: "Analyze the following story based on the content and entities provided."
  report_refinement:
    system: "You are an AI assistant that refines analysis reports."
    user: "Refine the following report using the original story and additional Wikipedia information."
```

**Key Sections:**

- **`openai`:** Contains settings related to OpenAI's API, including API keys and model configurations.
- **`qdrant`:** Configurations for connecting to the Qdrant vector database.
- **`prompts`:** Template prompts used for generating and refining reports.

### 2. Set Up Environment Variables

Sensitive information like API keys should be stored securely using environment variables. Create a `.env` file in the root directory of your project.

**`/.env` Example:**

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
NEWS_API_KEY=your_news_api_key_here  # Optional: If integrating NewsAPI for additional info
```

**Steps:**

1. **Create the `.env` File:**

   ```bash
   touch .env
   ```

2. **Add Your API Keys:**

   Open the `.env` file in a text editor and add your API keys as shown above.

**Important:** Ensure that the `.env` file is **not** committed to version control systems like Git to protect your credentials. Add `.env` to your `.gitignore` file.

---

## Preparing Your Data

Place your story files in the designated `stories/` directory. Each story should be a Markdown (`.md`) file.

**Example Directory Structure:**

```
story-processor/
│
├── stories/
│   ├── atlantis_adventure.md
│   ├── cyber_heist.md
│   └── time_traveler.md
├── config/
│   └── config.yaml
├── output/
│   └── report.md
├── main.py
├── requirements.txt
└── README.md
```

**Creating a Story File:**

1. **Navigate to the `stories/` Directory:**

   ```bash
   cd stories
   ```

2. **Create a Markdown File:**

   ```bash
   touch atlantis_adventure.md
   ```

3. **Add Content to the File:**

   Open `atlantis_adventure.md` in a text editor and add your story content in Markdown format.

**Sample Content:**

```markdown
# Atlantis Adventure

Once upon a time in the lost city of Atlantis, explorers discovered ancient technologies that could change the world...
```

---

## Running the Program

With all configurations and data in place, you're ready to run **Story Processor**.

1. **Ensure Virtual Environment is Activated:**

   ```bash
   # On Windows:
   venv\Scripts\activate

   # On Unix or MacOS:
   source venv/bin/activate
   ```

2. **Execute the Script:**

   ```bash
   python main.py
   ```

   **Note:** Ensure you're in the root directory of the project where `main.py` resides.

3. **Processing Steps:**

   - The script will read all `.md` files from the `stories/` directory.
   - For each story, it will:
     - Correct misspellings and preprocess text.
     - Extract and resolve key entities.
     - Fetch additional information from Wikipedia and Wikidata.
     - Generate embeddings using OpenAI's API.
     - Store embeddings in Qdrant.
     - Analyze the content and compile detailed reports.
   - After processing all stories, a comprehensive report will be generated in `output/report.md`.

---

## Understanding the Output

The final report is generated in Markdown format and can be viewed using any Markdown viewer or editor.

### **Report Structure:**

1. **Title:**
   - `# Analysis Report`

2. **Sections for Each Story:**
   - `## Story ID: <story_id>`
     - **Entities:**
       - Presented in a table with links to Wikipedia pages.
     - **Wikipedia Information:**
       - Detailed summaries of each entity.
     - **Analysis:**
       - Comprehensive analysis generated by OpenAI's model.
     - **Recent News:**
       - Latest news articles related to the entities (if available).

3. **Refined Analysis Report:**
   - `# Refined Analysis Report`
     - Contains refined sections for each story with improved formatting and additional insights.

### **Sample Report Snippet:**

```markdown
# Analysis Report

## Story ID: atlantis_adventure

### Entities:
| Entity | Description | URL |
|--------|-------------|-----|
| Atlantis | An advanced ancient civilization said to have sunk into the ocean. | [Link](https://en.wikipedia.org/wiki/Atlantis) |

### Wikipedia Information:
**Atlantis**: An advanced ancient civilization said to have sunk into the ocean.

### Analysis:
The story of Atlantis serves as a cautionary tale about the hubris of civilizations. The discovery of ancient technologies highlights the potential for both progress and destruction inherent in human innovation.

### Recent News:
- [Atlantis Discovery](https://news.example.com/atlantis-discovery): Scientists have uncovered new evidence supporting the existence of Atlantis.
```

**Visualizations:**

If implemented, charts like entity frequency can be saved as images in the `output/` directory and referenced in the report.

---

## Troubleshooting

Despite careful setup, you might encounter issues. Below are common problems and their solutions.

### 1. **BeautifulSoup Parser Warning Persists**

**Symptom:**

Upon running the script, warnings related to `GuessedAtParserWarning` appear in the logs.

**Solution:**

Ensure that the monkey-patch for BeautifulSoup is correctly implemented **before** any imports or usages of the `wikipedia` package. The patch must be applied at the very beginning of `main.py`.

### 2. **`TypeError`: Sequence Item ... Expected Str Instance, NoneType Found**

**Symptom:**

An error indicating that a `NoneType` object is found where a string is expected, typically during report generation.

**Solution:**

- **Review Logs:** Check `story_processor.log` for detailed debug information pinpointing where `None` values are introduced.
- **Data Validation:** Ensure that all story files contain valid content and that entity extraction is functioning correctly.
- **Default Values:** The refactored script includes defensive programming to handle `None` values, but verify your data sources for inconsistencies.

### 3. **Multiprocessing Issues**

**Symptom:**

Errors related to multiprocessing, such as pickling errors or hanging processes.

**Solution:**

- **Top-Level Worker Functions:** Ensure that worker functions (`preprocess_text_worker` and `extract_and_resolve_entities_worker`) are defined at the top level of `main.py` to be pickleable.
- **Start Method:** Confirm that the multiprocessing start method is set to `'spawn'` before creating any processes. This is especially important on Windows.
  
  ```python
  if __name__ == "__main__":
      try:
          multiprocessing.set_start_method('spawn')
      except RuntimeError:
          pass
      asyncio.run(main())
  ```
  
- **Exception Handling:** Any exceptions within worker processes will propagate back. Review the logs to understand the root cause.

### 4. **API Rate Limits Exceeded**

**Symptom:**

Errors indicating that API rate limits for OpenAI, Wikipedia, or other services have been exceeded.

**Solution:**

- **Implement Rate Limiting:** Adjust your script to respect the rate limits of each API. Introduce delays or reduce the number of concurrent requests if necessary.
- **Monitor Usage:** Keep track of your API usage to avoid unexpected throttling.

### 5. **Missing or Incorrect API Keys**

**Symptom:**

Authentication errors when making API calls.

**Solution:**

- **Verify `.env` File:** Ensure that the `.env` file contains the correct API keys and that it is properly formatted.
- **Environment Variables:** Confirm that environment variables are loaded correctly. The `python-dotenv` package should handle this, but you can print environment variables for verification.
  
  ```python
  import os
  print(os.getenv("OPENAI_API_KEY"))
  ```

### 6. **Qdrant Connection Issues**

**Symptom:**

Errors related to connecting to the Qdrant vector database, such as connection refused or timeout errors.

**Solution:**

- **Qdrant Service:** Ensure that the Qdrant service is running and accessible at the specified host and port.
- **Configuration Settings:** Verify that the `config.yaml` file contains the correct Qdrant settings.
- **Network Issues:** Check for any firewall or network configurations that might block access to Qdrant.

---

## Contributing

Contributions are welcome! If you wish to contribute to **Story Processor**, please follow these guidelines:

1. **Fork the Repository:** Click the "Fork" button at the top-right corner of the repository page.
2. **Clone Your Fork:**

   ```bash
   git clone your-forked-repo-url.git
   ```

3. **Create a New Branch:**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

4. **Make Your Changes:** Implement your feature or bug fix.
5. **Commit Your Changes:**

   ```bash
   git commit -m "Add feature: YourFeatureName"
   ```

6. **Push to Your Fork:**

   ```bash
   git push origin feature/YourFeatureName
   ```

7. **Create a Pull Request:** Navigate to the original repository and submit a pull request detailing your changes.

**Note:** Ensure that your code adheres to the project's coding standards and includes necessary documentation and tests.

---

## License

Distributed under the [MIT License](LICENSE). See `LICENSE` for more information.

---

## Acknowledgements

- [OpenAI](https://openai.com/) for providing powerful language models.
- [spaCy](https://spacy.io/) for advanced NLP capabilities.
- [Qdrant](https://qdrant.tech/) for efficient vector storage and management.
- [Wikipedia](https://www.wikipedia.org/) and [Wikidata](https://www.wikidata.org/) for rich, community-driven data.
- [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy) for string matching utilities.
- [Matplotlib](https://matplotlib.org/) for data visualization.

---

## Contact

For questions, issues, or suggestions, please contact [your-email@example.com](mailto:your-email@example.com).

---

**Happy Story Processing! 📚✨**
```

---

**Instructions:**

1. **Copy the Entire Block:**
   - Select all the text within the code block (between the triple backticks ```` ```markdown ```` and ```` ``` ````).
   
2. **Paste into `README.md`:**
   - Open your project's `README.md` file in a text editor.
   - Paste the copied content into the file.
   
3. **Customize Placeholders:**
   - **Project Logo:** Replace `path_to_logo.png` with the actual path to your project's logo image or remove the line if not applicable.
   - **Repository URL:** In the Installation section, replace `your-repo-url.git` with your actual repository URL.
   - **Your Email:** Update `[your-email@example.com](mailto:your-email@example.com)` with your actual contact email.
   - **Additional Configurations:** Ensure that `config.yaml` and `.env` files are correctly set up as per the instructions.
   
4. **Save the File:**
   - After making the necessary customizations, save the `README.md` file.

5. **Commit and Push:**
   - Add, commit, and push the updated `README.md` to your repository.

   ```bash
   git add README.md
   git commit -m "Add comprehensive README.md"
   git push origin main
   ```

---

**Final Notes:**

- **Update Links:** Ensure that all links, such as those in the Acknowledgements or any documentation links, are accurate and point to the correct resources.
- **Images:** If you include images (like the project logo or charts), ensure they are placed in the correct directories and that the paths in the `README.md` are accurate.
- **License File:** Ensure that you have a `LICENSE` file in your repository if you're referencing it in the `README.md`.
- **Testing:** After setting up, run the program to ensure that all configurations and dependencies are correctly set up, and the `README.md` accurately reflects the project's setup and usage.

By following these steps, your `README.md` will provide clear, detailed instructions and information about your **Story Processor** project, making it easy for users and contributors to understand and use your application effectively.