# HOSLLM v2.8

HOSLLM (Historian Open Source LLM) is an AI-powered historical chatbot that retrieves historical events, identifies key events per year, and allows searching for people and events by name. It securely integrates Wikipedia's REST API and automatically stores new data.

## Features

- Retrieve historical events by year
- Search for people and events by name
- Find events within a specific date range
- Fetch summaries from Wikipedia
- Automatically save new data to a CSV file

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/HOSLLM.git
    cd HOSLLM
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file with your Wikipedia API key:
    ```
    WIKIPEDIA_API_KEY=your_api_key_here
    ```

4. Run the chatbot:
    ```bash
    python [app.py](http://_vscodecontentref_/1)
    ```

## Usage

- Type a year to get the most important event of that year.
- Type a name to search for an event or person.
- Type a date range (e.g., `1900-2000`) to find events within that range.
- Type `wiki <topic>` to fetch a summary from Wikipedia.
- Type `help` for assistance.
- Type `exit` to quit the chat.

## License

This project is licensed under the MIT License.