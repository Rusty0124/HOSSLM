import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd  # CSV Handling
import re
import wikipediaapi
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# 📜 HOSLLM v2.8 (AI-Powered Historical Chatbot with Auto-Saving of Wikipedia Data)
# -------------------------------
class HOSLLM:
    """
    Historian Open Source LLM (HOSLLM v2.8).
    Uses AI to retrieve historical events, identify key events per year, and allow searching for people and events by name.
    Now securely integrates Wikipedia's REST API and automatically stores new data.
    """
    def __init__(self):
        load_dotenv()
        self.history_context = []  # Historical reference data
        self.conversation_context = []  # Conversational training data
        self.chat_history = []  # Memory for tracking conversation flow
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient NLP embedding model
        self.wiki = wikipediaapi.Wikipedia('en', headers={'User-Agent': 'HOSLLM'})
        self.WIKIPEDIA_API_KEY = os.getenv("WIKIPEDIA_API_KEY")
        self.history_csv_path = "HOSLLM_Historical_Dataset.csv"

    def load_csv(self, file_path):
        """
        Loads historical reference data from CSV.
        """
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            if {'Historical Event', 'Summary', 'Date'}.issubset(df.columns):
                # Ensure that the summaries are strings
                df['Summary'] = df['Summary'].astype(str)
                return df[['Historical Event', 'Date', 'Summary']].values.tolist()
            else:
                logging.warning("CSV must contain 'Historical Event', 'Date', and 'Summary' columns.")
                return []
        except Exception as e:
            logging.error(f"Error loading CSV: {e}")
            return []

    def extract_year(self, query):
        """
        Extracts a four-digit year from user input.
        """
        match = re.search(r"\b(\d{3,4})\b", query)
        return int(match.group()) if match else None

    def find_most_important_event(self, year):
        """
        Identifies the most significant event of the year.
        """
        events = [event for event in self.history_context if str(year) in str(event[1])]
        if events:
            # Ensure that the summary is a string before calculating its length
            return max(events, key=lambda x: len(str(x[2])))  # Returns event with longest summary (assumed most detailed)
        return None

    def find_by_name(self, query):
        """
        Searches for an event or person by name in the dataset.
        """
        for event, date, summary in self.history_context:
            if query.lower() in event.lower():
                return f"{event} ({date}): {summary}"
        return None

    def find_by_date_range(self, start_year, end_year):
        """
        Finds events that occurred within a specific date range.
        """
        events_in_range = [event for event in self.history_context if start_year <= int(event[1]) <= end_year]
        if events_in_range:
            return "\n".join([f"{event[0]} ({event[1]}): {event[2]}" for event in events_in_range])
        return f"No events found between {start_year} and {end_year}."

    def clean_wikipedia_summary(self, summary):
        """
        Cleans the Wikipedia summary by removing irrelevant information.
        """
        # Remove patterns like "1984 (MCMLXXXIV) was a leap year starting on Sunday..."
        summary = re.sub(r'\b\d{3,4}\b.*?century,.*?decade\.', '', summary)
        return summary.strip()

    def fetch_from_wikipedia(self, query):
        """
        Retrieves a Wikipedia summary securely via the REST API and saves new data to the dataset.
        """
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        headers = {
            "User-Agent": "HOSLLM",
            "Authorization": f"Bearer {self.WIKIPEDIA_API_KEY}"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            summary = data.get('extract', 'No summary available')[:500]  # Limit to 500 characters
            summary = self.clean_wikipedia_summary(summary)  # Clean the summary
            self.add_to_csv(query, "Unknown", summary)  # Store new data
            return f"Here's what I found on Wikipedia: {summary}..."
        elif response.status_code == 404:
            return "I couldn't find that topic on Wikipedia."
        else:
            logging.error(f"Error retrieving data from Wikipedia: {response.status_code}")
            return "There was an issue retrieving data from Wikipedia."

    def add_to_csv(self, event, date, summary):
        """
        Adds new historical data to the CSV file, preventing duplicates.
        """
        existing_entries = [e[0].lower() for e in self.history_context]
        if event.lower() in existing_entries:
            return  # Avoid saving duplicates
        
        try:
            df = pd.DataFrame([[event, date, summary]], columns=['Historical Event', 'Date', 'Summary'])
            df.to_csv(self.history_csv_path, mode='a', header=False, index=False, encoding='utf-8')
            self.history_context.append([event, date, summary])
            logging.info("New historical data saved.")
        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")

    def find_best_match(self, query):
        """
        Uses NLP-based similarity matching to retrieve the best conversational or historical response.
        Searches by year, person, or event. Stores new Wikipedia data automatically.
        """
        if query.lower().startswith("wiki"):
            return self.fetch_from_wikipedia(query.replace("wiki", "").strip())

        year = self.extract_year(query)
        if year:
            event = self.find_most_important_event(year)
            if event:
                return f"Most Important Event in {year}: {event[0]} ({event[1]}): {event[2]}"
            return f"I couldn't find a major event in {year}, but I can check Wikipedia if you type 'wiki {year}'."

        # Check for date range queries
        date_range_match = re.search(r"(\d{4})-(\d{4})", query)
        if date_range_match:
            start_year, end_year = map(int, date_range_match.groups())
            return self.find_by_date_range(start_year, end_year)

        # Check for specific questions about wars
        if "wars" in query.lower() and "occurred" in query.lower():
            date_range_match = re.search(r"(\d{4})\s*(?:AD|BC)?\s*to\s*(\d{4})\s*(?:AD|BC)?", query, re.IGNORECASE)
            if date_range_match:
                start_year, end_year = map(int, date_range_match.groups())
                return self.find_by_date_range(start_year, end_year)

        name_result = self.find_by_name(query)
        if name_result:
            return name_result

        return self.fetch_from_wikipedia(query)

    def chat(self):
        """
        Interactive chat mode with HOSLLM v2.8.
        """
        print("\n🟢 Welcome to HOSLLM v2.8 - Your AI Historian!")
        print("🔴 Type 'exit' to quit the chat.")
        print("🔵 Type 'help' for assistance.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("\n🔴 Exiting chat. Goodbye!")
                break
            elif user_input.lower() == "help":
                print("\n🔵 You can ask about historical events, people, or years.")
                print("🔵 To search Wikipedia, type 'wiki <topic>'.")
                print("🔵 To find events in a date range, type '<start_year>-<end_year>'.")
                print("🔵 To find wars in a date range, type 'What wars occurred from <start_year> to <end_year>'.\n")
                continue
            
            response = self.find_best_match(user_input)
            print(f"HOSLLM: {response}\n")

# -------------------------------
# 🚀 Running HOSLLM v2.8
# -------------------------------
if __name__ == "__main__":
    hosllm = HOSLLM()
    hosllm.history_context = hosllm.load_csv("HOSLLM_Historical_Dataset.csv")
    print(f"✅ Loaded {len(hosllm.history_context)} historical entries")
    hosllm.chat()