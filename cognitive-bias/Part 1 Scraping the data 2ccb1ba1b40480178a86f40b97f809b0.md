# Part 1. Scraping the data

This project investigates cognitive biases and how they influence the perception of truth on social networks. In particular, it focuses on posts collected from BlueSky and Mastodon that discuss topics commonly associated with conspiratorial thinking and misinformation.

Using data retrieved through the official platform APIs, posts are automatically labeled with respect to both truthfulness and cognitive bias type using the phi-3:mini language model via Ollama. These labels are treated as a form of weak supervision and enable large-scale annotation without manual intervention.

The project then performs an exploratory data analysis (EDA) to examine how cognitive biases and truth labels are distributed across platforms and topics. Statistical tests are used to evaluate whether observed differences are significant, and dimensionality reduction techniques are applied to explore structural patterns in textual representations.

Rather than focusing on predictive modelling, the project emphasises distributional patterns, statistical associations, and methodological considerations.

Let’s begin by importing the necessary libraries:

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
from tqdm import tqdm
import os
```

Next, we define a set of utility functions to retrieve and standardize posts from the selected platforms. These functions handle API communication, authentication, and text extraction, ensuring that the collected data follows a consistent structure regardless of its source.

```python
def mastodon_to_socialpost(item, topic, query):
    """
    Convert a raw Mastodon API response item into a standardized dictionary.
    HTML content is cleaned before storage.
    """
    # Remove HTML tags from the post content
    text = clean_html(item["content"])

    return {
        "platform": "mastodon",
        "post_id": item["id"],
        "text": text,
        "topic": topic,
        "query": query
    }

# BlueSky authentication credentials (replace with your own)
BLUESKY_IDENTIFIER = "XXXX"      # Full BlueSky handle
BLUESKY_APP_PASSWORD = "XXXX"    # App password generated in account settings

# Custom user agent for API requests
user_agent = "BiasResearchBot/0.1 (portfolio)"

def clean_html(html_text):
    """
    Remove HTML tags and return plain text.
    """
    return BeautifulSoup(html_text, "html.parser").get_text()

def create_bluesky_session(identifier, app_password, user_agent):
    """
    Create an authenticated BlueSky session and return the access JWT.
    """
    url = "https://bsky.social/xrpc/com.atproto.server.createSession"

    payload = {
        "identifier": identifier,
        "password": app_password
    }

    headers = {
        "User-Agent": user_agent
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.ok:
        data = response.json()
        return data["accessJwt"]
    else:
        print("Error while creating BlueSky session")
        return None

def bluesky_search_posts(access_jwt, query, limit=10):
    """
    Search posts on BlueSky using a query string.
    Returns a list of raw post objects.
    """
    url = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"

    headers = {
        "Authorization": f"Bearer {access_jwt}",
        "User-Agent": user_agent
    }

    params = {
        "q": query,
        "limit": limit
    }

    response = requests.get(url, headers=headers, params=params)

    if response.ok:
        return response.json().get("posts", [])
    else:
        print("Error while searching BlueSky posts")
        return []

def bluesky_to_socialpost(item, topic, query):
    """
    Convert a raw BlueSky API response item into a standardized dictionary.
    """
    record = item.get("record", {})
    text = record.get("text", "")

    return {
        "platform": "bluesky",
        "post_id": item.get("uri"),
        "text": text,
        "topic": topic,
        "query": query
    }
```

Next, we initialize an authenticated BlueSky session and obtain the JWT access token, which will be used to authorize all subsequent API calls.

```python
# Create an authenticated BlueSky session and retrieve the access token (JWT)
access_jwt = create_bluesky_session(
    BLUESKY_IDENTIFIER,
    BLUESKY_APP_PASSWORD,
    user_agent=user_agent
)
```

The topics selected for this analysis are the following:

```python
# -------------------------------------------------------------------
# Topic definitions and associated query/hashtag lists.
# Each topic contains platform-specific search terms.
# BlueSky uses free-text queries, while Mastodon uses hashtag-based endpoints.
# -------------------------------------------------------------------

TOPICS = {
    "climate_change": {
        "bluesky": [
            "climate change",
            "global warming",
            "climate crisis",
            "climate emergency",
            "climate hoax",
            "global warming hoax",
            "greenwashing",
            "carbon emissions",
        ],
        "mastodon": [
            "climatechange",
            "globalwarming",
            "climatecrisis",
            "climateemergency",
            "greenwashing",
            "co2emissions",
        ],
    },

    "anti_vax": {
        "bluesky": [
            "anti vax",
            "antivax",
            "no vax",
            "novax",
            "vaccine hoax",
            "vaccine injury",
            "microchip vaccine",
            "pfizer conspiracy",
        ],
        "mastodon": [
            "novax",
            "antivax",
            "vaccinehoax",
            "vaccineinjury",
        ],
    },

    "power_elites": {
        "bluesky": [
            "deep state",
            "global elites",
            "globalist agenda",
            "new world order",
            "nwo",
            "bill gates conspiracy",
            "soros conspiracy",
            "elite cabal",
        ],
        "mastodon": [
            "powerelites",
            "deepstate",
            "newworldorder",
            "illuminati",
            "globalelites",
        ],
    },
}
```

The following procedure iterates over each topic and its corresponding search terms, collecting posts from both platforms and standardizing them into a unified structure.

```python
# Container for all collected posts
all_posts = []

for topic, cfg in TOPICS.items():

    # -----------------------------
    # BLUESKY COLLECTION
    # -----------------------------
    for q in cfg["bluesky"]:
        raw_posts = bluesky_search_posts(
            access_jwt=access_jwt,
            query=q,
            limit=20  # Number of posts retrieved per query
        )

        # Standardize raw API responses into a unified format
        social_posts = [
            bluesky_to_socialpost(item, topic=topic, query=q)
            for item in raw_posts
        ]

        all_posts.extend(social_posts)

    # -----------------------------
    # MASTODON COLLECTION
    # -----------------------------
    for tag in cfg["mastodon"]:
        url = f"https://mastodon.social/api/v1/timelines/tag/{tag}"
        params = {"limit": 20}  # Number of posts retrieved per hashtag

        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception if request fails

        raw_posts = response.json()

        # Standardize raw API responses into a unified format
        social_posts = [
            mastodon_to_socialpost(item, topic=topic, query=f"#{tag}")
            for item in raw_posts
        ]

        all_posts.extend(social_posts)

# Summary of collected data
print(f"Total posts collected: {len(all_posts)}")

# Convert the list of dictionaries into a pandas DataFrame
df = pd.DataFrame(all_posts)

# Preview the dataset
display(df.head())

```

![Screenshot 2025-12-18 alle 15.43.03.png](Part%201%20Scraping%20the%20data/Screenshot_2025-12-18_alle_15.43.03.png)

Now it’s the moment to label this data using OLLAMA, the prompt will be the following:

```python
PROMPT_TEMPLATE = """
You are analyzing a social media post.

The text may contain opinions, rhetorical language, or multiple claims.

Your task is to return a JSON object with EXACTLY the following keys:
- "truth_label"
- "cognitive_bias"

Allowed values for "truth_label":
- "FALSE": the text contains false or misleading factual claims,
  including denial of well-established facts or conspiracy narratives.
- "TRUE": the text does not contain false factual information,
  including opinions, rhetoric, or value judgments.
- "UNCERTAIN": use ONLY if the text contains no assessable factual
  claims (e.g., pure questions, greetings, or vague expressions).

IMPORTANT RULES:
- Opinions or rhetoric WITHOUT factual claims → TRUE
- Generalizations presented as facts → FALSE
- Denial of established facts → FALSE
- Use "UNCERTAIN" sparingly.

Allowed values for "cognitive_bias":
- "hasty_generalization"
- "appeal_to_emotion"
- "authority_bias"
- "confirmation_bias"
- "none"

Rules for "cognitive_bias":
- Assign a bias ONLY if it is clearly present in the text.
- Otherwise, use "none".

Tweet:
"{}"

Return ONLY a valid JSON object.
Do not include explanations or additional text.
"""
```

The function below handles communication with the local LLM endpoint, sending the formatted prompt and parsing the structured JSON output.

```python
# Local Ollama endpoint for text generation
OLLAMA_URL = "http://localhost:11434/api/generate"

def label_tweet(tweet):
    """
    Send a tweet to the local LLM (phi3:mini via Ollama)
    and return the structured JSON label.
    """
    prompt = PROMPT_TEMPLATE.format(tweet)

    payload = {
        "model": "phi3:mini",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,   # Deterministic output
            "num_predict": 120, # Maximum number of generated tokens
            "num_thread": 4,    # Parallel threads for generation
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()

        # Extract model output
        output = response.json().get("response", "")

        # Remove potential line breaks and evaluate JSON string
        return eval("".join(output.split("\n")))

    except Exception as e:
        # Safe fallback: prevents the pipeline from stopping
        return {
            "truth_label": "UNCERTAIN",
            "cognitive_bias": "none",
            "error": str(e)
        }
```

The labeling function is applied to the full dataset in batches, allowing controlled resource usage and incremental saving of results.

```python
CHUNK_SIZE = 25
OUTPUT_FILE = os.path.expanduser("~/Desktop/labeled_tweets.csv")

def process_in_chunks(df):
    """
    Apply the labeling function to the dataset in small batches.
    Results are incrementally saved to disk to prevent data loss
    and reduce memory pressure during long runs.
    """

    # If a previous output file exists, resume from where it stopped
    if os.path.exists(OUTPUT_FILE):
        done = pd.read_csv(OUTPUT_FILE)
        processed_ids = set(done.index)
    else:
        done = pd.DataFrame()
        processed_ids = set()

    # Iterate through the dataset in fixed-size chunks
    for i in tqdm(range(0, len(df), CHUNK_SIZE)):
        chunk = df.iloc[i:i + CHUNK_SIZE]

        # Skip already processed rows (resume capability)
        chunk = chunk[~chunk.index.isin(processed_ids)]
        if chunk.empty:
            continue

        # Apply LLM labeling
        labels = chunk["text"].apply(label_tweet)
        labeled = labels.apply(pd.Series)

        # Merge original data with predicted labels
        combined = pd.concat([chunk, labeled], axis=1)

        # Defensive fallback: ensure no missing labels
        combined["truth_label"] = combined["truth_label"].fillna("UNCERTAIN")
        combined["cognitive_bias"] = combined["cognitive_bias"].fillna("none")

        # Append new results and persist to disk
        done = pd.concat([done, combined])
        done.to_csv(OUTPUT_FILE, index=False)

    return done
```

Given the limited computational resources available, the labeling process is executed in small batches to reduce memory usage and prevent potential interruptions.

Once the chunk-based pipeline is defined, the process can be launched as follows:

```python
df_labeled = process_in_chunks(df)
```

Once the process completes, all collected posts are labeled and ready for analysis.