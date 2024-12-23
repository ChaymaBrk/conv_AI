from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
def classify_query(query):
    prompt = f"Classify the following query as either 'food' or 'weather': {query}"
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",  
        prompt=prompt,
        max_tokens=10,
        temperature=0
    )
    
    return response.choices[0].text.strip()


