import json
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def fetch_weather_data(latitude, longitude, date=None, forecast_days=None):
    """
    Fetch weather data (current, past, or future) using WeatherAPI.com.

    Args:
        latitude (str): Latitude of the location.
        longitude (str): Longitude of the location.
        date (str): Optional. Date for historical data in YYYY-MM-DD format.
        forecast_days (int): Optional. Number of days for future forecasts (1-10).
    
    Returns:
        str: JSON string containing the weather data or an error message.
    """
    base = "http://api.weatherapi.com/v1"
    key = os.getenv("WEATHER_API_KEY")

    if date:
        # Fetch historical weather
        endpoint = f"{base}/history.json"
        request_url = f"{endpoint}?key={key}&q={latitude},{longitude}&dt={date}"
    elif forecast_days:
        # Fetch future weather
        endpoint = f"{base}/forecast.json"
        request_url = f"{endpoint}?key={key}&q={latitude},{longitude}&days={forecast_days}"
    else:
        # Fetch current weather
        endpoint = f"{base}/current.json"
        request_url = f"{endpoint}?key={key}&q={latitude},{longitude}"

    response = requests.get(request_url)

    if response.status_code != 200:
        return json.dumps({"error": f"API request failed with status code {response.status_code}", "details": response.text})

    try:
        return json.dumps(response.json())
    except ValueError:
        return json.dumps({"error": "Failed to parse JSON from Weather API response"})

def get_weather_response(latitude, longitude, date=None, forecast_days=None):
    raw_data = fetch_weather_data(latitude, longitude, date, forecast_days)
    data = json.loads(raw_data)

    if "error" in data:
        return raw_data

    if "current" in data:
        return json.dumps({
            "latitude": latitude,
            "longitude": longitude,
            "temperature_c": data["current"]["temp_c"],
            "condition": data["current"]["condition"]["text"],
            "humidity": data["current"]["humidity"],
            "wind_kph": data["current"]["wind_kph"]
        })
    elif "forecast" in data:
        forecast = data["forecast"]["forecastday"]
        return json.dumps({
            "latitude": latitude,
            "longitude": longitude,
            "forecast": [
                {
                    "date": day["date"],
                    "temperature_c": day["day"]["avgtemp_c"],
                    "condition": day["day"]["condition"]["text"]
                }
                for day in forecast
            ]
        })
    elif "history" in data:
        history = data["forecast"]["forecastday"][0]
        return json.dumps({
            "latitude": latitude,
            "longitude": longitude,
            "date": history["date"],
            "temperature_c": history["day"]["avgtemp_c"],
            "condition": history["day"]["condition"]["text"]
        })

    return json.dumps({"error": "Unexpected API response format"})

def run_conversation(content):
    messages = [{"role": "user", "content": content}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "fetch_weather_data",
                "description": "Fetch current, past, or future weather for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "string", "description": "The latitude of a place"},
                        "longitude": {"type": "string", "description": "The longitude of a place"},
                        "date": {"type": "string", "description": "Optional. Date for historical weather (YYYY-MM-DD)"},
                        "forecast_days": {"type": "integer", "description": "Optional. Number of forecast days (1-10)"}
                    },
                    "required": ["latitude", "longitude"]
                }
            }
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)

        available_functions = {
            "fetch_weather_data": get_weather_response,
        }
        for tool_call in tool_calls:
            print(f"Function: {tool_call.function.name}")
            print(f"Params: {tool_call.function.arguments}")
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                latitude=function_args.get("latitude"),
                longitude=function_args.get("longitude"),
                date=function_args.get("date"),
                forecast_days=function_args.get("forecast_days")
            )
            print(f"API: {function_response}")
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            stream=True
        )
        return second_response

if __name__ == "__main__":
    # Example question
    question = "What's the weather like in tunisia tomorrow ?"
    response = run_conversation(question)
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end='', flush=True)
