import gradio as gr
from huggingface_hub import InferenceClient
import json
import re
import uuid
from PIL import Image
from bs4 import BeautifulSoup
import requests
import random
from gradio_client import Client, file

def generate_caption_instructblip(image_path):
    client = Client("hysts/image-captioning-with-blip")
    return client.predict(file(image_path), api_name="/caption")

def extract_text_from_webpage(html_content):
    """Extracts visible text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove unwanted tags
    for tag in soup(["script", "style", "header", "footer", "nav", "form", "svg"]):
        tag.extract()
    return soup.get_text(strip=True)

def search(query):
    """Performs a Google search and returns the results."""
    print(f"Running web search for query: {query}")
    start = 0
    all_results = []
    max_chars_per_page = 8000  # Adjust this value based on your token limit and average webpage length
    
    with requests.Session() as session:
        try:
            resp = session.get(
                url="https://www.google.com/search",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"},
                params={"q": query, "num": 3, "udm": 14},
                timeout=5
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            result_block = soup.find_all("div", attrs={"class": "g"})
            for result in result_block:
                link = result.find("a", href=True)["href"]
                try:
                    webpage = session.get(link, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"}, timeout=5)
                    webpage.raise_for_status()
                    visible_text = extract_text_from_webpage(webpage.text)
                    if len(visible_text) > max_chars_per_page:
                        visible_text = visible_text[:max_chars_per_page]
                    all_results.append({"link": link, "text": visible_text})
                except requests.exceptions.RequestException:
                    all_results.append({"link": link, "text": None})
        except requests.exceptions.RequestException as e:
            print(f"Error during search: {e}")
    return all_results

# Initialize inference clients
client = InferenceClient("google/gemma-1.1-7b-it")
client_mixtral = InferenceClient("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
client_llama = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
image_generation_client = InferenceClient("Artples/LAI-ImageGeneration-vSDXL-2")

def respond(message, history):
    messages = []
    vqa = ""
    if message["files"]:
        try:
            for image in message["files"]:
                vqa += "[CAPTION of IMAGE] "
                gr.Info("Analyzing image")
                vqa += generate_caption_instructblip(image)
                print(vqa)
        except Exception as e:
            print(f"Error analyzing image: {e}")
            vqa = ""
    
    functions_metadata = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search query on Google and find the latest information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Web search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "general_query",
                "description": "Reply to general queries using a powerful LLM.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Detailed prompt for the LLM"},
                    },
                    "required": ["prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "image_generation",
                "description": "Generate an image based on the user's prompt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Detailed image generation prompt"},
                        "number_of_image": {"type": "integer", "description": "Number of images to generate"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "image_qna",
                "description": "Answer questions related to the image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User's question"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    message_text = message["text"]
    generate_kwargs = dict(max_new_tokens=2000, do_sample=True, stream=True, details=True, return_full_text=False)
    
    messages.append({"role": "user", "content": f'[SYSTEM] You are a helpful assistant with access to the following functions: \n {str(functions_metadata)}\n\nTo use these functions respond with:\n<functioncall> {{ "name": "function_name", "arguments": {{ "arg_1": "value_1", "arg_1": "value_1", ... }} }} </functioncall> [USER] {message_text} {vqa}'})
    
    response = client.chat_completion(messages, max_tokens=150)
    response = str(response)
    
    try:
        response = response[int(response.find("{")):int(response.rindex("</functioncall>"))]
    except Exception as e:
        print(f"Error processing response: {e}")
    
    response = response.replace("\\n", "").replace("\\'", "'").replace('\\"', '"')
    print(f"\n{response}")
    
    try:
        json_data = json.loads(response)
        function_name = json_data["name"]
        if function_name == "web_search":
            query = json_data["arguments"]["query"]
            gr.Info("Searching Web")
            web_results = search(query)
            gr.Info("Extracting relevant Info")
            web2 = ' '.join([f"Link: {res['link']}\nText: {res['text']}\n\n" for res in web_results])
            messages = f"system\nYou are H-GPT, a helpful assistant. You have access to web results to provide accurate and relevant information. Please respond to the user query using the web results."
            for msg in history:
                messages += f"\nuser\n{msg[0]}"
                messages += f"\nassistant\n{msg[1]}"
            messages += f"\nuser\n{message_text} {vqa}\nweb_result\n{web2}\nassistant\n"
            stream = client_mixtral.text_generation(messages, **generate_kwargs)
            output = ""
            for response in stream:
                if response.token.text:
                    output += response.token.text
                    yield output
        elif function_name == "image_generation":
            query = json_data["arguments"]["query"]
            gr.Info("Generating Image, Please wait...")
            seed = random.randint(1, 99999)
            image = image_generation_client.text_to_image(query)
            yield image
            gr.Info("Image generation complete.")
        elif function_name == "image_qna":
            messages = f"system\nYou are H-GPT, a helpful assistant. You are provided with both images and captions, and your task is to answer the user's questions based on the captions. Respond in a human-like style with emotions."
            for msg in history:
                messages += f"\nuser\n{msg[0]}"
                messages += f"\nassistant\n{msg[1]}"
            messages += f"\nuser\n{message_text} {vqa}\nassistant\n"
            stream = client_llama.text_generation(messages, **generate_kwargs)
            output = ""
            for response in stream:
                if response.token.text:
                    output += response.token.text
                    yield output
        else:
            messages = f"system\nYou are H-GPT, a helpful assistant. You answer users' queries like a human friend. You are an expert in many fields and strive to provide the best response possible. Show emotions using emojis and reply in a friendly tone."
            for msg in history:
                messages += f"\nuser\n{msg[0]}"
                messages += f"\nassistant\n{msg[1]}"
            messages += f"\nuser\n{message_text} {vqa}\nassistant\n"
            stream = client_llama.text_generation(messages, **generate_kwargs)
            output = ""
            for response in stream:
                if response.token.text:
                    output += response.token.text
                    yield output
    except Exception as e:
        print(f"Error handling function call: {e}")
        messages = f"system\nYou are H-GPT, a helpful assistant. You answer users' queries like a human friend. You are an expert in many fields and strive to provide the best response possible. Show emotions using emojis and reply in a friendly tone."
        for msg in history:
            messages += f"\nuser\n{msg[0]}"
            messages += f"\nassistant\n{msg[1]}"
        messages += f"\nuser\n{message_text} {vqa}\nassistant\n"
        stream = client_llama.text_generation(messages, **generate_kwargs)
        output = ""
        for response in stream:
            if response.token.text:
                output += response.token.text
                yield output

demo = gr.ChatInterface(
    fn=respond,
    chatbot=gr.Chatbot(show_copy_button=True, likeable=True, layout="panel"),
    title="H-GPT",
    textbox=gr.MultimodalTextbox(),
    multimodal=True,
    concurrency_limit=20,
    examples=[
        {"text": "Hi, who are you?"},
        {"text": "What's the current price of Bitcoin"},
        {"text": "Create a beautiful image of the Eiffel Tower at night"},
        {"text": "Write me a Python function to calculate the first 10 digits of the Fibonacci sequence."},
        {"text": "What's the color of the car in the given image", "files": ["./car1.png", "./car2.png"]},
        {"text": "Read what's written on the paper", "files": ["./paper_with_text.png"]}
    ],
    cache_examples=False
)

demo.launch()

