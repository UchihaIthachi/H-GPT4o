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

def generate_caption_instructblip(image_path, question):
    client = Client("hysts/image-captioning-with-blip")
    return client.predict(file(image_path), f"Answer this Question in detail {question}", api_name="/caption")

def extract_text_from_webpage(html_content):
    """Extracts visible text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove unwanted tags
    for tag in soup(["script", "style", "header", "footer", "nav", "form", "svg"]):
        tag.extract()
    return soup.get_text(strip=True)

# Perform a Google search and return the results
def search(query):
    """Performs a Google search and returns the results."""
    term=query
    print(f"Running web search for query: {term}")
    start = 0
    all_results = []
    # Limit the number of characters from each webpage to stay under the token limit
    max_chars_per_page = 8000  # Adjust this value based on your token limit and average webpage length
    
    with requests.Session() as session: 
        resp = session.get(  
            url="https://www.google.com/search",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"}, 
                params={
                    "q": term,
                    "num": 3,
                    "udm": 14,
                },
                timeout=5,
                verify=None,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        result_block = soup.find_all("div", attrs={"class": "g"})
        for result in result_block:
            link = result.find("a", href=True)
            link = link["href"]
            try:
                webpage = session.get(link, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"}, timeout=5,verify=False) 
                webpage.raise_for_status()
                visible_text = extract_text_from_webpage(webpage.text)
                        # Truncate text if it's too long
                if len(visible_text) > max_chars_per_page:
                    visible_text = visible_text[:max_chars_per_page]
                all_results.append({"link": link, "text": visible_text})
            except requests.exceptions.RequestException as e:
                all_results.append({"link": link, "text": None})
    return all_results


client = InferenceClient("google/gemma-1.1-7b-it")

def respond(
    message, history
):
    messages = []
    vqa=""
    if message["files"]:
        try:
            for image in message["files"]: 
                vqa += "[CAPTION of IMAGE]  "
                gr.Info("Analyzing image")
                vqa += generate_caption_instructblip(image, message["text"])
                print(vqa)
        except:
            vqa = ""
            
            
        
    functions_metadata = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search query on google and find latest information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "web search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "general_query",
                "description": "Reply general query of USER through LLM like you, it does'nt know latest information. But very helpful in general query. Its very powerful LLM. It knows many thing just like you except latest things, or thing that you don't know.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed prompt so that an LLm can understand better, what user wants.",
                        }
                    },
                    "required": ["prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "image_generation",
                "description": "Generate image for user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "image generation prompt in detail.",
                        },
                        "number_of_image": {
                            "type": "integer",
                            "description": "number of images to generate.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "image_qna",
                "description": "Answer question asked by user related to image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Question by user",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    message_text = message["text"]

    client_mixtral = InferenceClient("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
    client_llama = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
    generate_kwargs = dict( max_new_tokens=2000, do_sample=True, stream=True, details=True, return_full_text=False )

    messages.append({"role": "user", "content": f'[SYSTEM]You are a helpful assistant with access to the following functions: \n {str(functions_metadata)}\n\nTo use these functions respond with:\n<functioncall> {{ "name": "function_name", "arguments": {{ "arg_1": "value_1", "arg_1": "value_1", ... }} }} </functioncall> [USER] {message_text} {vqa}'})

    response = client.chat_completion( messages, max_tokens=150)
    response = str(response)
    try:
        response = response[int(response.find("{")):int(response.rindex("</functioncall>"))]
    except:
        print("A error occured")
    response = response.replace("\\n", "")
    response = response.replace("\\'", "'")
    response = response.replace('\\"', '"')
    print(f"\n{response}")
    # Extract JSON content from the response
    try:
        json_data = json.loads(str(response))
        if json_data["name"] == "web_search":
            query = json_data["arguments"]["query"]
            gr.Info("Searching Web")
            web_results = search(query)
            gr.Info("Extracting relevant Info")
            web2 = ' '.join([f"Link: {res['link']}\nText: {res['text']}\n\n" for res in web_results])
            messages = f"<|im_start|>system\nYou are OpenGPT 4o mini a helpful assistant made by KingNish. You are provided with WEB results from which you can find informations to answer users query in Structured and More better way. You do not say Unnecesarry things Only say thing which is important and relevant. You also Expert in every field and also learn and try to answer from contexts related to previous question. Try your best to give best response possible to user. You also try to show emotions using Emojis and reply like human, use short forms, friendly tone and emotions.<|im_end|>"
            for msg in history:
                messages += f"\n<|im_start|>user\n{str(msg[0])}<|im_end|>"
                messages += f"\n<|im_start|>assistant\n{str(msg[1])}<|im_end|>"
            messages+=f"\n<|im_start|>user\n{message_text} {vqa}<|im_end|>\n<|im_start|>web_result\n{web2}<|im_end|>\n<|im_start|>assistant\n"
            stream = client_mixtral.text_generation(messages, **generate_kwargs)
            output = ""
            for response in stream:
                if not response.token.text == "<|im_end|>":
                    output += response.token.text
                    yield output
        elif json_data["name"] == "image_generation":
            query = json_data["arguments"]["query"]
            gr.Info("Generating Image, Please wait...")
            seed = random.randint(1, 99999)
            image = f"![](https://image.pollinations.ai/prompt/{query}?{seed})"
            yield image
            gr.Info("We are going to Update Our Image Generation Engine to more powerful ones in Next Update. ThankYou")
        else:
            messages = f"<|start_header_id|>system\nYou are OpenGPT 4o mini a helpful assistant made by KingNish. You answers users query like human friend. You are also Expert in every field and also learn and try to answer from contexts related to previous question. Try your best to give best response possible to user. You also try to show emotions using Emojis and reply like human, use short forms, friendly tone and emotions.<|end_header_id|>"
            for msg in history:
                messages += f"\n<|start_header_id|>user\n{str(msg[0])}<|end_header_id|>"
                messages += f"\n<|start_header_id|>assistant\n{str(msg[1])}<|end_header_id|>"
            messages+=f"\n<|start_header_id|>user\n{message_text} {vqa}<|end_header_id|>\n<|start_header_id|>assistant\n"
            stream = client_llama.text_generation(messages, **generate_kwargs)
            output = ""
            for response in stream:
                if not response.token.text == "<|eot_id|>":
                    output += response.token.text
                    yield output
    except:
        messages = f"<|start_header_id|>system\nYou are OpenGPT 4o mini a helpful assistant made by KingNish. You answers users query like human friend. You are also Expert in every field and also learn and try to answer from contexts related to previous question. Try your best to give best response possible to user. You also try to show emotions using Emojis and reply like human, use short forms, friendly tone and emotions.<|end_header_id|>"
        for msg in history:
            messages += f"\n<|start_header_id|>user\n{str(msg[0])}<|end_header_id|>"
            messages += f"\n<|start_header_id|>assistant\n{str(msg[1])}<|end_header_id|>"
        messages+=f"\n<|start_header_id|>user\n{message_text} {vqa}<|end_header_id|>\n<|start_header_id|>assistant\n"
        stream = client_llama.text_generation(messages, **generate_kwargs)
        output = ""
        for response in stream:
            if not response.token.text == "<|eot_id|>":
                output += response.token.text
                yield output

demo = gr.ChatInterface(fn=respond,
                        chatbot=gr.Chatbot(show_copy_button=True, likeable=True, layout="panel"), 
                        title="OpenGPT 4o mini", 
                        textbox=gr.MultimodalTextbox(), 
                        multimodal=True,
                        concurrency_limit=20,
                        examples=[{"text": "Hy, who are you?",},
                                {"text": "What's the current price of Bitcoin",},
                                {"text": "Write me a Python function to calculate the first 10 digits of the fibonacci sequence.",},
                                {"text": "Create A Beautiful image of Effiel Tower at Night",},
                                {"text": "What's the colour of Car in given image","files": ["./car1.png", "./car2.png"]},
                                {"text": "Read what's written on paper", "files": ["./paper_with_text.png"]}],
                        cache_examples=False)

demo.launch()