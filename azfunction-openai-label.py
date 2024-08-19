import logging
import azure.functions as func
import openai
import os
from io import BytesIO

# Set up OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_data_with_gpt4(image_data):
    """Use GPT-4 to extract and process data from image."""
    try:
        response = openai.Image.create(
            file=BytesIO(image_data),
            model="gpt-4",  # Specify the model name here
            task="image-to-text"
        )
        
        # Assuming the response contains the extracted text
        text = response['choices'][0]['text']
        return text
    except Exception as e:
        logging.error(f"Error processing image with GPT-4: {e}")
        return ""

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Retrieve image from request
        image_data = req.get_body()

        # Process image using GPT-4
        text = extract_data_with_gpt4(image_data)

        if text:
            logging.info(f"Extracted Text: {text}")
            return func.HttpResponse(f"Extracted text: {text}", status_code=200)
        else:
            return func.HttpResponse("Failed to extract text.", status_code=400)
    
    except Exception as e:
        logging.error(f"Failed to process request: {e}")
        return func.HttpResponse(f"Error processing request: {str(e)}", status_code=500)
