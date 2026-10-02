from langchain_core.tools import tool 
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
import cv2 



@tool
def get_image_info(image_path: str):
    """takes an image path and returns a string with its resolution, channel count, and data type."""
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError("Image not found!")
    h, w, c = img.shape
    dtype = img.dtype 
    return f"The image {image_path} is {h}x{w}x{c} with {dtype} data type."

@tool 
def crop_region(image_path: str, x_start: int, y_start: int, width: int, height: int, output_path: str):
    """ takes an image path (str), x_start (int), y_start (int), width (int), height (int), and an output path (str). It must crop the image and save it."""
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError("Image not found!")
    cropped = img[y_start: y_start+height, x_start: x_start+width]
    cv2.imwrite(output_path, cropped)
    return 

llm = ChatMistralAI(model_name="mistral-small-latest")

app = create_agent(model=llm, tools=[get_image_info, crop_region])

# llm_with_tools = llm.bind_tools([get_image_info, crop_region])
# response = llm_with_tools.invoke("")
prompt = "What is the resolution of sample.jpg, and can you crop a 100x100 square from the top left and save it as images/sample.jpg ?"
response = app.invoke({"messages" : [{"role" : "user", "content" : prompt}]})