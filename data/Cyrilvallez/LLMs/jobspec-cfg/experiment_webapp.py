import os

import gradio as gr

from TextWiz.textwiz import HFCausalModel, GenericConversation
import TextWiz.textwiz.webapp as wi
from TextWiz.textwiz.webapp import generator
from helpers import utils

# Disable analytics (can be set to anything except True really, we set it to False)
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# Load model
MODEL = HFCausalModel('llama2-70B-chat')

# File where the valid credentials are stored
CREDENTIALS_FILE = os.path.join(utils.ROOT_FOLDER, '.gradio_login.txt')

# This will be a mapping between users and current conversation, to reload them with page reload
CACHED_CONVERSATIONS = {}

# Need to define one logger per user
LOGGERS = {}


def chat_generation(conversation: GenericConversation, prompt: str, max_new_tokens: int) -> generator[tuple[str, GenericConversation, list[list]]]:
    yield from wi.chat_generation(MODEL, conversation=conversation, prompt=prompt, max_new_tokens=max_new_tokens,
                                  do_sample=True, top_k=None, top_p=0.9, temperature=0.8, use_seed=False,
                                  seed=None, system_prompt='')
        

def continue_generation(conversation: GenericConversation, additional_max_new_tokens) -> generator[tuple[GenericConversation, list[list]]]:
    yield from wi.continue_generation(MODEL, conversation=conversation, additional_max_new_tokens=additional_max_new_tokens,
                                      do_sample=True, top_k=None, top_p=0.9, temperature=0.8, use_seed=False, seed=None)
   

def authentication(username: str, password: str) -> bool:
    return wi.simple_authentication(CREDENTIALS_FILE, username, password)
    


def clear_chatbot(username: str) -> tuple[GenericConversation, str, str, list[list[str]]]:
    """Erase the conversation history and reinitialize the elements.

    Parameters
    ----------
    model_name : str
        Current "true" model name.
    username : str
        The username of the current session.

    Returns
    -------
    tuple[GenericConversation, str, str, list[list[str]]]
        Corresponds to the tuple of components (conversation, conv_id, prompt_chat, output_chat)
    """

    # Create new global conv object (we need a new unique id)
    conversation = MODEL.get_empty_conversation()
    # Cache value
    CACHED_CONVERSATIONS[username] = conversation

    return conversation, conversation.id, '', conversation.to_gradio_format()



def loading(request: gr.Request) -> tuple[GenericConversation, str, str, str, str, list[list[str]]]:
    """Retrieve username and all cached values at load time, and set the elements to the correct values.

    Parameters
    ----------
    request : gr.Request
        Request sent to the app.

    Returns
    -------
    tuple[GenericConversation, str, str, str, str, list[list[str]]]
        Corresponds to the tuple of components (conversation, conv_id, username, model_name, output_chat)
    """

    # Retrieve username
    if request is not None:
        username = request.username
    else:
        raise RuntimeError('Impossible to find username on startup.')
    
    # Check if we have cached a value for the conversation to use
    if username in CACHED_CONVERSATIONS.keys():
        actual_conv = CACHED_CONVERSATIONS[username]
    else:
        actual_conv = MODEL.get_empty_conversation()
        CACHED_CONVERSATIONS[username] = actual_conv
        LOGGERS[username] = gr.CSVLogger()

    conv_id = actual_conv.id
    
    return actual_conv, conv_id, username, MODEL.model_name, actual_conv.to_gradio_format()
    



# Define generation parameters
max_new_tokens = gr.Slider(10, 4000, value=500, step=10, label='Max new tokens',
                           info='Maximum number of new tokens to generate.')
max_additional_new_tokens = gr.Slider(1, 500, value=100, step=1, label='Max additional new tokens',
                           info='Maximum number of new tokens to generate when using "Continue last answer" feature.')


# Define elements of the chatbot Tab
prompt_chat = gr.Textbox(placeholder='Write your prompt here.', label='Prompt')
output_chat = gr.Chatbot(label='Conversation')
generate_button_chat = gr.Button('Generate text', variant='primary')
continue_button_chat = gr.Button('Continue last answer', variant='primary')
clear_button_chat = gr.Button('Clear conversation')


# State variable to keep one conversation per session (default value does not matter here -> it will be set
# by loading() method anyway)
conversation = gr.State(MODEL.get_empty_conversation())


# Define NON-VISIBLE elements: they are only used to keep track of variables and save them to the callback.
model_name = gr.Textbox(MODEL.model_name, label='Model Name', visible=False)
username = gr.Textbox('', label='Username', visible=False)
conv_id = gr.Textbox('', label='Conversation id', visible=False)


# Define the inputs for the main inference
inputs_to_chatbot = [prompt_chat, max_new_tokens]
# Define inputs for the logging callbacks
inputs_to_chat_callback = [model_name, max_new_tokens, max_additional_new_tokens, output_chat, conv_id, username]


# Some prompt examples
prompt_examples = [
    "Please write a function to multiply 2 numbers `a` and `b` in Python.",
    "Hello, what's your name?",
    "What's the meaning of life?",
    "How can I write a Python function to generate the nth Fibonacci number?",
    ("Here is my data {'Name':['Tom', 'Brad', 'Kyle', 'Jerry'], 'Age':[20, 21, 19, 18], 'Height' :"
     " [6.1, 5.9, 6.0, 6.1]}. Can you provide Python code to plot a bar graph showing the height of each person?"),
]


demo = gr.Blocks(title='Text generation with LLMs')

with demo:

    # State variable
    conversation.render()

    # Variables we track with usual components: they do not need to be State variables -- will not be visible
    model_name.render()
    conv_id.render()
    username.render()

    # Actual UI
    output_chat.render()
    prompt_chat.render()
    with gr.Row():
        generate_button_chat.render()
        continue_button_chat.render()
        clear_button_chat.render()

    # Accordion for generation parameters
    with gr.Accordion("Text generation parameters", open=False):
        max_new_tokens.render()
        max_additional_new_tokens.render()

    gr.Markdown("### Prompt Examples")
    gr.Examples(prompt_examples, inputs=prompt_chat)


    # Perform chat generation when clicking the button
    generate_event1 = gr.on(triggers=[generate_button_chat.click, prompt_chat.submit], fn=chat_generation,
                            inputs=[conversation, *inputs_to_chatbot], outputs=[prompt_chat, conversation, output_chat],
                            concurrency_id='generation', concurrency_limit=4)

    # Add automatic callback on success (args[-1] is the username)
    generate_event1.success(lambda *args: LOGGERS[args[-1]].flag(args, flag_option=f'generation'),
                            inputs=inputs_to_chat_callback, preprocess=False, queue=False, concurrency_limit=None)
    
    # Continue generation when clicking the button
    generate_event2 = continue_button_chat.click(continue_generation, inputs=[conversation, max_additional_new_tokens],
                                                 outputs=[conversation, output_chat], concurrency_id='generation')
    
    # Add automatic callback on success (args[-1] is the username)
    generate_event2.success(lambda *args: LOGGERS[args[-1]].flag(args, flag_option=f'continuation'),
                            inputs=inputs_to_chat_callback, preprocess=False, queue=False, concurrency_limit=None)
    
    # Clear the prompt and output boxes when clicking the button
    clear_button_chat.click(clear_chatbot, inputs=[username], outputs=[conversation, conv_id, prompt_chat, output_chat],
                            queue=False, concurrency_limit=None)
    
    # Correctly set all variables and callback at load time
    loading_events = demo.load(loading, outputs=[conversation, conv_id, username, model_name, output_chat],
                               queue=False, concurrency_limit=None)
    loading_events.then(lambda username: LOGGERS[username].setup(inputs_to_chat_callback, flagging_dir=f'chatbot_logs/{username}'),
                        inputs=username, queue=False, concurrency_limit=None)


if __name__ == '__main__':
    demo.queue(default_concurrency_limit=None).launch(share=False, server_port=7860, auth=authentication,
                                                      favicon_path='https://ai-forge.ch/favicon.ico')
