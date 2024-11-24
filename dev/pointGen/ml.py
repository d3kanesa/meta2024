import asyncio
import os
from groq import AsyncGroq
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import ollama
from dotenv import load_dotenv
load_dotenv()

# Calls the PersistentClient Vector Store
async def hyde(client, model, user_pose):
    """
    This function takes the initial user question and reinterprets it into 3 more specific questions. In the case that a user's question is too 
    broad or bland, these new question will create a more accurate search radius and will pull more relevant chunks of data
    client (Groq): The Groq client used to interact with the untrained model.
    model (str): The name of the untrained model.
    user_question (str): The question asked by the user.
    Returns:
    list: A list of 3 relevant questions.
    """
    query_prompt = f'''
    Your goal is to generate a detailed description of the pose of the user based on the coordinates of the user's specific joints.
    Outline specific angles made by these joints, the orientation of each face of the body, and the rate of change of each joint relative to others.
    Only use specific joints relevant to the exercise: a lunge.
    User Joint Coordinates: {user_pose}
    '''
    chat_completion = await client.chat.completions.create(
        messages= [
            {"role": "system", "content" : query_prompt}
        ],
        model=model,
        temperature = 0,
        stream = False
    )
    response = chat_completion.choices[0].message.content
    #print(response)
    return response

async def confidence_s(client, model, exercise_context, user_p):
    """
    This function takes the initial user question and reinterprets it into 3 more specific questions. In the case that a user's question is too 
    broad or bland, these new question will create a more accurate search radius and will pull more relevant chunks of data
    client (Groq): The Groq client used to interact with the untrained model.
    model (str): The name of the untrained model.
    user_question (str): The question asked by the user.
    Returns:
    list: A list of 3 relevant questions.
    """
    query_prompt = f'''
    Your task is to analyze the relative relationship between these joint coordinates to determine how good the user's form is.
    Based on the following context on good form, give the user a rating out of 100 on how they adhere to the guidelines & rules of the exercise:
    Allow for some level of uncertainty.
    A high grade reflects that there is no major concern with the user's form. If there is a large discrepency or many errors with the user's form, do not report above an 80.
    Context on good form: {exercise_context}

    Return only an integer from 0 to 100.
    '''
    chat_completion = await client.chat.completions.create(
        messages= [
            {"role": "system", "content" : query_prompt},
            {"role": "user", "content": user_p}
        ],
        model=model,
        temperature = 0,
        stream = False
    )
    response = chat_completion.choices[0].message.content
    print(response)
    return response


async def ota_speech_chat_completion(client, model, conversation_history):
    """
    This function generates a response to the user's question using a pre-trained model.
    Parameters:
    client (Groq): The Groq client used to interact with the pre-trained model.
    model (str): The name of the pre-trained model.
    conversation_history (dict): A dictionary that holds all the previous questions from the user and the previous responses from the model
    Returns:
    dict: A dictionary containing the response to the user's question in .choices[0].delta.content.
    """
   
    
    
    chat_completion = await client.chat.completions.create(
        messages= conversation_history,
        model=model,
        temperature = 0, #randomness of model's answers set to 0
        stream = True 
    )
    return chat_completion

async def main(d):

    """
    This is the main function that runs the application. It initializes the Groq client, 
    retrieves relevant excerpts from articles based on the user's question,
    generates a response to the user's question using a pre-trained model, and displays the response.
    """
    model = 'llama3-8b-8192'
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_client = AsyncGroq(api_key=groq_api_key)
    
    multiline_text = """
    Welcome! Ask me any question about Sri Lankan history and using information from hundreds of news reports, articles, and papers, will answer whatever question you have. I was built to provide specific information from a unbias point of view on historical events.
    """
    #print(multiline_text)

    system_prompt = '''
    You are a fitness coach having a conversation with their athletes between exercises to help improve their form. 
    Fitness coach always speaks with a positive encouraging human tone. 
    Fitness coach responds with 20 words MAXIMUM. 
    Fitness coach never explicitly mentions the joint coordinates or analysis.
    Fitness coach speaks to user in second person.
    Fitness coach responds in 1 line MAX.  
    Analyze the relative relationship between these coordinates to give the user tips on how to fix their form. 
    Focus on one simple actionable solution for the one main concern in the user's form. Keep your answer as concise as possible.
    Do not use specific mathematics in your answer, but use them to inform your answer. 
    Focus on the biggest problem of the user's form. 
    Fitness coach complements users on good form and improvement on form.
    Fitness coach's solution should be one very actionable tip and be easy to follow
    Fitness coach allows user some level of uncertainty due to camera warps and measuring uncertainty 
    Fitness coach must respond in a dictionary format:
    
    {"problem": explain problem here in 1 line,
    "solution": explain solution here in less than 10 words,
     "compliment": Compliment an impressive part of the user's form that adheres to the rules of the exercise.}

    '''

    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": '''
            
        Here are various definitions of "good form":
        Depth: Most importantly, ensure the user's back knee is at a low depth, in line with the depth of the heel of the foot. Ensure that the front thigh is parallel to the ground and the back shin is barely hovering over the ground. IT SHOULD NOT TOUCH THE GROUND.
        Feet Alignment: Your feet should be shoulder-width apart to maintain stability.
        Step Forward with Control: When lunging, step forward with one leg, ensuring the knee doesn't extend beyond the toes. This helps prevent knee strain.
        Knee Alignment: Keep your front knee in line with your toes, avoiding any inward or outward movement.
        Lower Body: Drop your back knee low towards the ground, keeping it hovering above the floor (about an inch off the ground). Ensure your front thigh is parallel to the ground at a 90-degree angle.
        Upper Body: Keep your torso upright and your core engaged throughout the movement. Avoid leaning forward, sideways or backwards. Your back must be straight vertical.
        

        Stand in a split stance with the right foot roughly 2 to 3 feet in front of the left foot. Your torso is straight vertical, the shoulders are back and down, your core is engaged, and your hands are resting on your hips.
        Bend the knees and lower your body until the back knee is a few inches from the floor. At the bottom of the movement, the front thigh is parallel to the ground, the back knee points toward the floor, and your weight is evenly distributed between both legs.
        Push back up to the starting position, keeping your weight on the heel of the front foot.

        Your lead knee should not go past your toes as you lower toward the ground, yet it should be far out in front of your body.
        Your rear knee should not touch the ground.
        Aim to keep your hips symmetrical (at the same height, without dropping the hip of your back leg or hiking the hip of your front leg).
        Contract your abdominals during the movement to help keep your trunk upright.
        Your feet should stay hip-width apart during the landing and return.

        Step forward and slowly lower your body until your front thigh is parallel with the ground and your lower leg is leaning slightly forward. Hips should be moving primarily downward. Avoid wobbling and driving hips forward. Keep slight forward bend at hips and maintain straight back. To return to standing, push off by activating your “thigh and butt muscles” to return to an upright standing position.

        
        '''}
        ]
    
    count = 1
    while count == 1:
        # Get the user's question
        user_question = d
        user_context = await hyde(groq_client, model, user_question)
        context = await confidence_s(groq_client, model, system_prompt, user_context)
        count += 1
        conversation_history.append({"role": "user", "content": user_context})
        # Generate multiquery for user question

        if user_question:
            # Generate a response using the pre-trained model
            response = await ota_speech_chat_completion(groq_client, model, conversation_history)
            stream = []
            async for chunk in response:
                block = str(chunk.choices[0].delta.content)
                if "None" in block:
                    break
                else:
                    print(block, end="")
                    stream.append(block)
                    stream_str = "".join(stream)
            # add chatbot answer to context
            conversation_history.append({"role": "assistant", "content": stream_str})


if __name__ == "__main__":
    asyncio.run(main())