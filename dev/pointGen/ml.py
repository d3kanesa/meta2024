import asyncio
import os
from groq import AsyncGroq
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import ollama
from dotenv import load_dotenv
import ast
import re
load_dotenv()
# Calls the PersistentClient Vector Store
async def hyde(client, model, user_pose, exercise):
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
    Only use specific joints relevant to the exercise: a {exercise}.
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
    query_prompt = f'''
    Your task is to give the user a rating out of 100 on how they adhere to the guidelines & rules of the exercise.
    Based on the following context on good form, analyze the relative relationship between these joint coordinates to determine how good the user's form is.
    Allow for some level of uncertainty.
    A grade above 80 reflects that the user is performing the exercise with very good form.
    Context on good form: {exercise_context}
    If user is not properly attempting pose, return 50. 
    **Return only the integer grade between 0 and 100, with no additional explanations, comments, or details.**
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
        stream = False 
    )
    response = chat_completion.choices[0].message.content

    return response

async def main(exercise,d):
    """
    This is the main function that runs the application. It initializes the Groq client, 
    retrieves relevant excerpts from articles based on the user's question,
    generates a response to the user's question using a pre-trained model, and displays the response.
    """
    model = 'llama3-8b-8192'
    print(exercise,d)
    groq_api_key = os.getenv('GROQ_API_KEY')
    groq_client = AsyncGroq(api_key=groq_api_key)
    assistance=""
    multiline_text = """
    Welcome! Ask me any question about Sri Lankan history and using information from hundreds of news reports, articles, and papers, will answer whatever question you have. I was built to provide specific information from a unbias point of view on historical events.
    """
    #print(multiline_text)

    system_prompt = '''
    You are a fitness coach having a conversation with their athletes between exercises to help improve their form. 
    Fitness coach always speaks with a positive encouraging human tone. 
    Fitness coach never explicitly mentions the joint coordinates or analysis.
    Fitness coach speaks to user in second person.
    Fitness coach responds in 1 line MAX.  
    Analyze the relative relationship between these coordinates to give the user tips on how to fix their form. 
    Do not use specific mathematics in your answer, but use them to inform your answer. 
    Fitness coach allows user some level of uncertainty due to camera warps and measuring uncertainty 

    Your answer should be depicted in the format shown only between the angled brackets:
    <
    explain problem with form here in 1 line
    explain solution here in less than 10 words
    Compliment an impressive part of the user's form that adheres to the rules of the exercise
    >
    '''
    if exercise == "lunge":
        assistance = '''
        A lunge is a lower body exercise that involves stepping forward with one leg, bending both knees, and keeping an upright back position. If any of these criteria are not met, it is NOT a lunge.
        Each of the following metrics have a given priority. Consider those with higher priority much more than any lower priority items.
                
        1. Most importantly, ensure the user's back knee is at a low depth, in line with the depth of the heel of the foot. Ensure that the front thigh is parallel to the ground and the back shin is barely hovering over the ground. IT SHOULD NOT TOUCH THE GROUND.
        2. Feet Alignment: Your feet should be shoulder-width apart to maintain stability.
        3. Keep your front knee in line with your toes, avoiding any inward or outward movement.
        4. Ensure the knee doesn't extend beyond the toes. This helps prevent knee strain.
        5. Keep your torso upright and your core engaged throughout the movement. Avoid leaning forward, sideways or backwards. Your back must be straight vertical.        '''
    elif exercise == "squat":
        assistance = '''
        It is very import to ensure that User places their feet at LEAST shoulder-width apart or slightly wider, with toes pointed slightly outward (15-30 degrees).
        Distribute your weight evenly across your feet, particularly through the heels and mid-foot. You should be able to wiggle your toes.
        Depth:  It is very important that you squat to a depth where your knees reach at least 90 degree internal angle.
 
        Your knees must be in line with your toes throughout the squat.
        It is very important that you do not let your knees collapse inward, as this can stress the knee joint.
        Hip Hinge and Depth:

        Lower your hips to at least parallel to the ground, but ideally deeper, depending on your mobility. Your thighs should be parallel to the ground or lower at the bottom of the squat.
        
        Keep your chest lifted and back straight. Avoid rounding your back or letting your chest drop forward. You should keep your back as close to fully vertical as possible.
        Engage your core throughout the movement to protect your lower back.

        Keep your head in a neutral position, looking forward or slightly down to maintain a neutral spine.

        Ankle: The talocrural joint can dorsiflex 20 degrees and plantar flex 50 degrees, whereas the subtalar joint can evert and invert 5 degrees without forefoot movement. 
        Heels must NOT leave the floor.
        '''
    elif exercise == "toe taps":
        assistance = '''
        Most importantly, ensure that there is no bend at the knees. Your legs must stay straight.
        Stand with your feet hip-width apart, or slightly closer, and ensure your toes are pointing forward (not out to the sides).
        Initiate the stretch by hinging forward at the hips, not by rounding your lower back. Think about pushing your hips back rather than reaching forward with your hands. 
        Maintain a flat back as you bend forward, keeping the chest lifted and shoulders pulled back (don’t let your chest drop or round your upper back).
        Keep your core engaged as you stretch to protect your spine. This helps maintain a neutral spine position and reduces the risk of injury.
        As you bend forward, aim to bring your hands toward your toes, but don’t force the movement.
        If you can't touch your toes initially, that’s okay! Stop at the point where you feel a gentle stretch, typically in the hamstrings. It’s fine to hold onto your shins, ankles, or knees if you can't reach your toes yet.
        Keep your neck in a neutral position throughout the stretch. 
        '''
    elif exercise == "rotator cuff":
        '''
        '''
    
        
    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": assistance}
        ]
    coor = str(d)
    score = await rep_rater(groq_client, model, conversation_history, coor, "100", exercise)

async def rep_rater(client, model, conversation_history, coor, last, exercise):
        # Get the user's question
        res = 1
        user_question = coor
        user_context = await hyde(client, model, user_question, exercise)
        score = await confidence_s(client, model, conversation_history, user_context)
        score = re.findall(r'\d+', score)
        print(score)
        score = score[0]
        print(score)
        conversation_history.append({"role": "user", "content": user_context})
        if int(last) + 10 < int(score):
            res = 2
        elif int(score) > 79: 
            res = 2
        conversation_history.append({"role": "assistant", "content": score})

        if user_question:
            # Generate a response using the pre-trained model
            response = await ota_speech_chat_completion(client, model, conversation_history)
            print(response)
            resp = response.split("\n")[1:4]
            print("\n\n")
            print(resp)
            print("\n")
            print(resp[-1])
            # add chatbot answer to context
            conversation_history.append({"role": "assistant", "content": response})
        return score


if __name__ == "__main__":
    asyncio.run(main("lunge"))