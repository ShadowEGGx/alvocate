from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["interview_chatbot"]

# Collections for storing users and interview results
users_collection = db["users"]
interviews_collection = db["interviews"]

# Function to add a user
def add_user(username, password):
    if users_collection.find_one({"username": username}):
        return {"error": "User already exists"}
    users_collection.insert_one({"username": username, "password": password})
    return {"message": "User added successfully"}

# Function to store interview results
def store_interview_result(username, question, user_answer, correct_answer, time_taken):
    interviews_collection.insert_one({
        "username": username,
        "question": question,
        "user_answer": user_answer,
        "correct_answer": correct_answer,
        "time_taken": time_taken
    })

print("MongoDB Database initialized!")
