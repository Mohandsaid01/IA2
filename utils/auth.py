USERS = {"admin": "password123"}

def authenticate(username, password):
    return USERS.get(username) == password

