import json

USERS_FILE = "users.json"

def load_users():
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading users: {e}")
        return []

def save_users(users):
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        print(f"Error saving users: {e}")

def add_subscription_field():
    users = load_users()
    updated = False
    for user in users:
        if 'subscription' not in user:
            user['subscription'] = {
                'plan': None,
                'status': 'inactive',
                'start_date': None,
                'expiry_date': None,
                'stripe_customer_id': None,
                'stripe_subscription_id': None
            }
            updated = True
    if updated:
        save_users(users)
        print("Updated users.json with subscription fields.")
    else:
        print("No updates needed.")

if __name__ == "__main__":
    add_subscription_field()