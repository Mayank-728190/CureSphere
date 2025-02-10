import streamlit as st
import pandas as pd
import numpy as np
import bcrypt
from datetime import datetime

# Temporary centralized database for medicines and store details
centralized_db = {
    "stores": [
        {"id": 1, "name": "Health Pharmacy", "location": "Location A", "distance": 2.5, "open": True},
        {"id": 2, "name": "Wellness Store", "location": "Location B", "distance": 1.2, "open": True},
        {"id": 3, "name": "Care Plus", "location": "Location C", "distance": 3.0, "open": False},
        {"id": 4, "name": "MedicZone", "location": "Location D", "distance": 2.8, "open": True}
    ],
    "medicines": [
        {"name": "Paracetamol", "usage": "Fever and pain relief", "expiry": "2025-02-01", "store_id": 1, "quantity": 50},
        {"name": "Ibuprofen", "usage": "Inflammation and pain relief", "expiry": "2025-01-30", "store_id": 2, "quantity": 30},
        {"name": "Cetrizine", "usage": "Allergy relief", "expiry": "2025-03-15", "store_id": 3, "quantity": 20},
        {"name": "Amoxicillin", "usage": "Bacterial infections", "expiry": "2025-01-25", "store_id": 4, "quantity": 40},
    ]
}

# Store login credentials
store_credentials = {
    1: {"username": "health_admin", "password": bcrypt.hashpw("password1".encode(), bcrypt.gensalt())},
    2: {"username": "wellness_admin", "password": bcrypt.hashpw("password2".encode(), bcrypt.gensalt())},
    3: {"username": "careplus_admin", "password": bcrypt.hashpw("password3".encode(), bcrypt.gensalt())},
    4: {"username": "mediczone_admin", "password": bcrypt.hashpw("password4".encode(), bcrypt.gensalt())},
}

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "store_id" not in st.session_state:
    st.session_state.store_id = None
if "medicines" not in st.session_state:
    st.session_state.medicines = centralized_db["medicines"]  # Store medicines in session state

# Page: Login
def login_page():
    st.sidebar.title("Login")
    store_id = st.sidebar.selectbox("Select Store", options=[1, 2, 3, 4], format_func=lambda x: centralized_db["stores"][x-1]["name"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == store_credentials[store_id]["username"] and bcrypt.checkpw(password.encode(), store_credentials[store_id]["password"]):
            st.session_state.store_id = store_id
            st.session_state.page = "Inventory"
            st.rerun()
        else:
            st.error("Invalid username or password!")

# Page: Inventory
def inventory_page():
    store_id = st.session_state.store_id
    st.title(f"{centralized_db['stores'][store_id-1]['name']} - Inventory")
    
    # Display store inventory
    store_inventory = [med for med in st.session_state.medicines if med["store_id"] == store_id]
    inventory_df = pd.DataFrame(store_inventory)
    st.write(inventory_df)

    # Navigate to Billing
    if st.button("Go to Billing"):
        st.session_state.page = "Billing"
        st.rerun()

# Page: Billing
def billing_page():
    store_id = st.session_state.store_id
    st.title("Billing System")

    # Search Medicine
    st.header("Search Medicine")
    search_query = st.text_input("Enter medicine name to search:")
    if st.button("Search"):
        results = [
            {
                "store": centralized_db["stores"][med["store_id"]-1]["name"],
                "location": centralized_db["stores"][med["store_id"]-1]["location"],
                "distance": centralized_db["stores"][med["store_id"]-1]["distance"],
                "open": centralized_db["stores"][med["store_id"]-1]["open"],
                "expiry": med["expiry"],
                "usage": med["usage"]
            }
            for med in st.session_state.medicines
            if med["name"].lower() == search_query.lower()
        ]
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df[results_df["open"] == True]
            st.write(results_df.nsmallest(4, "distance"))
        else:
            st.error("No stores found with the specified medicine.")

    # Billing Process
    st.header("Billing")
    available_meds = [med["name"] for med in st.session_state.medicines if med["store_id"] == store_id]
    
    if not available_meds:
        st.warning("No medicines available for billing.")
        return

    med_name = st.selectbox("Select Medicine", options=available_meds)
    med_details = next(med for med in st.session_state.medicines if med["name"] == med_name and med["store_id"] == store_id)
    
    st.write(f"**Usage:** {med_details['usage']}")  # Medicine description on selection
    quantity = st.number_input("Quantity", min_value=1, max_value=med_details["quantity"])

    if st.button("Bill Medicine"):
        for med in st.session_state.medicines:
            if med["name"] == med_name and med["store_id"] == store_id:
                if med["quantity"] >= quantity:
                    med["quantity"] -= quantity
                    st.success(f"Billed {quantity} units of {med_name}. Updated inventory!")
                else:
                    st.error("Insufficient quantity available!")

    # Back to Inventory
    if st.button("Back to Inventory"):
        st.session_state.page = "Inventory"
        st.rerun()

# Page Navigation
if st.session_state.page == "Login":
    login_page()
elif st.session_state.page == "Inventory":
    inventory_page()
elif st.session_state.page == "Billing":
    billing_page()
