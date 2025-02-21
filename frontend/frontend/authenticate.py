import os
import streamlit as st
from streamlit import session_state as sst
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the root .env file
# We need to get the correct path by moving up from the current file location
root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / '.env')

def authenticate():
    """
    Handle authentication using the APP_PASSWORD from the root .env file.
    This function creates a clean login interface and manages the authentication state.
    """
    # Check if we're already authenticated
    if not sst.get("authenticated", False):
        # Get password from environment variable
        password = os.environ.get("APP_PASSWORD")

        if not password:
            st.error("No APP_PASSWORD found in environment variables. Please check your .env file.")
            st.stop()

        # Create a clean login interface
        login_container = st.container()
        with login_container:
            st.header("PixidaChat")
            password_placeholder = st.empty()
            entered_password = password_placeholder.text_input("Password", type="password")

            if entered_password != password:
                if entered_password != "":
                    st.error("Falsches Passwort! Bitte versuchen Sie es erneut.")
                st.stop()
            else:
                # Clean up the login interface after successful authentication
                password_placeholder.empty()
                sst["authenticated"] = True
                login_container.empty()