import streamlit as st
from openai import OpenAI
import base64
from PIL import Image
import io
import re

# Directly set the API key
API_KEY = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=API_KEY)

class ReceiptChatbot:
    def __init__(self):
        self.receipt_data = None
        self.image = None

    def clean_text(self, text):
        """
        Clean up extracted text by removing asterisks, extra spaces, and normalizing formatting
        """
        # Remove asterisks and extra whitespace
        cleaned = re.sub(r'\*', '', text)
        # Remove excessive newlines and spaces
        cleaned = re.sub(r'\n+', '\n', cleaned)
        cleaned = re.sub(r' +', ' ', cleaned)
        return cleaned.strip()

    def extract_receipt_data(self, image):
        """
        Extract receipt information using OpenAI's Vision API
        """
        # Store the image for potential future use
        self.image = image

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract **comprehensive details** from this receipt. Include:\n1. Restaurant Name\n2. Address\n3. Date and Time\n4. Itemized List of Purchases\n5. Total Amount\n6. Tax\n7. Payment Method if visible and Identify the **Restaurant Name by looking for the **largest, most prominent text at the **top."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                        ]
                    }
                ],
                max_tokens=300
            )
            
            # Clean and store the response
            self.receipt_data = self.clean_text(response.choices[0].message.content)
            return self.receipt_data
        except Exception as e:
            return f"Error extracting receipt data: {str(e)}"

    def chat_with_receipt(self, query, chat_history):
        """
        Chatbot functionality to answer questions about the receipt
        """
        if not self.receipt_data:
            return "No receipt data available. Please upload a receipt first."

        try:
            # Combine chat history with current context
            messages = [
                {"role": "system", "content": "You are a helpful assistant analyzing a restaurant receipt. Answer questions based strictly on the receipt information. Provide clear, concise responses."},
                {"role": "user", "content": f"Receipt Details:\n{self.receipt_data}"}
            ]
            
            # Add previous chat history
            messages.extend(chat_history)
            
            # Add current query
            messages.append({"role": "user", "content": query})

            chat_response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            return f"Error processing query: {str(e)}"

def main():
    st.title("🧾 Smart Receipt Assistant")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ReceiptChatbot()
        st.session_state.receipt_extracted = False
        st.session_state.chat_history = []
        st.session_state.extracted_details = ""

    # Image Upload
    uploaded_file = st.file_uploader("Upload Receipt Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Receipt', use_container_width=True)
        
        # Automatically extract receipt details
        if not st.session_state.receipt_extracted:
            with st.spinner('Extracting receipt details...'):
                receipt_details = st.session_state.chatbot.extract_receipt_data(image)
                st.session_state.receipt_extracted = True
                st.session_state.extracted_details = receipt_details
                st.success("Receipt details extracted successfully!")

    # Always display extracted details if available
    if st.session_state.extracted_details:
        st.header("Receipt Details")
        st.text_area("Extracted Information", 
                     st.session_state.extracted_details, 
                     height=200, 
                     disabled=True)

    # Chat Interface
    st.header("Chat about the Receipt")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            st.chat_message("assistant").write(message['content'])

    # User input
    user_query = st.chat_input("Ask a question about the receipt")
    
    # Chat functionality
    if user_query:
        # Validate receipt data
        if not st.session_state.extracted_details:
            st.error("Please upload a receipt first!")
            return

        # Add user query to chat history
        st.chat_message("user").write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Get response
        with st.spinner('Generating response...'):
            response = st.session_state.chatbot.chat_with_receipt(
                user_query, 
                st.session_state.chat_history
            )
            
            # Display and store assistant response
            st.chat_message("assistant").write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()