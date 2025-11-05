# AI Chatbot with Sentiment Analysis Utility

A modern, interactive web-based chatbot built with Python Flask and powered by Mistral AI's free API. The chatbot features a unique **sentiment analysis utility** that analyzes the emotional tone of user messages before providing responses.

## Features

### Core Features
- **Interactive Web Interface**: Clean, modern chat interface with real-time messaging
- **Mistral AI Integration**: Uses the free Mistral AI API for natural language processing
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Auto-scrolling Chat**: Messages automatically scroll to the latest message

### Unique Utility: Sentiment Analysis
The chatbot includes a **unique sentiment analysis feature** that:
- Analyzes the sentiment of each user message (Positive, Negative, Neutral, or Mixed)
- Displays the sentiment analysis alongside the chatbot's response
- Helps users understand how their emotional tone is being perceived
- Provides context-aware responses based on the detected sentiment

## Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.8+**
- **pip** (Python package manager)
- **Git** (optional, for cloning the repository)

## Installation

### Step 1: Clone or Download the Project
```bash
git clone <repository-url>
cd chatbot_project
```

Or simply download and extract the project folder.

### Step 2: Create a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Get a Free Mistral AI API Key
1. Visit [Mistral AI Platform](https://mistral.ai/)
2. Sign up for a free account
3. Navigate to the API Keys section
4. Create a new API key
5. Copy the API key

### Step 5: Configure the API Key
1. Open the `.env` file in the project root
2. Replace `YOUR_MISTRAL_API_KEY_HERE` with your actual Mistral AI API key
3. Save the file

```env
MISTRAL_API_KEY="your_actual_api_key_here"
```

## Running the Application

### Start the Flask Server
```bash
python app.py
```

You should see output similar to:
```
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

### Access the Chatbot
Open your web browser and navigate to:
```
http://localhost:5000
```

The chatbot interface will load, and you can start chatting immediately!

## How It Works

### User Interaction Flow
1. User types a message in the input field
2. Message is sent to the Flask backend via AJAX
3. Backend sends the message to Mistral AI's API
4. Mistral AI analyzes the sentiment and generates a response
5. Response is formatted as JSON with sentiment and reply
6. Frontend displays both the sentiment analysis and the chatbot's response
7. User can continue the conversation

### Sentiment Analysis
The chatbot uses Mistral AI's language understanding to categorize messages into:
- **Positive**: Expressing happiness, satisfaction, or enthusiasm
- **Negative**: Expressing frustration, anger, or dissatisfaction
- **Neutral**: Factual or objective statements
- **Mixed**: Containing both positive and negative sentiments

## Project Structure

```
chatbot_project/
├── app.py                 # Main Flask application
├── .env                   # Environment variables (API key)
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── templates/
    └── index.html         # Web interface template
```

## Configuration

### Changing the Model
To use a different Mistral AI model, edit `app.py` and change the `MODEL_NAME` variable:

```python
MODEL_NAME = "mistral-small"  # or "mistral-medium", "mistral-large"
```

Available models:
- `mistral-tiny` (fastest, free tier)
- `mistral-small`
- `mistral-medium`
- `mistral-large`

### Customizing the System Prompt
Edit the `system_prompt` in the `get_chatbot_response()` function to change the chatbot's personality or behavior.

## Troubleshooting

### Issue: "MISTRAL_API_KEY not found"
**Solution**: Ensure the `.env` file exists in the project root and contains your API key.

### Issue: "Connection refused" or "Cannot connect to server"
**Solution**: Make sure the Flask server is running. Check that port 5000 is not in use by another application.

### Issue: Chatbot returns empty responses
**Solution**: Verify your Mistral AI API key is valid and has available credits/quota.

### Issue: JSON parsing error
**Solution**: This may occur if the API returns an unexpected format. Check your API key and internet connection.

## Customization Ideas

### Enhance the Chatbot
- Add conversation history/memory
- Implement user authentication
- Add support for file uploads
- Create custom commands
- Add voice input/output
- Implement conversation export

### Improve the UI
- Add dark mode toggle
- Implement typing indicators
- Add emoji support
- Create custom themes
- Add message timestamps

## API Rate Limits

Mistral AI's free tier includes:
- Generous request limits
- Suitable for development and testing
- Check [Mistral AI Pricing](https://mistral.ai/pricing) for current limits

## Security Considerations

- **Never commit your `.env` file** to version control
- Keep your API key confidential
- Use environment variables for sensitive data
- Validate user input on the backend
- Consider implementing rate limiting for production

## Deployment

For production deployment:
1. Use a production WSGI server (e.g., Gunicorn)
2. Set `debug=False` in Flask
3. Use a reverse proxy (e.g., Nginx)
4. Implement proper error handling and logging
5. Consider using a process manager (e.g., Supervisor)

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

This project is open-source and available under the MIT License.

## Support

For issues with:
- **Mistral AI API**: Visit [Mistral AI Documentation](https://docs.mistral.ai/)
- **Flask**: Visit [Flask Documentation](https://flask.palletsprojects.com/)
- **This Project**: Check the README or create an issue in the repository

## Contributing

Contributions are welcome! Feel free to fork the project and submit pull requests with improvements.

## Acknowledgments

- Built with [Flask](https://flask.palletsprojects.com/) - Python web framework
- Powered by [Mistral AI](https://mistral.ai/) - Free AI API
- Inspired by modern chatbot design principles

---

**Happy Chatting!** 🤖
