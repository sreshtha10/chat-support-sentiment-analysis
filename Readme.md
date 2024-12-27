# Chat Support Sentiment Analysis

This project aims to analyze the sentiment of chat support conversations. By leveraging natural language processing (NLP) techniques, we can determine whether the sentiment of a conversation is positive, negative, or neutral.

## Features

- Sentiment analysis of chat support conversations
- Visualization of sentiment trends over time
- Exportable reports for further analysis

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/sreshtha10/chat-support-sentiment-analysis.git
    ```
2. Navigate to the project directory:
    ```sh
    cd chat-support-sentiment-analysis
    ```
3. Create a `data` directory and download the dataset from [Kaggle](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset). Place the dataset inside the `data` directory.
    ```sh
    mkdir data
    # Download the dataset from Kaggle and place it inside the data directory
    ```
4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
5. Train the model by running `model.py`. The trained model will be saved in the `trained_model` directory.
    ```sh
    python model.py
    ```
6. Test the model by running `predict.py`.
    ```sh
    python predict.py
    ```
7. Run the Flask application by executing `app.py`.
    ```sh
    python app.py
    ```
    
## Usage

The Flask service provided by this project is designed to integrate with the chat-support-service, a Spring Boot application available at [Chat Support Service](https://github.com/sreshtha10/chat-support-service). Ensure both services are properly configured to work together for seamless sentiment analysis of chat conversations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
