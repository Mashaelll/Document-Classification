from flask import Flask, request, jsonify
import subprocess
import tempfile
import os

app = Flask(__name__)

# Paths to the WEKA jar and trained model (set these as environment variables for flexibility)
WEKA_JAR_PATH = os.getenv('WEKA_JAR_PATH', 'C:\\Users\\iimas\\Downloads\\weka-3-8-0-monolithic.jar')
MODEL_PATH = os.getenv('MODEL_PATH', 'C:\\Users\\iimas\\Downloads\\UI\\Randomforest.model')

def classify_article(article_text):
    # Use tempfile to create a temporary ARFF file
    with tempfile.NamedTemporaryFile(suffix=".arff", delete=False, mode='w', encoding='utf-8') as temp_arff:
        arff_file_path = temp_arff.name
        # Write ARFF header and data format
        temp_arff.write("@RELATION article_class\n")
        temp_arff.write("@ATTRIBUTE text STRING\n")
        temp_arff.write("@ATTRIBUTE class {religion, sport, economy}\n")
        temp_arff.write("@DATA\n")
        temp_arff.write(f"\"{article_text}\",?\n")  # Add the article text

    # Command to run Weka with the classifier and predict the class
    command = [
        'java', '-cp', WEKA_JAR_PATH,
        'weka.classifiers.Classifier',
        '-l', MODEL_PATH,
        '-T', arff_file_path,
        '-p', '0'
    ]

    try:
        # Execute the command and capture output
        result = subprocess.check_output(command, stderr=subprocess.STDOUT)
        result_str = result.decode().strip()
        
        # Extract predicted class from the output (assuming the class is the last column in the result)
        # Parsing logic will depend on WEKA output structure
        for line in result_str.splitlines():
            if line.strip().startswith("inst#"):
                # Example output line: " 0   1:sport (1)"
                predicted_class = line.split()[-1]  # Extract class from the last word
                return predicted_class

        return "Prediction error: could not extract result"
    except subprocess.CalledProcessError as e:
        return f"Error occurred: {e.output.decode()}"
    finally:
        # Clean up the temporary ARFF file
        if os.path.exists(arff_file_path):
            os.remove(arff_file_path)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            h1 {
                color: #333;
            }
            form {
                margin-bottom: 20px;
            }
            textarea {
                width: 100%;
                margin-bottom: 10px;
            }
            button {
                padding: 10px 15px;
            }
        </style>
    </head>
    <body>
        <h1>Article Classifier</h1>
        <form id="prediction-form">
            <textarea name="article_text" rows="10" cols="50" placeholder="Enter your article here..."></textarea><br>
            <button type="submit">Predict Category</button>
        </form>
        <h3 id="prediction-result"></h3> <!-- Result displayed here -->
        <script>
            document.getElementById('prediction-form').addEventListener('submit', async function (event) {
                event.preventDefault();
                const formData = new FormData(this);
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                });
                const data = await response.json();
                document.getElementById('prediction-result').innerText = `Predicted Category: ${data.prediction}`;
            });
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    article = request.form.get('article_text')
    if article:
        prediction = classify_article(article)
        return jsonify({'prediction': prediction})
    return jsonify({'prediction': "No article provided"})

if __name__ == '__main__':
    app.run(debug=True)
