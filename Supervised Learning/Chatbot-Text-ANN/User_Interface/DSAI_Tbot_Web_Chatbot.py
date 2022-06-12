'''
Disclaimer:

DeepSphere.AI developed these materials based on its team’s expertise and technical infrastructure, and we are sharing these materials strictly for learning and research.
These learning resources may not work on other learning infrastructures and DeepSphere.AI advises the learners to use these materials at their own risk. As needed, we will
be changing these materials without any notification and we have full ownership and accountability to make any change to these materials.

Author :                          Chief Architect :       Reviewer :
____________________________________________________________________________
Avinash R & Jothi Periasamy       Jothi Periasamy         Jothi Periasamy
'''

from flask import Flask, render_template, request
from DSAI_Tbot_Utility import *

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("DSAI_Tbot_Index.html")

@app.route('/get')
def chatbot_response():
    msg = request.args.get('msg')
    bag = preprocess(msg, words)
    ints = predict_class(bag, classes, model)
    res = getResponse(ints, intents)
    return res

if __name__ == "__main__":
    model, intents, words, classes = load_dependencies()
    app.run(debug=True)

'''
Copyright Notice:

Local and international copyright laws protect this material. Repurposing or reproducing this material without written approval from DeepSphere.AI violates the law.

(c) DeepSphere.AI
'''