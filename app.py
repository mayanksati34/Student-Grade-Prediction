from flask import Flask,render_template,request

#-------------------------------------------------PYTHON CODE start--------------------------------------------

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import math
import seaborn as sns

style.use("ggplot")

# Import dataset with student's data
data = pd.read_csv("student-mat.csv", sep=";")






# Select the value we want to predict
predict = "G3"

# List the variables we want to use for our predictions in this model
data = data[["G1", "G2", "G3", "studytime", "health", "famrel", "failures", "absences"]]
data = shuffle(data)

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Train model multiple times to find the highest accuracy
best = 0
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
  #  print("Accuracy: " + str(acc))

    # Save the highest accuracy
    if (acc > best):
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)
#print("Highest Accuracy:", best)

# Load model
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)
'''''
print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")'''

predictions = linear.predict(x_test)

# Print the predictions, the variables we used and the actual final grade
#for x in range(len(predictions)):
 #  print("Predicted Final grade:", predictions[x], "Data:", x_test[x], "Final grade:", y_test[x])


anslist = []
pred = []
finalans = []
anslist = linear.coef_

#-------------------------------------------------PYTHON CODE END--------------------------------------------



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/graph")
def about():
    return render_template("graph.html")

'''
@app.route("/data")
def data():
    return render_template("data.html")'''

@app.route("/about")
def graph():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/form',methods=['POST'])
def getData():
    # g1,g2,studytime,health,famrel,failure,absences
    g1=int(request.form['g1'])
    g2 = int(request.form['g2'])
    studytime = int(request.form['studytime'])
    health=int(request.form['health'])
    famrel=int(request.form['famrel'])
    failures= int(request.form['failures'])
    absences=int(request.form['absences'])
    fname=(request.form['fname'])
    lname=(request.form['lname'])
    print(g1,g2,studytime,health,famrel,failures,absences)
    pred.clear()
    finalans.clear()

    tot = 0
    pred.append(g1)
    pred.append(g2)
    pred.append(studytime)
    pred.append(health)
    pred.append(famrel)
    pred.append(failures)
    pred.append(absences)
   # print(pred)
    for num1, num2 in zip(pred, anslist):
        finalans.append(num1 * num2)

    tot = linear.intercept_
    for i in finalans:
        tot += i
    print("Predicted Grade  = ", tot)
    if tot>=19:
        tot=19
    if tot<=0:
        tot=0

    #Test graph
    x = data['G3']
    plt.hist(data['G3'], bins=20, color='blue')
    plt.xlabel('Grade')
    plt.ylabel('Count')
    plt.title('Distribution of Final Grades')
    mask = (x >= math.floor(tot)) & (x <= math.ceil(tot))
    plt.hist(x[mask], bins=20, histtype='bar', color='red', lw=0)
    plt.savefig('static/answer.png')

    #plt.show()
    #test graph end
    return render_template('pass.html',g1=g1 ,g2=g2,tot=tot, fname=fname, lname=lname,url='/static/answer.png')

if __name__ == '__main__':

    app.run(debug=True)





#----------------------------------- TEST------------------------------------------

# g1,g2,studytime,health,famrel,failure,absences
