import cv2
import os
from flask import Flask, request, render_template
from datetime import date
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
from modules.database import create_tables
from config import SECRET_KEY
from modules.database import get_db_connection
from collections import Counter

app = Flask(__name__)
app.secret_key = SECRET_KEY 

# Initialize the database
create_tables()

nimgs = 20

imgBackground=cv2.imread("frame_background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    uniqueids = []
    userlistarray = []
    l = len(userlist)

    for i in userlist:
        name, uniqueid = i.split('_')
        names.append(name)
        uniqueids.append(uniqueid)
        userlistarray.append([name, uniqueid])

    return userlist, names, uniqueids, l, userlistarray


@app.route('/')
def home():
    return render_template('index.html', current_route="home")

@app.route('/about')
def about():
    return render_template('about.html', current_route="about")

@app.route('/add-user')
def addUser ():
    return render_template('adduser.html', current_route="add-user", totalreg=totalreg())

@app.route('/vote')
def vote():
    return render_template('vote.html', current_route="vote", identified="false")

@app.route('/results')
def results():
        # Fetch users and their votes from the database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT username, vote FROM users')  # Fetch usernames and votes
    user_votes = cursor.fetchall()
    conn.close()

    # Check if all users have voted
    non_voters = [user for user, vote in user_votes if not vote]  # Users with no votes

    if non_voters:
        # Return a message if not all users have voted
        return render_template(
            'results.html',
            current_route="results",
            message="All users must cast their votes before results can be displayed.",
            vote_counts={},
            winner=[],
        )

    # Separate valid votes
    votes = [vote for _, vote in user_votes]

    # Count the votes for each party
    from collections import Counter
    vote_counts = Counter(votes)

    # Determine the maximum vote count
    max_votes = max(vote_counts.values(), default=0)

    # Find all parties with the maximum vote count (handles ties)
    winner = [party for party, count in vote_counts.items() if count == max_votes]

    # Pass data to the results page
    return render_template(
        'results.html',
        current_route="results",
        message=None,  # No message needed as all users have voted
        vote_counts=dict(vote_counts),
        winner=winner,
        foundwinner="True"
    )

@app.route('/all-users')
def allusers():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT username, uniqueid, vote FROM users')
    userlist = cursor.fetchall()
    conn.close()

    userlistarray = [[user[0], user[1], user[2]] for user in userlist]
    return render_template('allusers.html', current_route="all-users", userlistarray=userlistarray)


@app.route('/start', methods=['GET'])
def start():

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('adduser.html', totalreg=totalreg(), message='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    person = 'Unknown'
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            person = identified_person
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, f'{identified_person}', (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.imshow('Vote', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if(person == 'Unknown'):
        return render_template('vote.html', current_route="vote", person='Unknown', identified = 'false', message='Face not recognized')
    
    username = person.split('_')[0]
    userid = person.split('_')[1]
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT vote FROM users WHERE uniqueid = ?
    ''', (userid,))
    vote_status = cursor.fetchone()
    conn.close()
    
    if vote_status and vote_status[0]:
        return render_template('vote.html', current_route="vote", username=username, userid=userid, identified='True',notVoted="False", message='You have already voted')

    
    return render_template('vote.html', current_route="vote", username=username, userid=userid, identified = 'True',notVoted="True", message='Face recognized You are ready to vote') 

@app.route('/register-vote', methods=['GET', 'POST'])
def register_vote():
    username = request.form['username']
    userid = request.form['userid']
    clicked_button = None

    if 'party1' in request.form:
        clicked_button = request.form['party1']
    elif 'party2' in request.form:
        clicked_button = request.form['party2']
    elif 'party3' in request.form:
        clicked_button = request.form['party3']
    elif 'party4' in request.form:
        clicked_button = request.form['party4']

    if not username or not userid:
        return render_template('vote.html', current_route="vote", identified="false", message="Please fill all the fields")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users SET vote = ? WHERE uniqueid = ?
    ''', (clicked_button, userid))
    conn.commit()
    conn.close()
    return render_template('vote.html', current_route="vote", identified="false", message="Vote Registered Successfully")

def add_user_to_database(username, userid):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO users (username, uniqueid, vote) VALUES (?, ?, ?)
    ''', (username, userid, ''))
    conn.commit()
    conn.close()

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['name']
    newuserid = request.form['id']
    if not newusername or not newuserid:
        return render_template("adduser.html", current_route="add-user", totalreg=totalreg(), message="Please fill all the fields")
    # print('Adding new User')/

    userlist = os.listdir('static/faces')
    for user in userlist:
        if user.split('_')[1] == str(newuserid):
            return render_template("adduser.html", current_route="add-user", totalreg=totalreg(), message="User already exists")

    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if (i+2) == nimgs:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    add_user_to_database(newusername, newuserid)
    print('Training Model')
    train_model()
    return render_template('adduser.html', totalreg=totalreg(), message="User Added Successfully")

if __name__ == '__main__':
    app.run(debug=True)
