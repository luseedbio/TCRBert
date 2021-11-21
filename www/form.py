from flask import Flask, render_template, request, json, jsonify

app = Flask(__name__)

@app.route('/form', methods=['GET', 'POST'])
def app_main():
    return render_template('form.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    results = {}

    results['user'] =  request.form['username'];
    results['pass'] = request.form['password'];
    return jsonify(results=results)
    # return json.dumps({'status': 'OK', 'user': user, 'pass': password});

if __name__ == '__main__':
    app.run()