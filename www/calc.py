from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/calc')
def app_main():
    return render_template('form.html')

@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a + b)

if __name__ == '__main__':
    app.run()