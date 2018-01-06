import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request




# webapp
app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()