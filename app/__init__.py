from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev'

    # Import main directly instead of using blueprint
    from . import main
    app = main.app

    return app