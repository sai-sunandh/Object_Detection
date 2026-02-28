from flask import Flask
from routes.api import api_bp

def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(api_bp)

    # Additional configurations can be added here

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)