import os

# Define the project structure
project_name = "flask_app"
folders = [
    f"{project_name}/app/templates",
    f"{project_name}/app/static",
]
files = {
    f"{project_name}/app/__init__.py": """from flask import Flask

def create_app():
    app = Flask(__name__)

    from .routes import main
    app.register_blueprint(main)

    return app
""",
    f"{project_name}/app/routes.py": """from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')
""",
    f"{project_name}/app/templates/index.html": """<!doctype html>
<html>
<head><title>Flask App</title></head>
<body>
    <h1>Hello from Flask Template!</h1>
</body>
</html>
""",
    f"{project_name}/run.py": """from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
""",
    f"{project_name}/requirements.txt": "Flask==2.3.2\n",
    f"{project_name}/README.md": "# Flask App\n\nBasic Flask application scaffolded using Python script.\n"
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for filepath, content in files.items():
    with open(filepath, 'w') as f:
        f.write(content)

print(f"✅ Flask project '{project_name}' created successfully.")
