# Makefile

# Create the Python environment and install dependencies
setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r backend/requirements.txt && pip install -r frontend/requirements.txt

# Build and run the Docker containers
run:
	docker-compose up --build
