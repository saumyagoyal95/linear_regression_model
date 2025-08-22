# Linear Regression Model

This repository is designed for practicing Machine Learning Engineering concepts. It includes training scripts, deployment configurations, and a complete setup for frontend and backend deployment. The project demonstrates the end-to-end workflow of building, training, and deploying a machine learning model.

## Project Structure

- **Training**: Contains scripts and data for training the linear regression model.
- **Backend**: Includes FastAPI-based backend for serving the model.
- **Frontend**: Streamlit-based frontend for user interaction.
- **Deployment**: Docker and docker-compose configurations for containerized deployment.

## Features

- **Training**: Train a linear regression model using the provided dataset.
- **Backend**: Serve the trained model via REST API.
- **Frontend**: User-friendly interface for interacting with the model.
- **Deployment**: Containerized setup for easy deployment.

## Setup Instructions

### Prerequisites

- Docker
- Python 3.8 or higher
- `make` utility

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/saumyagoyal95/linear_regression_model.git
   cd linear_regression_model
   ```

2. Create the environment and install dependencies:
   ```bash
   make setup
   ```

3. Build and run the Docker containers:
   ```bash
   make run
   ```

4. Access the application:
   - Frontend: `http://localhost:8501`
   - Backend API: `http://localhost:8000`

## Best Practices

- Follow the `TODO` file for pending tasks.
- Use version control effectively.
- Ensure proper testing before deployment.

---

**License**: This project is licensed under the MIT License.

**Contributors**: Contributions are welcome! Feel free to open issues or submit pull requests.

**Contact**: For any queries, contact [Saumya Goyal](mailto:saumya@example.com).
