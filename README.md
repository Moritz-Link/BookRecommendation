# BookRecommendation

This repository contains a small end-to-end book recommendation project that combines a machine learning model (graph neural network) with a web application for demonstration and interaction.

## Purpose

The goal of the project is to provide personalized book recommendations to a user based on their selection and historical ratings. It demonstrates how to train a recommender using graph-based deep learning and expose it through a simple Django web interface so recommendations can be browsed and tested by users.

## High-level architecture

- Data layer: Book Crossing dataset (users, books, ratings) is preprocessed using pandas and numpy.
- Model training: A Graph Neural Network (SAGE-style GNN) implemented in PyTorch is trained on a user-book-rating graph to learn embeddings and predict likely user-book interactions.
- Serving & UI: A Django app serves HTML templates and static assets and provides an endpoint that calls the trained model to generate recommendations for the selected book/user.

The project is split so model training and web serving are separate concerns: training is done offline (notebooks and scripts under the repo), and the trained artifacts can be loaded by the Django app at runtime for inference.

## Frameworks and libraries used

- PyTorch: model implementation and training (GNN / SAGE layers, training loop).
- pandas / numpy: data ingestion, cleaning, and preprocessing.
- Django: web framework used to build the frontend, routes, and model-serving endpoints.
- (Other common utilities): scikit-learn for metrics and preprocessing helpers, and typical Python packaging tools for environment management.

## Data

The project uses the Book-Crossing dataset (public dataset). CSV files and small preprocessing notebooks are included in `recommendation/data/` for extracting ISBN groupings and building the user-book-rating graph used by the GNN.
Data: The Book Crossing Dataset from 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip'

## Model

- Type: Graph Neural Network (GraphSAGE-style) trained to produce embeddings for users and books and to predict user-book interactions.
- Training artifacts and the training notebook are in the repository (`modelTraining.ipynb`). The notebooks demonstrate preprocessing, model architecture, training loop, and evaluation (precision/recall or ranking metrics).

## Web application

- Built with Django. The app lives in `recommendation/` and contains views, templates, and static assets to display a selected book and show recommended books.
- The Django app includes a small inference wrapper that loads the trained model (or a lightweight approximation) to generate recommendations on request.

## Results / Outcome

This project is primarily a demonstrator and research prototype. The main outcomes are:

- A working training notebook that builds and trains a SAGE-style GNN on the Book-Crossing dataset.
- A Django demo app that can serve recommendations for a chosen book and present them in a browser.
- Artifacts & code suitable as a starting point for further experimentation (improving the GNN, adding user sessions, or deploying the model as a microservice).

## Project contents (short)

- `modelTraining.ipynb` — Notebook that preprocesses data, defines and trains the GNN, and evaluates results.
- `recommendation/` — Django application with views, templates (`templates/`), static assets (`static/`), and data helper scripts.
- `recommendation/data/` — Raw and intermediate data and helper notebooks for preprocessing.
- `AppImage.png` / README image — Visual overview of the demo UI.

![alt text](https://github.com/Moritz-Link/BookRecommendation/blob/main/AppImage.png?raw=true)

## How to run (quick)

1. Create and activate a Python environment (Python 3.8+ recommended).
2. Install dependencies (minimal): pandas, numpy, torch, django, scikit-learn. Example (pip):

	python -m pip install pandas numpy torch django scikit-learn

3. Prepare data: run the preprocessing notebook or scripts in `recommendation/data/` to create the CSVs used by the training notebook and the Django app.
4. (Optional) Train the model using `modelTraining.ipynb` or load precomputed artifacts if available.
5. Run the Django dev server from `recommendation_project/` root:

	python manage.py runserver

6. Open the web UI (usually http://127.0.0.1:8000/) to browse books and view recommendations.

## Next steps / Improvements

- Add a serialized model checkpoint loader for fast startup of the Django app.
- Expose a REST API for programmatic recommendation requests.
- Add user authentication and per-user session tracking to provide personalized history-aware recommendations.

## License and credits

This repository is for research and demo purposes. The Book-Crossing dataset is used under its original terms. See individual files and notebooks for any additional acknowledgements.
