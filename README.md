## 🎵 Spotify Track Recommendation using Machine Learning

This repository contains a Python-based machine learning project that predicts and recommends musical tracks based on acoustic similarities such as danceability, energy, loudness, and tempo.

The project uses a comprehensive Spotify tracks dataset containing over 114,000 songs and implements an unsupervised learning workflow in a Jupyter Notebook.

---

## 📊 Project Workflow & Code Structure

The project follows a structured data science and recommendation pipeline:

1. **Data Exploration & Cleaning:**
   - Loaded and inspected the dataset using `pandas`.
   - Dropped arbitrary identifier indices (`Unnamed: 0`, `Unnamed: 0.1`) to keep the dataset focused entirely on track metadata and features.
2. **Feature Selection & Scaling:**
   - Isolated 9 core numerical audio metrics: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, and `tempo`.
   - Applied `StandardScaler` to normalize the features, ensuring that columns with large numerical ranges (like tempo) do not disproportionately bias the distance calculations.
3. **Model Construction:**
   - Initialized an unsupervised `NearestNeighbors` model configured to look for the top 6 closest neighbors.
   - Set the similarity metric to `cosine` distance to evaluate tracks based on the proportional balance and "vibe" of their audio features rather than sheer spatial distance.
4. **Recommendation Inference Engine:**
   - Built a dynamic `recommend_song()` function that takes a track name from user input, locates its vector, and extracts the top 5 most sonically matching track suggestions.

---

## 📈 Results & Example Output

- **Model Used:** Nearest Neighbors (with Cosine Metric)
- **Recommendation Pool:** Top 5 similar tracks per query

**Sample Query Output:**

Recommendations for: shape of you
----------------------------------------
Reno Ride | Clarence White | Genre: bluegrass
Always Will | Steve Martin;Steep Canyon Rangers | Genre: bluegrass
Minuet in F, K.1d | Wolfgang Amadeus Mozart;Erik Smith | Genre: classical
Minuet in F, K.1d | Wolfgang Amadeus Mozart;Erik Smith | Genre: classical
Les Indes Galantes - Air pour les esclaves | Jean-Philippe Rameau | Genre: opera

## 🛠️ Tech Stack & Libraries Used
The following Python libraries are required to run this project:

Data Manipulation: pandas, numpy

Machine Learning: scikit-learn

StandardScaler (Preprocessing & Normalization)

NearestNeighbors (Unsupervised Clustering/Distance Mapping)

cosine_similarity (Pairwise Metric Evaluations)

## 📦 How to Run the Project
 1- Clone this repository to your computer.
 2- Ensure you have Jupyter Notebook or VS Code installed.
 3- Install the required packages via your terminal if you haven't already:
      Bash
      pip install pandas numpy scikit-learn
 4- Configure Dataset Path:
     Open Spotify recommendation.ipynb.
     update the dataset file path to your local directory containing the csv file.
 5- Execute the Code:
     Launch your notebook interface using jupyter notebook in your terminal.
     Run the cells sequentially to generate your own music recommendations!
