#  Movie Recommendation System

This project implements a **Movie Recommendation System** using various machine learning techniques to predict ratings and suggest movies. The system uses data from the **MovieLens 100K dataset** and includes different methods such as **Collaborative Filtering (SVD)**, **Content-Based Filtering (TF-IDF + Cosine Similarity)**, and **Hybrid Recommender Systems** that combine both approaches for more accurate predictions.

---

##  Project Status

This project is currently **a work in progress**. Features, optimizations, and improvements are being added regularly. Stay tuned for more updates!

---

##  Dataset

The **MovieLens 100K dataset** contains **100,000 movie ratings** from 943 users on 1682 movies, including:

- `u.data`: User ratings (user_id, movie_id, rating)
- `u.item`: Movie metadata (movie_id, title, genre)
- `u.user`: User demographics (user_id, age, gender, occupation)
- `u.genre`: Movie genres

You can find the dataset [here](https://grouplens.org/datasets/movielens/100k/).

---


##  Planned Features

- **Data Preprocessing**: Merging user, movie, and ratings data to create a usable dataset for the recommendation system.
- **Collaborative Filtering**:
  - **K-Nearest Neighbors (KNN)**: Find the most similar users or movies based on the collaborative filtering method.
- **Content-Based Filtering**:
  - **TF-IDF + Cosine Similarity**: Recommending movies based on movie features like genres, title, and description.
- **Hybrid Recommendation System**: Combining collaborative and content-based predictions to make better recommendations.
- **Evaluation**: Metrics such as **RMSE** (Root Mean Squared Error) and **Precision** are used to evaluate the model's performance.

---



## How to Run



#### 1. Clone the Repository

```bash
git clone https://github.com/vinidujay/Movie-Recommendations.git
cd Movie-Recommendations


#### 2. Set Up the Virtual Environment

``` bash
python3 -m venv venv
source venv/bin/activate #Mac/Linux
venv\Scripts\actvate #Windows

#### 3. Install Dependencies

```bash
pip install -r requirements.txt

#### 4. Run the Jupyter Notebook

```bash 
jupyter notebook
Open the relevant notebook (movies.ipynb) and run the cells step by step.


---

## Future Improvements

- Model Optimization: Fine-tune hyperparameters for better prediction accuracy (e.g., using grid search or randomized search for SVD and KNN).

- Real-Time Recommendations: Build a web app with real-time movie recommendations using Flask or Streamlit.

- Cold Start Problem: Incorporate additional data (e.g., movie plot summaries, tags) to improve recommendations for new users or movies.

- Neural Collaborative Filtering: Implement neural networks for better handling of non-linear relationships between users and items.

- Cross-Validation: Implement cross-validation techniques to evaluate the models more thoroughly.

---

## Author
[@vinidujay](https://github.com/vinidujay)

---

## Requirements

- Python 3.6+

- Jupyter Notebook

- Libraries: pandas, numpy, scikit-learn, surprise, matplotlib, seaborn, tensorflow, etc.


