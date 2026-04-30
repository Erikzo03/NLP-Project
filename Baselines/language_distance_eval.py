from lang2vec.lang2vec import get_features
from scipy.spatial.distance import cosine
import numpy as np

def get_vector(lang, feature_set="syntax_wals"):
    """Get the feature vector for a language, with missing values handled as np.nan."""
    features = get_features([lang], feature_set)
    vec = features[lang]
    return vec

def compute_distance(vec1, vec2):
    """Compute cosine distance between two language vectors, ignoring missing values."""
    v1 = np.array([float(x) if x != '--' else np.nan for x in vec1])
    v2 = np.array([float(x) if x != '--' else np.nan for x in vec2])
    mask = ~np.isnan(v1) & ~np.isnan(v2)
    v1_aligned = v1[mask]
    v2_aligned = v2[mask]
    if len(v1_aligned) == 0:
        return np.nan
    return cosine(v1_aligned, v2_aligned)

def compute_distance_matrix(languages, feature_set="syntax_wals"):
    """Compute a symmetric distance matrix for a list of languages."""
    vectors = {lang: get_vector(lang, feature_set) for lang in languages}
    n = len(languages)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0
            elif i < j:
                dist = compute_distance(vectors[languages[i]], vectors[languages[j]])
                matrix[i, j] = dist
                matrix[j, i] = dist  
    return matrix

def main():
    languages = ["eng", "dan", "nob"] 
    print(f"Using feature set: syntax_wals (WALS typological features)\n")

    dist_matrix = compute_distance_matrix(languages)

    print("=== Language Distance Matrix ===\n")
    header = "\t" + "\t".join(languages)
    print(header)
    for lang, row in zip(languages, dist_matrix):
        row_str = "\t".join(f"{x:.4f}" for x in row)
        print(f"{lang}\t{row_str}")

if __name__ == "__main__":
    main()