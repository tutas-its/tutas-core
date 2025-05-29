import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========== PARAMETER ==========
MAX_CIRCLE_SIZE = 5
INPUT_FILE = "data/dummy/dummy_circle_input.csv"
OUTPUT_CIRCLE = "data/output/circle_output_2.csv"
OUTPUT_UNMATCHED = "data/output/unmatched.csv"

# ========== STEP 1: LOAD DATA ==========
df = pd.read_csv(INPUT_FILE)
df.fillna("", inplace=True)

# Filter data valid
valid_mask = df["Topik"] != ""
df_valid = df[valid_mask].reset_index(drop=True)
df_invalid = df[~valid_mask].reset_index(drop=True)
print(f"✅ Data valid: {len(df_valid)} | ❌ Data tidak lengkap: {len(df_invalid)}")

# ========== STEP 2: BANGUN GRAF SIMILARITY ==========
G = nx.Graph()

# Preprocess fitur sebagai string gabungan
def feature_string(row):
    return " ".join([
        row["Topik"].lower(),
        row["Mode"].lower(),
        row["Gaya"].lower()
    ])

df_valid["features"] = df_valid.apply(feature_string, axis=1)

# TF-IDF + Cosine Similarity
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_valid["features"])
cosine_sim = cosine_similarity(tfidf_matrix)

# Tambahkan edge jika similarity > threshold
THRESHOLD = 0.3
for i, j in combinations(range(len(df_valid)), 2):
    sim = cosine_sim[i][j]
    if sim >= THRESHOLD:
        user_i = df_valid.loc[i, "Nama"]
        user_j = df_valid.loc[j, "Nama"]
        G.add_edge(user_i, user_j, weight=sim)

# Tambahkan semua node
for nama in df_valid["Nama"]:
    if nama not in G:
        G.add_node(nama)

# ========== STEP 3: LOUVAIN CLUSTERING ==========
try:
    import community as community_louvain
except ImportError:
    print("❌ ERROR: python-louvain belum terinstall. Install dengan: pip install python-louvain")
    exit()

partition = community_louvain.best_partition(G, weight='weight')

# Gabungkan hasil komunitas
df_valid["Circle ID"] = df_valid["Nama"].map(partition)

# ========== STEP 4: BATASI MAKSIMAL 5 PER CIRCLE ==========
circle_groups = df_valid.groupby("Circle ID")
circle_result = []
unmatched = []

new_circle_id = 0

for _, group in circle_groups:
    group = group.sample(frac=1, random_state=42)  # acak
    while len(group) > MAX_CIRCLE_SIZE:
        sub_group = group.iloc[:MAX_CIRCLE_SIZE]
        circle_result.append((new_circle_id, sub_group))
        group = group.iloc[MAX_CIRCLE_SIZE:]
        new_circle_id += 1
    if len(group) > 1:
        circle_result.append((new_circle_id, group))
        new_circle_id += 1
    else:
        unmatched.append(group.iloc[0])

# ========== STEP 5: SIMPAN HASIL ==========
output_rows = []
for cid, grp in circle_result:
    for _, row in grp.iterrows():
        output_rows.append({
            "Nama": row["Nama"],
            "Circle ID": cid,
            "Tutor/Murid": row["Status"],
            "No WA": row["No WA"]
        })

df_output = pd.DataFrame(output_rows)
df_output.to_csv(OUTPUT_CIRCLE, index=False)
print(f"✅ Disimpan: {OUTPUT_CIRCLE}")

# Simpan unmatched
if unmatched:
    df_unmatched = pd.DataFrame(unmatched)
    df_unmatched.to_csv(OUTPUT_UNMATCHED, index=False)
    print(f"⚠️ Unmatched disimpan ke: {OUTPUT_UNMATCHED}")
else:
    print("✅ Semua user mendapat Circle.")

