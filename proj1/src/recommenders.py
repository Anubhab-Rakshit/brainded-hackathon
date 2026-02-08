
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class BaseRecommender:
    def fit(self, train_df):
        pass
    
    def recommend(self, user_id, n=10):
        pass

class PopularityRecommender(BaseRecommender):
    def __init__(self):
        self.popular_items = None
    
    def fit(self, train_df):
        item_counts = train_df['movieId'].value_counts()
        self.popular_items = item_counts.index.tolist()
    
    def recommend(self, user_id, n=10):
        return self.popular_items[:n]

class CollaborativeRecommender(BaseRecommender):
    def __init__(self, method='user_user', n_neighbors=50):
        self.method = method
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None
        self.reverse_item_map = None
        self.similarity_matrix = None
        
    def fit(self, train_df):
        users = train_df['userId'].unique()
        items = train_df['movieId'].unique()
        
        self.user_map = {u: i for i, u in enumerate(users)}
        self.reverse_user_map = {i: u for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}
        self.reverse_item_map = {j: i for j, i in enumerate(items)}
        
        user_indices = train_df['userId'].map(self.user_map)
        item_indices = train_df['movieId'].map(self.item_map)
        ratings = train_df['rating']
        
        self.user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), 
                                           shape=(len(users), len(items)))
        
        if self.method == 'user_user':
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        elif self.method == 'item_item':
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            
    def recommend(self, user_id, n=10):
        if user_id not in self.user_map:
            return []
        
        u_idx = self.user_map[user_id]
        
        if self.method == 'user_user':
            sim_scores = self.similarity_matrix[u_idx]
            sim_scores[u_idx] = 0
            
            neighbor_indices = np.argsort(sim_scores)[::-1][:self.n_neighbors]
            
            neighbor_sims = sim_scores[neighbor_indices].reshape(-1, 1)
            neighbor_ratings = self.user_item_matrix[neighbor_indices].toarray()
            
            pred_scores = np.sum(neighbor_ratings * neighbor_sims, axis=0)
            
            top_item_indices = np.argsort(pred_scores)[::-1][:n]
            return [self.reverse_item_map.get(i) for i in top_item_indices]

        return []

class SVDRecommender(BaseRecommender):
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=n_components)
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, train_df):
        users = train_df['userId'].unique()
        items = train_df['movieId'].unique()
        
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}
        self.reverse_item_map = {j: i for j, i in enumerate(items)}
        
        user_indices = train_df['userId'].map(self.user_map)
        item_indices = train_df['movieId'].map(self.item_map)
        ratings = train_df['rating']
        
        matrix = csr_matrix((ratings, (user_indices, item_indices)), shape=(len(users), len(items)))
        
        self.model.fit(matrix)
        self.user_factors = self.model.transform(matrix) 
        self.item_factors = self.model.components_.T 
        
    def recommend(self, user_id, n=10):
        if user_id not in self.user_map:
            return []
        
        u_idx = self.user_map[user_id]
        user_vector = self.user_factors[u_idx]
        
        scores = np.dot(user_vector, self.item_factors.T)
        
        top_indices = np.argsort(scores)[::-1][:n]
        return [self.reverse_item_map.get(i) for i in top_indices]

class ContentRecommender(BaseRecommender):
    def __init__(self):
        self.item_profiles = None
        self.tfidf_matrix = None
        self.movie_indices = None
        self.cosine_sim = None
        
    def fit(self, train_df, movies_df, tags_df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
        
        movies_df['genres_str'] = movies_df['genres'].str.replace('|', ' ')
        
        tags_agg = tags_df.groupby('movieId')['tag'].agg(lambda x: ' '.join(str(v) for v in x)).reset_index()
        
        content_df = movies_df[['movieId', 'genres_str']].merge(tags_agg, on='movieId', how='left')
        content_df['tag'] = content_df['tag'].fillna('')
        content_df['content'] = content_df['genres_str'] + ' ' + content_df['tag']
        
        self.movie_indices = pd.Series(content_df.index, index=content_df['movieId'])
        
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(content_df['content'])
        
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        
    def recommend(self, user_id, user_rated_movie_ids, n=10):
        if not user_rated_movie_ids:
            return []
            
        valid_indices = [self.movie_indices[mid] for mid in user_rated_movie_ids if mid in self.movie_indices]
        
        if not valid_indices:
            return []
            
        user_sim_scores = self.cosine_sim[valid_indices].mean(axis=0)
        
        sim_scores = list(enumerate(user_sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        top_indices = [i[0] for i in sim_scores[:n]]
        
        idx_to_movie = self.movie_indices.index
        return [idx_to_movie[i] for i in top_indices]

class HybridRecommender(BaseRecommender):
    def __init__(self, svd_model, content_model, alpha=0.3):
        self.svd = svd_model
        self.content = content_model
        self.alpha = alpha
        self.user_history = {} 
        
    def fit(self, train_df, movies_df, tags_df):
        print("Training Hybrid: Fitting SVD...")
        self.svd.fit(train_df)
        print("Training Hybrid: Fitting Content...")
        self.content.fit(train_df, movies_df, tags_df)
        
        self.user_history = train_df.groupby('userId')['movieId'].apply(list).to_dict()
        
    def recommend(self, user_id, n=10):
        if user_id not in self.svd.user_map:
            return []
            
        u_idx = self.svd.user_map[user_id]
        user_vector = self.svd.user_factors[u_idx]
        svd_scores = np.dot(user_vector, self.svd.item_factors.T)
        
        min_s, max_s = svd_scores.min(), svd_scores.max()
        if max_s - min_s > 0:
            svd_scores = (svd_scores - min_s) / (max_s - min_s)
        
        top_indices = np.argsort(svd_scores)[::-1][:100]
        top_movie_ids = [self.svd.reverse_item_map[i] for i in top_indices]
        
        user_liked_movies = self.user_history.get(user_id, [])
        if not user_liked_movies:
            return self.svd.recommend(user_id, n) 
            
        liked_indices = [self.content.movie_indices[mid] for mid in user_liked_movies if mid in self.content.movie_indices]
        if not liked_indices:
             return self.svd.recommend(user_id, n)

        user_content_profile = np.asarray(self.content.tfidf_matrix[liked_indices].mean(axis=0))
        
        cand_content_indices = [self.content.movie_indices[mid] for mid in top_movie_ids if mid in self.content.movie_indices]
        
        if not cand_content_indices:
             return self.svd.recommend(user_id, n)
             
        from sklearn.metrics.pairwise import linear_kernel
        
        candidate_matrix = self.content.tfidf_matrix[cand_content_indices]
        content_scores = linear_kernel(user_content_profile, candidate_matrix).flatten()
        
        candidates_df = pd.DataFrame({
            'movieId': top_movie_ids,
            'svd_score': svd_scores[top_indices]
        })
        
        c_scores_map = {}
        valid_c_count = 0
        for mid in top_movie_ids:
            if mid in self.content.movie_indices:
                c_scores_map[mid] = content_scores[valid_c_count]
                valid_c_count += 1
            else:
                c_scores_map[mid] = 0.0
                
        candidates_df['content_score'] = candidates_df['movieId'].map(c_scores_map)
        
        candidates_df['final_score'] = (1 - self.alpha) * candidates_df['svd_score'] + self.alpha * candidates_df['content_score']
        
        recs = candidates_df.sort_values('final_score', ascending=False).head(n)
        return recs['movieId'].tolist()

class NeuralCFRecommender(BaseRecommender):
    def __init__(self, embedding_dim=32, n_epochs=5, batch_size=256, lr=0.001):
        self.embedding_dim = embedding_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None
        
    def fit(self, train_df):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        
        users = train_df['userId'].unique()
        items = train_df['movieId'].unique()
        
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}
        self.reverse_item_map = {j: i for j, i in enumerate(items)}
        
        class RatingDataset(Dataset):
            def __init__(self, df, user_map, item_map):
                self.users = torch.tensor(df['userId'].map(user_map).values, dtype=torch.long)
                self.items = torch.tensor(df['movieId'].map(item_map).values, dtype=torch.long)
                self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
                
            def __len__(self):
                return len(self.ratings)
                
            def __getitem__(self, idx):
                return self.users[idx], self.items[idx], self.ratings[idx]
                
        class NeuMF(nn.Module):
            def __init__(self, n_users, n_items, embedding_dim):
                super(NeuMF, self).__init__()
                
                self.gmf_user_embedding = nn.Embedding(n_users, embedding_dim)
                self.gmf_item_embedding = nn.Embedding(n_items, embedding_dim)
                
                self.mlp_user_embedding = nn.Embedding(n_users, embedding_dim)
                self.mlp_item_embedding = nn.Embedding(n_items, embedding_dim)
                
                self.fc1 = nn.Linear(embedding_dim * 2, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 16)
                
                self.output = nn.Linear(embedding_dim + 16, 1)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, user, item):
                u_gmf = self.gmf_user_embedding(user)
                i_gmf = self.gmf_item_embedding(item)
                x_gmf = u_gmf * i_gmf
                
                u_mlp = self.mlp_user_embedding(user)
                i_mlp = self.mlp_item_embedding(item)
                x_mlp = torch.cat([u_mlp, i_mlp], dim=1)
                x_mlp = torch.relu(self.fc1(x_mlp))
                x_mlp = self.dropout(x_mlp)
                x_mlp = torch.relu(self.fc2(x_mlp))
                x_mlp = self.dropout(x_mlp)
                x_mlp = torch.relu(self.fc3(x_mlp))
                
                x = torch.cat([x_gmf, x_mlp], dim=1)
                x = self.output(x)
                return x.squeeze()
        
        n_users = len(users)
        n_items = len(items)
        
        dataset = RatingDataset(train_df, self.user_map, self.item_map)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = NeuMF(n_users, n_items, self.embedding_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        self.n_epochs = 20 
        print(f"Training Neural CF for {self.n_epochs} epochs...")
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for u, i, r in loader:
                optimizer.zero_grad()
                pred = self.model(u, i)
                loss = criterion(pred, r)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {total_loss/len(loader):.4f}")
            
    def recommend(self, user_id, n=10):
        import torch
        
        if self.model is None:
            return []
            
        if user_id not in self.user_map:
            return []
            
        u_idx = self.user_map[user_id]
        
        self.model.eval()
        with torch.no_grad():
            all_item_indices = torch.arange(len(self.item_map), dtype=torch.long)
            user_indices = torch.full((len(self.item_map),), u_idx, dtype=torch.long)
            
            predictions = self.model(user_indices, all_item_indices)
            
            _, top_indices_tensor = torch.topk(predictions, n)
            top_indices = top_indices_tensor.numpy()
            
            return [self.reverse_item_map.get(i) for i in top_indices]

class DiversityRecommender(BaseRecommender):
    def __init__(self, base_recommender, lambda_param=0.5):
        self.base_recommender = base_recommender
        self.lambda_param = lambda_param 
        self.item_genres = {}
        
    def fit(self, train_df, movies_df, **kwargs):
        self.base_recommender.fit(train_df, movies_df, **kwargs)
        for idx, row in movies_df.iterrows():
            self.item_genres[row['movieId']] = set(row['genres'].split('|'))
            
    def recommend(self, user_id, n=10):
        candidates = self.base_recommender.recommend(user_id, n=n*3)
        if not candidates:
            return []
            
        selected = []
        candidates_set = set(candidates)
        
        while len(selected) < n and candidates_set:
            best_item = None
            best_score = -float('inf')
            
            for item in candidates_set:
                rank = candidates.index(item) + 1
                relevance = 1.0 / rank
                
                max_sim = 0.0
                if selected:
                    item_genes = self.item_genres.get(item, set())
                    for s_item in selected:
                        s_genres = self.item_genres.get(s_item, set())
                        u = len(item_genes.union(s_genres))
                        i = len(item_genes.intersection(s_genres))
                        sim = i / u if u > 0 else 0
                        max_sim = max(max_sim, sim)
                
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = item
            
            if best_item:
                selected.append(best_item)
                candidates_set.remove(best_item)
                
        return selected

class EnsembleRecommender(BaseRecommender):
    def __init__(self, models_dict, weights=None):
        self.models = models_dict 
        self.weights = weights if weights else {k: 1.0/len(models_dict) for k in models_dict}
        
    def fit(self, train_df, *args, **kwargs):
        pass
            
    def recommend(self, user_id, n=10):
        candidates = set()
        for name, model in self.models.items():
            recs = model.recommend(user_id, n=n*3)
            candidates.update(recs)
            
        candidate_list = list(candidates)
        if not candidate_list:
            return []
            
        final_scores = {mid: 0.0 for mid in candidate_list}
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 0.0)
            
            recs = model.recommend(user_id, n=len(candidate_list)*2) 
            
            for rank, mid in enumerate(recs):
                if mid in final_scores:
                    score = 1.0 / (rank + 1)
                    final_scores[mid] += score * weight
                    
        sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_candidates[:n]]
 
class LightGCNRecommender(BaseRecommender):
    def __init__(self, embedding_dim=64, n_layers=3, n_epochs=20, batch_size=1024, lr=0.001):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None
        self.adj_matrix = None
        
    def fit(self, train_df):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        import scipy.sparse as sp
        
        users = train_df['userId'].unique()
        items = train_df['movieId'].unique()
        n_users = len(users)
        n_items = len(items)
        
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}
        self.reverse_item_map = {j: i for j, i in enumerate(items)}
        
        user_idx = train_df['userId'].map(self.user_map).values
        item_idx = train_df['movieId'].map(self.item_map).values
        
        R = sp.coo_matrix((np.ones(len(user_idx)), (user_idx, item_idx)), shape=(n_users, n_items))
        
        top_left = sp.csr_matrix((n_users, n_users))
        bottom_right = sp.csr_matrix((n_items, n_items))
        
        adj_mat = sp.vstack([sp.hstack([top_left, R]), sp.hstack([R.T, bottom_right])])
        
        rowsum = np.array(adj_mat.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        
        coo = norm_adj.tocoo()
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        self.adj_matrix = torch.sparse_coo_tensor(indices, values, shape)
        
        class LightGCN(nn.Module):
            def __init__(self, n_users, n_items, embedding_dim, n_layers, adj_matrix):
                super(LightGCN, self).__init__()
                self.n_users = n_users
                self.n_items = n_items
                self.embedding_dim = embedding_dim
                self.n_layers = n_layers
                self.adj_matrix = adj_matrix
                
                self.user_embedding = nn.Embedding(n_users, embedding_dim)
                self.item_embedding = nn.Embedding(n_items, embedding_dim)
                
                nn.init.normal_(self.user_embedding.weight, std=0.1)
                nn.init.normal_(self.item_embedding.weight, std=0.1)
                
            def forward(self):
                all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
                embs = [all_emb]
                
                for i in range(self.n_layers):
                    all_emb = torch.sparse.mm(self.adj_matrix, all_emb)
                    embs.append(all_emb)
                    
                embs = torch.stack(embs, dim=1)
                final_embs = torch.mean(embs, dim=1)
                
                users_emb, items_emb = torch.split(final_embs, [self.n_users, self.n_items])
                return users_emb, items_emb
                
            def get_rating(self, user_indices, item_indices, u_emb, i_emb):
                return (u_emb[user_indices] * i_emb[item_indices]).sum(1)
        
        self.model = LightGCN(n_users, n_items, self.embedding_dim, self.n_layers, self.adj_matrix)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        train_data = np.vstack([user_idx, item_idx]).T
        
        self.n_epochs = 30 
        print(f"Training LightGCN for {self.n_epochs} epochs...")
        
        self.model.train()
        for epoch in range(self.n_epochs):
            neg_items = np.random.randint(0, n_items, size=len(train_data))
            
            u_batch = torch.LongTensor(train_data[:, 0])
            pos_i_batch = torch.LongTensor(train_data[:, 1])
            neg_i_batch = torch.LongTensor(neg_items)
            
            optimizer.zero_grad()
            
            u_emb, i_emb = self.model()
            
            pos_scores = (u_emb[u_batch] * i_emb[pos_i_batch]).sum(1)
            neg_scores = (u_emb[u_batch] * i_emb[neg_i_batch]).sum(1)
            
            pos_loss = criterion(pos_scores, torch.ones_like(pos_scores))
            neg_loss = criterion(neg_scores, torch.zeros_like(neg_scores))
            
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {loss.item():.4f}")

    def recommend(self, user_id, n=10):
        import torch
        if self.model is None or user_id not in self.user_map:
            return []
            
        u_idx = self.user_map[user_id]
        
        self.model.eval()
        with torch.no_grad():
            u_emb, i_emb = self.model()
            user_vec = u_emb[u_idx] 
            
            scores = torch.matmul(i_emb, user_vec) 
            
            _, top_indices = torch.topk(scores, n)
            return [self.reverse_item_map.get(i.item()) for i in top_indices]

class SASRecRecommender(BaseRecommender):
    def __init__(self, embedding_dim=64, n_heads=2, n_blocks=2, max_seq_len=50, dropout=0.1, lr=0.001, n_epochs=50, batch_size=128):
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.model = None
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None
        self.user_history = {} 
        self.item_genre_matrix = None 
        
    def fit(self, train_df, movies_df=None, *args, **kwargs):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Training SASRec on: {device}")
        
        df_sorted = train_df.sort_values(by=['userId', 'timestamp'])
        
        users = df_sorted['userId'].unique()
        items = df_sorted['movieId'].unique()
        n_items = len(items)
        
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {item: i+1 for i, item in enumerate(items)}
        self.reverse_item_map = {i+1: item for i, item in enumerate(items)}
        self.n_items_vocab = n_items + 1
        
        if movies_df is not None:
            all_genres = set()
            for g_str in movies_df['genres']:
                for g in g_str.split('|'):
                    all_genres.add(g)
            
            genre_map = {g: i for i, g in enumerate(list(all_genres))}
            n_genres = len(genre_map)
            
            self.item_genre_matrix = torch.zeros((self.n_items_vocab, n_genres), device=device)
            
            for mid in items: 
                if mid in self.item_map:
                    idx = self.item_map[mid]
                    g_str = movies_df[movies_df['movieId'] == mid]['genres'].values
                    if len(g_str) > 0:
                        gs = g_str[0].split('|')
                        for g in gs:
                            if g in genre_map:
                                self.item_genre_matrix[idx, genre_map[g]] = 1.0
        else:
            n_genres = 0
            
        self.user_history = df_sorted.groupby('userId')['movieId'].apply(list).to_dict()
        
        train_sequences = []
        train_targets = []
        
        for u, history in self.user_history.items():
            seq = [self.item_map[i] for i in history if i in self.item_map]
            for i in range(1, len(seq)):
                t = seq[i]
                s = seq[:i]
                if len(s) > self.max_seq_len:
                    s = s[-self.max_seq_len:]
                
                pad_len = self.max_seq_len - len(s)
                s = [0] * pad_len + s
                
                train_sequences.append(s)
                train_targets.append(t)
                
        class SASRecDataset(Dataset):
            def __init__(self, seqs, targets):
                self.seqs = torch.LongTensor(seqs)
                self.targets = torch.LongTensor(targets)
                
            def __len__(self):
                return len(self.seqs)
                
            def __getitem__(self, idx):
                return self.seqs[idx], self.targets[idx]
                
        dataset = SASRecDataset(train_sequences, train_targets)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        class SASRec(nn.Module):
            def __init__(self, n_items, embed_dim, n_heads, n_blocks, max_len, dropout, genre_matrix=None):
                super(SASRec, self).__init__()
                self.item_embedding = nn.Embedding(n_items, embed_dim, padding_idx=0)
                self.pos_embedding = nn.Embedding(max_len, embed_dim)
                self.dropout = nn.Dropout(dropout)
                
                self.genre_matrix = genre_matrix
                if genre_matrix is not None:
                    # Projection for genres to embed_dim
                    self.genre_proj = nn.Linear(genre_matrix.shape[1], embed_dim)
                
                encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dropout=dropout, batch_first=True)
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
                
                self.output_layer = nn.Linear(embed_dim, n_items)
                
            def forward(self, x):
                seq_len = x.size(1)
                pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
                
                emb = self.item_embedding(x)
                
                # Context-Aware Augmentation
                if self.genre_matrix is not None:
                     # Gather genre features for items in sequence
                     # x is [B, Seq]
                     # genre_matrix is [Vocab, GenreDim]
                     genres = torch.index_select(self.genre_matrix, 0, x.view(-1)).view(x.shape[0], x.shape[1], -1)
                     genre_emb = self.genre_proj(genres)
                     emb = emb + genre_emb
                
                emb = emb + self.pos_embedding(pos)
                emb = self.dropout(emb)
                
                mask = (x == 0)
                
                feat = self.encoder(emb, src_key_padding_mask=mask)
                
                # We want to predict the next item for the last position?
                # Actually during training we predict next item for every position usually?
                # Simplified: Predict based on last state
                final_feat = feat[:, -1, :] 
                logits = self.output_layer(final_feat)
                return logits

        self.model = SASRec(self.n_items_vocab, self.embedding_dim, self.n_heads, self.n_blocks, self.max_seq_len, self.dropout, genre_matrix=self.item_genre_matrix).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        print(f"Training SASRec for {self.n_epochs} epochs...")
        for epoch in range(self.n_epochs):
            total_loss = 0
            for seq, target in loader:
                seq, target = seq.to(device), target.to(device)
                optimizer.zero_grad()
                logits = self.model(seq)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {total_loss/len(loader):.4f}")
                
    def recommend(self, user_id, n=10):
        import torch
        
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        if self.model is None or user_id not in self.user_history:
            return []
            
        history = self.user_history[user_id]
        seq = [self.item_map[i] for i in history if i in self.item_map]
        
        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(seq)
        seq = [0] * pad_len + seq
        
        seq_tensor = torch.LongTensor([seq]).to(device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(seq_tensor)
            # Logits is [1, n_items]
            probs = torch.softmax(logits, dim=1)
            
            _, top_indices = torch.topk(probs, n)
            return [self.reverse_item_map.get(i.item()) for i in top_indices[0]]
