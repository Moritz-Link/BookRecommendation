import pandas as pd
import numpy as np
import os

from torch_geometric.data import Data
import torch
import dill as pickle
import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F


        



class Data_load():
    def __init__(self) -> None:
        self.script_path = os.path.dirname(__file__)
        self.ratings =  pd.read_csv(os.path.join(self.script_path, f'data/BX-Book-Ratings.csv'), sep=';', encoding='latin-1')
        self.users = pd.read_csv(os.path.join(self.script_path, f'data/BX-Users.csv'), sep=';', encoding='latin-1')
        self.books = pd.read_csv(os.path.join(self.script_path, f'data/BX-Books.csv'), sep=';', encoding='latin-1', on_bad_lines="skip")
        self.books_filtered = None
        self.mapping_user = None
        self.mapping_item = None
        self.ratings_filtered_m = None
        self.grouped_books_rating =None
        self.books_images = None
        self.books_choice = []
        self.books_images_ISBN = None
        self.choice_dict = {}
        self.data = self.load_data_graph()
        print(self.data)
        self.gnn_loaded = self.load_model(self.data)
        #self.gnn_loaded = pickle.load(os.path.join(self.script_path, f'data/final_book_model.sav'), "rb") ,
        
        self.grouped = pd.read_csv(os.path.join(self.script_path, f'data/ISBNS_grouped.csv'))
     
    def load_model(self, data):
        # with open(os.path.join(self.script_path, f'data/final_book_model.sav'), "rb") as f:
        #     model = pickle.load(f) 
        class GNN(torch.nn.Module):
            def __init__(self, hidden_channels):
                super().__init__()
                self.conv1 = SAGEConv(hidden_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            def forward(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x
        class Classifier(torch.nn.Module):
            def forward(self, x_user, x_movie, edge_label_index):
                edge_feat_user = x_user[edge_label_index[0]]
                edge_feat_movie = x_movie[edge_label_index[1]]
                return (edge_feat_user * edge_feat_movie).sum(dim=-1)

        class Model(torch.nn.Module):
            def __init__(self, hidden_channels):
                super().__init__()
                self.movie_lin = torch.nn.Linear(8751, hidden_channels)
                self.user_emb = torch.nn.Embedding(47074, hidden_channels)# hier müsste es mehr sein!
                self.movie_emb = torch.nn.Embedding(98417, hidden_channels)
                self.gnn = GNN(hidden_channels)
                self.gnn = to_hetero(self.gnn, metadata=data.metadata())
                self.classifier = Classifier()
                
            def forward(self, data: HeteroData) :
            
                x_dict = {
                "user": self.user_emb(data["user"].node_id),
                "isbn": self.movie_lin(data["isbn"].x.float()) + self.movie_emb(data["isbn"].node_id),
                } 
                x_dict = self.gnn(x_dict, data.edge_index_dict)
                pred = self.classifier(
                    x_dict["user"],
                    x_dict["isbn"],
                    data["user", "review", "isbn"].edge_label_index,
                )
                return pred
        model = Model(64)
        #model.load_state_dict(torch.load(PATH, weights_only=True))
        model.load_state_dict(torch.load(os.path.join(self.script_path, f'data\\model_torch.pt'), weights_only=True))
        print("Hier wurde das MOdell schon geladen")
        print(model)
        model.eval()
        return model 
    def load_data_graph(self):
        #with open('data_graph.pkl', 'rb') as fp:
        with open(os.path.join(self.script_path, f'data/data_graph.pkl'), 'rb') as fp:
            person = pickle.load(fp)
            print('Graph Loaded from dictionary')
        return HeteroData(person)
        
    def preprocess_data(self):
        ratings_filtered = self.ratings.loc[self.ratings["Book-Rating"] >= 8]
        ratings_filtered = ratings_filtered.loc[ratings_filtered['ISBN'].isin(self.books['ISBN'].unique()) & ratings_filtered['User-ID'].isin(self.users['User-ID'].unique())]
        
        books_filtered = self.books.loc[self.books['ISBN'].isin(ratings_filtered['ISBN'].unique())]
        books_filtered["urlS"] = books_filtered["Image-URL-S"]
        books_filtered["urlM"] = books_filtered["Image-URL-M"]
        books_filtered["urlL"] = books_filtered["Image-URL-L"]
        self.books_filtered = books_filtered.set_index('ISBN')
        
        self.mapping_user = { user_id: index for index, user_id in enumerate(ratings_filtered["User-ID"].unique())}
        self.mapping_item = { isbn_id: index for index, isbn_id in enumerate(ratings_filtered["ISBN"].unique())}
    
        df_mapping_user = pd.DataFrame()
        df_mapping_user["user_id"] = self.mapping_user.keys()
        df_mapping_user["user_id_mapped"] = self.mapping_user.values()

        df_mapping_item = pd.DataFrame()
        df_mapping_item["isbn_id"] = self.mapping_item.keys()
        df_mapping_item["isbn_id_mapped"] = self.mapping_item.values()
        
        ratings_filtered_m = ratings_filtered.merge(df_mapping_user, left_on = "User-ID", right_on="user_id", how = "left")
        self.ratings_filtered_m = ratings_filtered_m.merge(df_mapping_item, left_on = "ISBN", right_on="isbn_id", how = "left")
        
        grouped = self.ratings_filtered_m.groupby(["isbn_id"]).mean(numeric_only=True)
        self.grouped_books_rating =  grouped.sort_values(['Book-Rating'], ascending=False)
        return None
    
    def load_books_images(self):
        isbn_list = self.grouped_books_rating.index.values[:50]
        self.books_images = self.books_filtered.loc[isbn_list][["urlS"]]
        self.books_images['isbn'] = self.books_images.index
        return self.books_images.copy()
    
    def add_book2choice(self,isbn ):
        self.books_choice.append(isbn)
        return None
    def add_choice2dict(self, isbn):
        self.choice_dict[isbn] = self.get_bookurl_by_isbn(isbn)
        return None
    
    def remove_choice2dict(self, isbn):
        del self.choice_dict[isbn]
        return None
    def reset_choice2dict(self):
        self.choice_dict = {}
        return None
    def get_dict_from_choice(self):
        isbn_l = list(self.choice_dict.keys())
        urls_l = list(self.choice_dict.values())
        l = [  {"isbn" :i, "urls": self.get_bookurl_by_isbn(i) }  for i in   isbn_l ]
        return l
    
    def reset_choice(self):
        self.books_choice = []
        return None
    def get_title_by_isbn(self, isbn):
        title = self.books_filtered.loc[isbn]["Book-Title"]
        return title
    def remove_choice(self,image_isbn):
        self.books_choice.remove(image_isbn)
        return None
    def get_bookurl_by_isbn(self, isbn):
        url = self.books_filtered.loc[isbn].urlS
        return str(url)
    def get_Lbookurl_by_isbn(self, isbn):
        url = self.books_filtered.loc[isbn].urlL
        return str(url)
    def get_author_by_isbn(self, isbn):
        author = self.books_filtered.loc[isbn]["Book-Author"]
        pub_year = self.books_filtered.loc[isbn]["Year-Of-Publication"]
        pub = self.books_filtered.loc[isbn]["Publisher"]
        return author, pub_year, pub
    def define_user(self):
        
        user_id = 9999
        isbn_l = self.books_choice
        
        
    def build_graph_for_new_user(self,user_isbn_selection_ids, user_id):
    
        user_id_a = np.full((len(user_isbn_selection_ids)),user_id)
        isbn_a = np.array(user_isbn_selection_ids)
        isbn_choice_a_to_new_user = np.vstack((isbn_a,user_id_a))
        safe = {}
        output_array = np.array([[],[]])
        for isbn in user_isbn_selection_ids:
            subset, edge_index_user_Isbn, mapping, edge_mask = torch_geometric.utils.k_hop_subgraph(
            node_idx = isbn,
            num_hops = 1,
            edge_index = self.data["user", "review", "isbn"].edge_index )  
            
            if len(edge_index_user_Isbn[0]) >= 5:
                choose_5_user_of_isbn = np.random.choice(edge_index_user_Isbn[0],size =5, replace=False)
            else:
                choose_5_user_of_isbn = np.random.choice(edge_index_user_Isbn[0],size =len(edge_index_user_Isbn[0]), replace=False)

            isbn_a = np.full((len(choose_5_user_of_isbn)),isbn )
            user_a = choose_5_user_of_isbn
    
            isbn_user_a = np.vstack((isbn_a,user_a ))
            safe[isbn] = isbn_user_a
            output_array = np.hstack((output_array,isbn_user_a ))
        output_array = np.hstack((output_array,isbn_choice_a_to_new_user )).astype(int)
        output_array[[0,1],:] = output_array[[1,0],:] 
        isbn_nodes = output_array[1]
        user_nodes = output_array[0]
        
        return output_array, isbn_nodes, user_nodes
    
    def build_neighbors_for_isbn(self,isbn_index):

        isbn_index = isbn_index.item()
        subset, edge_index_user_Isbn, mapping, edge_mask = torch_geometric.utils.k_hop_subgraph(
            node_idx = isbn_index,
            num_hops = 1,
            edge_index = self.data["user", "review", "isbn"].edge_index)
        if len(edge_index_user_Isbn[0]) >= 5:
            choose_5_user_of_isbn = np.random.choice(edge_index_user_Isbn[0],size =5, replace=False)
        else:
            choose_5_user_of_isbn = np.random.choice(edge_index_user_Isbn[0],size =len(edge_index_user_Isbn[0]), replace=False)

        safe = {}
        for user in choose_5_user_of_isbn:
            subset, edge_index_isbn_user, mapping, edge_mask = torch_geometric.utils.k_hop_subgraph( 
                                        node_idx = int(user), 
                                        num_hops = 1, 
                                        edge_index = self.data["isbn" , "rev_review", "user"].edge_index,)

            if len(edge_index_isbn_user[0]) >= 5:
                choose_5_isbn_of_user = np.random.choice(edge_index_isbn_user[0],size =5, replace=False)
            else:
                choose_5_isbn_of_user = np.random.choice(edge_index_isbn_user[0],size =len(edge_index_isbn_user[0]), replace=False)
            safe[user] = choose_5_isbn_of_user

        return safe


    def create_tensor_user_isbn(self,output_dict):
        output_array = np.array([[],[]])
        for k,v in output_dict.items():
        
            value_size = len(v)
            array_index = np.full(value_size, k)
            
            connect_array = np.vstack((array_index,v))
            output_array = np.hstack((output_array,connect_array))
        return output_array

    def create_tensor_isbn_user(self,output_dict, index_in):
        key_size = len(output_dict.keys())
        array_index = np.full(key_size, index_in)
        array_users = np.array(list(output_dict.keys()))
        connect_array = np.vstack((array_index,array_users))
        return connect_array

    def get_isbn_x(self,isbn_node_id):
        x_tensor = np.zeros((len(isbn_node_id), 8751))
        for node_i , node in enumerate(isbn_node_id) :
            node_t = self.data["isbn"].x[node]   #torch_isbn[node]
            x_tensor[node_i] = node_t
            

        return torch.tensor(x_tensor)
    
    
    def build_data_for_isbn_ls(self,isbn_ls):
        data_dict = {}
        for isbn_in in isbn_ls:
            index_in = isbn_in
            output_dict = self.build_neighbors_for_isbn(index_in)
            user_to_isbn = self.create_tensor_user_isbn(output_dict)
            isbn_to_user = self.create_tensor_isbn_user(output_dict, index_in)

            isbn_to_user[[0,1],:] = isbn_to_user[[1,0],:] 

            b = np.hstack((isbn_to_user ,user_to_isbn))
            edge_index_user_to_isbn_new = torch.tensor(b).type(torch.int64)
            edge_index_user_isbn_isbn_in  = edge_index_user_to_isbn_new
            data_dict[isbn_in] = edge_index_user_isbn_isbn_in
        number = 0
        
        for k,v in data_dict.items():
            if number == 0:
                t = v
            else:
                t = torch.hstack((t,v))
            number += 1
            
        user_node_id = list(set(t[0].tolist()))
        isbn_node_id = list(set(t[1].tolist()))
        isbn_x = self.get_isbn_x(isbn_node_id)
        return t,user_node_id,isbn_node_id, isbn_x, data_dict
    
    def alg_isbn(self,choice_l):
        df20_10 = self.grouped.loc[self.grouped["Book-Rating"] >= 9].isbn_id_mapped.values.astype(int)
        df20_10_random = list(np.random.choice(df20_10, 80, replace = False))
        
        df20_10_random = self.clear_list(choice_l,df20_10_random)
        
        
        recom_list = df20_10_random 
        return recom_list
    
    def clear_list(self,own, updated_list): 
        for o in own:
            for d in updated_list:
                if o == d:
                    updated_list.remove(o)
        return updated_list
    def run_recommendation(self, user_isbn_selection_ids,alg_chose_books ):
        user_isbn_selection_ids = user_isbn_selection_ids #[23461, 80205, 93285, 44055] #[0,8,70]
        alg_chose_books = alg_chose_books #[0,1,2]
        new_user_id = np.random.randint(1, high = 47074)


        new_user_to_isbn__edges, isbn_nodes_new_user, user_nodes_new_user =  self.build_graph_for_new_user(user_isbn_selection_ids,new_user_id )
        edge_index_user_to_isbn_new,user_node_id,isbn_node_id,isbn_x, data_dict = self.build_data_for_isbn_ls(alg_chose_books)

        ISBN_new_graph_nodes_id = list(set(isbn_nodes_new_user.tolist() + isbn_node_id))
        USER_new_graph_nodes_id = list(set(user_nodes_new_user.tolist() + user_node_id))
        ISBN_new_graph_nodes_id.sort()
        USER_new_graph_nodes_id.sort()


        ISBN_new_graph_isbn_x = self.get_isbn_x(ISBN_new_graph_nodes_id)
        ISBN_new_graph_edge_index_user_to_isbn_new = np.hstack((new_user_to_isbn__edges,edge_index_user_to_isbn_new))

        mapping_USER_new_graph_nodes_id = { user_id: index for index, user_id in enumerate(USER_new_graph_nodes_id)}
        mapping_ISBN_new_graph_nodes_id = { isbn_id: index for index, isbn_id in enumerate(ISBN_new_graph_nodes_id)}

        zero_a = np.zeros((ISBN_new_graph_edge_index_user_to_isbn_new.shape))
        zero_a[0] = [mapping_USER_new_graph_nodes_id[i] for index,i in enumerate(ISBN_new_graph_edge_index_user_to_isbn_new[0]) ]
        zero_a[1] = [mapping_ISBN_new_graph_nodes_id[i] for index,i in enumerate(ISBN_new_graph_edge_index_user_to_isbn_new[1]) ]
        ISBN_new_graph_edge_index_user_to_isbn_new = zero_a

        new_user_alg_chosen_books_edge_label_index = np.vstack((np.full((len(alg_chose_books)),new_user_id),np.array(alg_chose_books)))
        new_user_alg_chosen_books_edge_label_index[0] = [mapping_USER_new_graph_nodes_id[i] for index,i in enumerate(np.full((len(alg_chose_books)),new_user_id)) ]
        new_user_alg_chosen_books_edge_label_index[1] = [mapping_ISBN_new_graph_nodes_id[i] for index,i in enumerate(alg_chose_books) ]

        

        graph_new = HeteroData()
        graph_new["user"].node_id = torch.tensor(USER_new_graph_nodes_id).type(torch.int64)
        graph_new["isbn"].node_id =torch.tensor(ISBN_new_graph_nodes_id).type(torch.int64)
        graph_new["isbn"].x = ISBN_new_graph_isbn_x
        graph_new["user", "review", "isbn"].edge_index  = torch.tensor(ISBN_new_graph_edge_index_user_to_isbn_new).type(torch.int64)
        graph_new = T.ToUndirected()(graph_new)
        graph_new["user", "review", "isbn"].edge_label_index = torch.tensor(new_user_alg_chosen_books_edge_label_index).type(torch.int64)
        
        
        
        
        
        
        predictions = self.gnn_loaded(graph_new)
        return (predictions)
       