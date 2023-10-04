from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse
import numpy as np
from .data_class import Data_load
import json
import pandas
data = Data_load()
data.preprocess_data()
#preds = data.run_recommendation()

def index(request):
    books_images = data.load_books_images().to_json(orient ='records')
    books_images = json.loads(books_images)
    template = loader.get_template('base.html')
    context = {"books": books_images}
    return HttpResponse(template.render(context, request))


def simple_function(request):
    books_images = data.books_images.to_json(orient ='records')
    books_images = json.loads(books_images)

    image_isbn = request.GET["link"]
    if len(list(data.choice_dict.keys())) < 5:
        data.add_choice2dict(image_isbn)

    choice = data.get_dict_from_choice()  
    
    template = loader.get_template('base.html')
    context = {
        "books": books_images,
        "choice": choice,
        }
    return HttpResponse(template.render(context, request))
    #return HttpResponse(""" <html><script>window.location.replace('/');</script></html>""")

def reset_function(request):
    books_images = data.books_images.to_json(orient ='records')
    books_images = json.loads(books_images)
    
    data.reset_choice2dict()
    choice = []
    
    template = loader.get_template('base.html')
    context = {
        "books": books_images,
        "choice": choice,
        }
    return HttpResponse(template.render(context, request))

def remove_function(request):
    books_images = data.books_images.to_json(orient ='records')
    books_images = json.loads(books_images)
    image_isbn = request.GET["cisbn"]
    data.remove_choice2dict(image_isbn)
    choice = data.get_dict_from_choice() 
    
    template = loader.get_template('base.html')
    context = {
        "books": books_images,
        "choice": choice,
        }
    return HttpResponse(template.render(context, request))

def predict_function(request):
    #print("predict")
    books_images = data.books_images.to_json(orient ='records')
    books_images = json.loads(books_images)
    choice = data.get_dict_from_choice()
    recomm_l = []
    recomm_s = [data.mapping_item[ c["isbn"]] for c in choice]
    alg_books = data.alg_isbn(recomm_s)
    preds = data.run_recommendation(recomm_s,alg_books)
    pred_array = preds.detach().numpy()
    b = np.where(pred_array > 0.7, True, False)
    final_recommendations = np.array(alg_books)[b]
    final_recommendations = np.argsort(final_recommendations)[-5:]
    rec_ids = []
    for rec in final_recommendations:
        for k, v in data.mapping_item.items():
            if rec == v:
               rec_ids.append({"isbn": k, "isbn_i": rec, "urls": data.get_bookurl_by_isbn(k)})

    template = loader.get_template('base.html')
    context = {
        "books": books_images,
        "choice": choice,
        "recommendation": rec_ids,
        }
    return  HttpResponse(template.render(context, request))

def load_more(request):
    #Hier muss ich was ändern mit den Wertens
    books_images = data.books_images.to_json(orient ='records')
    books_images = json.loads(books_images)

    choice = data.get_dict_from_choice()  
    template = loader.get_template('base.html')
    context = {
        "books": books_images,
        "choice": choice,
        }
    return HttpResponse(template.render(context, request))
    
def load_book_page(request):
    book_isbn = request.GET["recome_book_isbn"]
    print(f'book_isbn')
    book_url = data.get_Lbookurl_by_isbn(book_isbn)
    author, pub_year, pub = data.get_author_by_isbn(book_isbn)
    book_data = {"title" : data.get_title_by_isbn(book_isbn), 
                 "isbn": book_isbn, 
                 "url": book_url, 
                 "author": author,
                 "pub_year":pub_year ,
                 "pub": pub,} 
    
    template = loader.get_template('book.html')
    context = {
        "book": book_data
        }
    return HttpResponse(template.render(context, request))
    
