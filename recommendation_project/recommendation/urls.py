from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("simple_function", views.simple_function),
    path("reset_function", views.reset_function),
    path("remove_function", views.remove_function),
    path("predict_function", views.predict_function),
    path("load_book_page", views.load_book_page),
    path("load_more", views.load_more),
]