from django.shortcuts import render, redirect
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.db import IntegrityError
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone
from django.contrib import messages
from .models import CustomUser
from .signals import create_token

# Create your views here.
def index(request):
    return render(request, 'index.html')

def home(request):
    return render(request, 'home.html')



def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]

        # Ensure password matches confirmation
        password = request.POST["password"]
        confirmation = request.POST["confirmation"]
        if password != confirmation:
            return render(request, "auctions/register.html", {
                "message": "Passwords must match."
            })

        # Attempt to create new user
        try:
            user = CustomUser.objects.create_user(username, email, password)
            user.save()
            create_token(CustomUser, user, created=True)
            messages.success(request, "Account created successfully! An OTP was sent to your Email")
        except IntegrityError:
            return render(request, "auth/register.html", {
                "message": "Username already taken."
            })

        return redirect("verify-email", username=request.POST['username'])
    else:
         return render(request, 'auth/register.html')

def verify_email(request, username):
    user = CustomUser.objects.get(username=username)
    if request.method == "POST":
        otp = request.POST['otp']
        user_otp = user.otps.last()
        if otp == user_otp.otp_code:

            if user_otp.otp_expires_at < timezone.now():
                return render(request, 'auth/verify_email.html', {
                    'message': 'OTP has expired'
                })
            else:
                user.is_active = True
                user.save()
                return redirect('login')
        else:
            messages.warning(request, "Invalid OTP entered, enter a valid OTP!")
            return render(request, 'auth/verify_email.html', {

            })
    else:
        return render(request, 'auth/verify_email.html', {
            'username': username
        })

def login(request):
    if request.method == "POST":
         # Attempt to sign user in
        email = request.POST["email"]
        password = request.POST["password"]


        user = authenticate(request, email=email, password=password)

         # Check if authentication successful
        if user is not None:
            auth_login(request, user)
            
            return redirect('getColor')
        else:
            return render(request, "auth/login.html", {
                "message": "Invalid username and/or password."
            })

    return render(request, 'auth/login.html')

def logout_view(request):
    auth_logout(request)
    return HttpResponseRedirect(reverse("index"))

def searchEmail(request):
    return render(request, 'auth/searchEmail.html')

def forgotPassword(request):
    return render(request, 'auth/forgotPassword.html')