from django.shortcuts import render, redirect
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.contrib.auth.tokens import default_token_generator
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone
from django.contrib import messages
from .models import CustomUser
from .services import *

# Create your views here.
def index(request):
    return render(request, 'index.html')

def home(request):
    return render(request, 'home.html')

def aboutUs(request):
    return render(request, 'aboutUs.html')

def contactUs(request):
    return render(request, 'contactUs.html')

# register new user
def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        if username == "" or email == "" or username.strip() == "" or email.strip() == "":
            messages.warning(request, "Please fill all fields")
            return render(request, "auth/register.html", {
                    
                })

        # Ensure password matches confirmation
        password = request.POST["password"]
        confirmation = request.POST["confirmation"]
        if password != confirmation:
            messages.warning(request, "Passwords must match.")
            return render(request, "auth/register.html", {
                
            })

        # Attempt to create new user
        try:
            user = CustomUser.objects.create_user(username, email, password)
            user.save()
            create_token(CustomUser, user, created=True)
        except IntegrityError:
            messages.warning(request, "Username or email address is already registered.")
            return render(request, "auth/register.html", {

            })
        user_uuid = user.uuid

        return redirect('verify-email', user_uuid=user_uuid)
    else:
         return render(request, 'auth/register.html')

# verify user email
def verify_email(request, user_uuid):
    user = CustomUser.objects.get(uuid=user_uuid)

    if request.method == "POST":
        otp = request.POST['otp']
        user_otp = user.otps.last() # get the last otp sent to the user
        if otp == user_otp.otp_code:

            if user_otp.otp_expires_at < timezone.now(): # check if otp has expired
                messages.warning(request, "OTP has expired, request a new OTP")
                return render(request, 'auth/verify_email.html', 
                 {'user_uuid': user_uuid}

                )
            else:
                user.is_active = True
                user.save()
                return redirect('login')
        else:
            messages.warning(request, "Invalid OTP entered, enter a valid OTP!")
            return render(request, 'auth/verify_email.html',{'user_uuid': user_uuid})
    else:
        messages.success(request, "Account created successfully! An OTP was sent to your Email")
        return render(request, 'auth/verify_email.html', {'user_uuid': user_uuid})


def deleteAccount(request):
    request.user.delete()
    return HttpResponseRedirect(reverse("index"))

#login user
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
            messages.warning(request, "Invalid email or password.")
            return render(request, "auth/login.html", {

            })

    return render(request, 'auth/login.html')



#password reset
def forget_password(request):
    if request.method == "POST":
        req_email = request.POST['email']

        if get_user_model().objects.filter(email=req_email).exists():

            user = CustomUser.objects.get(email=req_email) # get the user
            reset_link = generate_reset_link(user) # generate unique reset password link 
            send_reset_link(user, reset_link) # send the link to the user email address
            messages.success(request, "A reset link has been sent to your email address")

       
        else:
            messages.error(request, "This email is not registered")
            return redirect("forgotPassword")
    
    return render(request, 'auth/forgotPassword.html')
             






# logout
def logout_view(request):
    auth_logout(request)
    return HttpResponseRedirect(reverse("index"))


# set new password
def setPassword(request, uidb64, token):
    user = is_valid_uidb64(uidb64)
    if user is not None and default_token_generator.check_token(user, token):
        if request.method == 'POST':
            new_password = request.POST['newPassword']
            confirm_password = request.POST['confirmation']
            if new_password != confirm_password:
                messages.error(request, "Passwords do not match")
                return render(request, 'auth/setPassword.html', {
                'uidb64': uidb64,
                'token': token,
               

                })
            user.set_password(new_password)
            user.save()
            
            return redirect(login)



        
        else:
            return render(request, 'auth/setPassword.html', {
            'uidb64': uidb64,
            'token': token

        })
    else:
        return redirect('forgotPassword') 



def getColor(request):
    return render(request, 'app/getColor.html')