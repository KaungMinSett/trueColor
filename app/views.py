from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

from django.http import HttpResponseRedirect
from app.services import *
from app.analysis import *
from django.contrib import messages

from mycolor.models import Image

# Create your views here.

@login_required(login_url="/login")
def getColor(request):
    if request.method == "POST":
        image = request.FILES["image"]
    
        season = classify_skin_tone(image) # call to api
        print(season)

        if season == "No face detected.":
            return render(request, 'app/getColor.html', {
                'error_message': "No face detected in the image. Please try again with a different image."
            })

        if season == "Neutral":

            return render(request, 'app/getColor.html', {

                'error_message': f"Unable to determine a precise color season. The analysis returned: {season} try with different photo",
                
            })


        Image.objects.create(image=image, user=request.user, season=season) # save image to database

       
        image = Image.objects.last()

        season_name, personal_colors, lipstick_colors = get_season_and_colors(season) # return colors for season

        seasonInfo = SeasonInfo.objects.get(season__name=season_name) # get season info
    
      

        return render(request, 'app/colorResult.html',{
            'image_url': image.image.url,

            'seasonInfo': seasonInfo,

            "season_name": season_name,
            "personal_colors": personal_colors,
            "lipstick_colors": lipstick_colors
        })
        
    return render(request, 'app/getColor.html')

@login_required(login_url="/login")
def myColor(request, id):
    image = Image.objects.get(id=id) # get clicked record from database
    season_name, personal_colors, lipstick_colors = get_season_and_colors(image.season) # return colors for season
    seasonInfo = SeasonInfo.objects.get(season__name=season_name) # get season info
    

    return render(request, 'app/colorResult.html',
                  {
                        "image_url": image.image.url,
                        "seasonInfo": seasonInfo,
                        "season_name": season_name,
                        "personal_colors": personal_colors,
                        "lipstick_colors": lipstick_colors
                    
                  })

@login_required(login_url="/login")
def viewHistory(request):
    images = Image.objects.filter(user=request.user).order_by('-created_at') # get all images by latest from database for logged in user 
    return render(request, 'app/history.html', {
        "images": images
    })

@login_required(login_url="/login")
def viewStyle(request):
    return render(request, 'app/getStyle.html')

@login_required(login_url="/login")
def viewSetting(request):
    if request.method == "POST":
        user = request.user
        user.username = request.POST["username"]
        user.email = request.POST["email"]
        user.save() # update user details
        messages.success(request, "Profile updated successfully")

    username = request.user.username
    email = request.user.email
    
    return render(request, 'app/setting.html',{
            "username": username,
            "email": email

        })

@login_required(login_url="/login")
def deleteImage(request, id):
    image = Image.objects.get(id=id) # get image id  from database to delete
    image.delete()
    return redirect('history') # redirect to history page

@login_required(login_url="/login")
def viewColorPalette(request):
    return render(request, 'app/colorPalette.html')