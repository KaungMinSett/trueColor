from app.models import *


def get_colors_for_season(season_name):
    try:
        season = Season.objects.get(name__iexact=season_name)
    except Season.DoesNotExist:
        return None, None

    personal_category = Category.objects.get(name="Personal Color")
    lipstick_category = Category.objects.get(name="LipStick")

    personal_colors = Color.objects.filter(season=season, category=personal_category)[:14]
    lipstick_colors = Color.objects.filter(season=season, category=lipstick_category)[:5]


    return personal_colors, lipstick_colors

def get_season_and_colors(season):
    season_name = season # call to openai api
    personal_colors, lipstick_colors = get_colors_for_season(season_name)

    if personal_colors is None or lipstick_colors is None:
        return None, None, None


    return season_name, personal_colors, lipstick_colors


# print(get_season_and_colors("True Spring"))

