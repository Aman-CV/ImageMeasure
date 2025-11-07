from django.contrib import admin
from .models import PetVideos, SingletonHomographicMatrixModel
# Register your models here.
class PetVideosAdmin(admin.ModelAdmin):
    list_display = ("name", "participant_name", "pet_type", "distance", "duration")
admin.site.register(PetVideos, PetVideosAdmin)
admin.site.register(SingletonHomographicMatrixModel)