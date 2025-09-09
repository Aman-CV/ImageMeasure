from django.contrib import admin
from .models import PetVideos
# Register your models here.
class PetVideosAdmin(admin.ModelAdmin):
    list_display = ("name", "participant_name", "pet_type")
admin.site.register(PetVideos, PetVideosAdmin)