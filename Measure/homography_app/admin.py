from django.contrib import admin, messages
from django.urls import path
from django.shortcuts import redirect, get_object_or_404
from django.utils.html import format_html

from .models import PetVideos, SingletonHomographicMatrixModel, CalibrationDataModel


class PetVideosAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "participant_name",
        "pet_type",
        "distance",
        "duration",
        "process_button",
    )
    
    # Add search fields
    search_fields = (
        "name",
        "participant_name",
    )
    
    # Add filters
    list_filter = (
        "pet_type",
        "is_video_processed",
        "uploaded_at",
    )

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "process/<int:pk>/",
                self.admin_site.admin_view(self.process_video),
                name="petvideos-process",
            ),
        ]
        return custom_urls + urls

    def process_button(self, obj):
        return format_html(
            '<a class="button" href="process/{}/">Process</a>',
            obj.pk
        )

    process_button.short_description = "Process Video"

    def process_video(self, request, pk):
        obj = get_object_or_404(PetVideos, pk=pk)
        obj.run_processing()

        messages.success(
            request,
            f"Processing started for {obj.name}"
        )

        return redirect(request.META.get("HTTP_REFERER"))

class CalibrationDataModelAdmin(admin.ModelAdmin):
    list_display = (
        "test_id",
        "assessment_id",
        "origin_x",
        "origin_y",
        "unit_distance",
        "use_homograph",
    )
admin.site.register(PetVideos, PetVideosAdmin)
admin.site.register(SingletonHomographicMatrixModel)
admin.site.register(CalibrationDataModel, CalibrationDataModelAdmin)
