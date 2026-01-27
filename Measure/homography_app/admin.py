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


admin.site.register(PetVideos, PetVideosAdmin)
admin.site.register(SingletonHomographicMatrixModel)
admin.site.register(CalibrationDataModel)
